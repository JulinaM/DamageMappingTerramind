#stop `albumentations` info, TerraTorch relies on 1.4
import logging
logging.getLogger("albumentations").setLevel(logging.ERROR)

from geography.damage_mapping_terramind.V2.damage_mapping.datasets.DataLoader import Train_Val_Loader
from geography.damage_mapping_terramind.V2.damage_mapping.models.utils import weights, calc_batch_metrics, calc_epoch_metrics, move_to_device, set_seeds, save_checkpoint
from damage_mapping.models.Decoder_UNet2D import UNet2D
from damage_mapping.models.Encoder_TerraMind import TerraMindEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

class Trainer:
     def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module | None,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        ckpt_dir: pathlib.Path | str,
        device: torch.device,
        distributed: bool,
        cudnn_backend: bool,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
    ):
        self.rank = int(os.environ["RANK"])
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.n_epochs = n_epochs
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.distributed = distributed
        self.cudnn_backend = cudnn_backend
        self.use_wandb = use_wandb
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.best_metric_key = best_metric_key
        
        self.logger = logging.getLogger()
        self.training_stats = {
            name: RunningAverageMeter(length=len(self.train_loader))
            for name in ["loss", "data_time", "batch_time", "eval_time"]
        }
        self.best_metric = -1
        self.best_metric_comp = operator.gt
        self.num_predictands = len(self.train_loader.dataset.predictands)

        assert precision in [
            "fp32",
            "fp16",
            "bfp16",
        ], f"Invalid precision {precision}, use 'fp32', 'fp16' or 'bfp16'."
        
        torch.backends.cudnn.enabled = self.cudnn_backend
        
        self.enable_mixed_precision = precision != "fp32"
        self.precision = torch.float16 if (precision == "fp16") else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_mixed_precision)
        self.start_epoch = 0

        if self.use_wandb:
            import wandb
            self.wandb = wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
 
@hydra.main(version_base = "1.2", config_path = "configs/train_val" , config_name = 'config')
def main(cfg: DictConfig):
    hc = HydraConfig.get()
    dir, subdir = hc['sweep']['dir'], hc['sweep']['subdir']
    log_dir = os.path.join(dir, subdir)
    writer= SummaryWriter(log_dir = log_dir) 

    #set seeds for replicability when using torch
    set_seeds(cfg.model.seed)
    
    #getting data in usable format from config paths
    train_modalities = {
        name: (paths.before, paths.after)
        for name, paths in cfg.train_loader.modalities.items()}
    val_modalities = {
        name: (paths.before, paths.after)
        for name, paths in cfg.validation_loader.modalities.items()}
    
# ------------------------------Loading in data & setting up model  --------------------------------------- #    
    #set up train and validation datasets using our dataloader function
    train_data = Train_Val_Loader(modalities = train_modalities,
        label_dir = cfg.train_loader.label_dir,
        split = 'train', #probably does not need to be a param in config file
        num_augmentations = cfg.train_loader.num_augmentations,
        patch_size = cfg.train_loader.patch_size,
        stride = cfg.train_loader.stride,
        preload = cfg.train_loader.preload)
    # Torch dataloader tool for batching etc.
    train_dataloader = DataLoader(train_data, 
                                  batch_size = cfg.train_loader.batch_size,
                                  shuffle = cfg.train_loader.shuffle,
                                  num_workers = cfg.train_loader.num_workers)

    val_data = Train_Val_Loader(
        modalities = val_modalities,
        label_dir = cfg.validation_loader.label_dir,
        split = 'validation', #probably not necessary in configs 
        patch_size =  cfg.validation_loader.patch_size,
        stride =  cfg.validation_loader.stride, #probably could recycle from train
        preload = cfg.validation_loader.preload) #probably could recycle from train
    # Torch dataloader tool for batching etc.
    val_dataloader = DataLoader(val_data, 
                                batch_size = cfg.validation_loader.batch_size, 
                                shuffle = cfg.validation_loader.shuffle,
                                num_workers = cfg.validation_loader.num_workers)

    # Set up model configurations
    if cfg.model.apply_weight_loss:
        inverse, pixels = weights(train_dataloader, num_classes=cfg.model.num_classes, ignore_index=cfg.model.ignore_index, device = device)
        if cfg.model.weight_type == 'pixels':
            criterion = nn.CrossEntropyLoss(weight = pixels, ignore_index= cfg.model.ignore_index)
        if cfg.model.weight_type == 'inverse':
            criterion = nn.CrossEntropyLoss(weight = inverse, ignore_index= cfg.model.ignore_index)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index)
    
    encoder = TerraMindEncoder(version = cfg.model.TM_version, 
                               pretrained =  cfg.model.pretrained, 
                               modalities =  list(cfg.model.modalities))
    decoder = UNet2D(num_classes= cfg.model.num_classes)
    encoder.to(device)
    decoder.to(device)

    # set encoder to eval/train according to configs. Fine tuning will be a longer train
    if cfg.model.TM_finetune:
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.model.learning_rate)
        encoder_mode = encoder.train
    else:
        optimizer = optim.Adam(decoder.parameters(), lr=cfg.model.learning_rate)
        encoder_mode = encoder.eval

# ---------------------------------------- Training Loop ---------------------------------------------- #
    best_val_loss = float("inf")
    for epoch in range(cfg.model.num_epochs):
        # set the encoder to train/eval as specified by fine-tune config above
        encoder_mode()
        decoder.train() 
        
        # Set up values to get per epoch losses
        running_train_loss = 0.0
        running_val_loss = 0.0
        TP = FP = FN = TN = 0.0
        
        # each for statement runs over n image patches per loop. n=batch size
        for x, y in train_dataloader:
            x = move_to_device(x, device)
            y = y.to(device)
            
            # Pass before and after modalities separately. Difference to pass embeddings to decoder
            z_before, z_after = encoder(x["before"]), encoder(x["after"]) #
            z_differenced = [after - before for before, after in zip(z_before, z_after)]
            logits = decoder(z_differenced)

            # Get losses and format them to be viewed per epoch
            train_loss = criterion(logits, y)
            sz_batch = next(iter(x["before"].values())).size(0)
            running_train_loss += train_loss.item() * sz_batch

            # Set learning from loss function, this will backpropogate through the decoder, differences, then encoder. 
            # If encoder set to fine_tune = False, encoder will not train
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
                

        epoch_loss = running_train_loss / len(train_data)
        print(f"Epoch {epoch+1}/{cfg.model.num_epochs} - Train Loss: {epoch_loss:.4f}")

        # Running through validation loop to see results on out sample. Model will not learn from this data
        decoder.eval()
        encoder.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                x = move_to_device(x, device)
                y = y.to(device)

                z_before, z_after = encoder(x["before"]), encoder(x["after"])
                z_differenced = [after - before for before, after in zip(z_before, z_after)]
                logits = decoder(z_differenced)
                
                batch_val_loss = criterion(logits, y)
                sz_batch = next(iter(x["before"].values())).size(0)
                running_val_loss += batch_val_loss.item() * sz_batch

                batch_metrics = calc_batch_metrics(logits, y, ignore_index = cfg.model.ignore_index, 
                                                   positive_class = cfg.model.positive_class, negative_class = cfg.model.negative_class)
                TP, FP, FN, TN = [x + y for x, y in zip((TP, FP, FN, TN), batch_metrics)]

            
            val_loss = running_val_loss/len(val_data)
            print(f"--Validation Loss: {val_loss}")

            # This will ensure we save the key variables to a checkpoint file in the multirun output folders
            # It will overwrite and only save results, model checkpoints & configs for the best model (according to validation Cross Entropy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, cfg, 
                                save_dir=os.path.join(log_dir, "checkpoints")
)
            # Print key metrics
            epoch_metrics = calc_epoch_metrics(TP, FP, FN, TN)
            print(f'--IoU: {epoch_metrics["IoU"]:.4f}\
            - Accuracy: {epoch_metrics["Accuracy"]:.4f}\
            - Precision: {epoch_metrics["Precision"]:.4f}\
            - Recall: {epoch_metrics["Recall"]:.4f}\
            - F1: {epoch_metrics["F1"]:.4f}\n')

            # Saving key metrics to look at in Tensorboard
            # To see tensorboard, type following into command line: 'tensorboard --logdir=./multirun'
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Metrics/IoU", epoch_metrics["IoU"], epoch)
            writer.add_scalar("Metrics/Accuracy", epoch_metrics["Accuracy"], epoch)
            writer.add_scalar("Metrics/Precision", epoch_metrics["Precision"], epoch)
            writer.add_scalar("Metrics/Recall", epoch_metrics["Recall"], epoch)
            writer.add_scalar("Metrics/F1", epoch_metrics["F1"], epoch)

    writer.close()

if __name__ == "__main__":
    main()