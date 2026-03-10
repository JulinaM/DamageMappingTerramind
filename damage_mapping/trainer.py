import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from datasets.DataLoader import Train_Val_Loader
from models.utils import weights, calc_batch_metrics, calc_epoch_metrics, move_to_device, set_seeds, save_checkpoint
from models.Decoder_UNet2D import UNet2D
from models.Encoder_TerraMind import TerraMindEncoder
from logger import init_logger

device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG_DIR = "/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2/configs/old_configs/train_val"


def _get_output_dir() -> str:
    hc = HydraConfig.get()
    if "sweep" in hc:
        return os.path.join(hc["sweep"]["dir"], hc["sweep"]["subdir"])
    return hc["runtime"]["output_dir"]


@hydra.main(version_base = "1.2", config_path = CONFIG_DIR , config_name = 'config')
def main(cfg: DictConfig):
    log_dir = _get_output_dir()
    writer= SummaryWriter(log_dir = log_dir)
    logger = init_logger(
        filepath=os.path.join(log_dir, "trainer.log"),
        rank=0,
        add_rank_suffix=False,
        use_console=False,
    )
    logger.info("Trainer started")
    logger.info("Output directory: %s", log_dir)
    logger.info("Device: %s", device)
    logger.info("Model config: %s", OmegaConf.to_container(cfg.model, resolve=True))
    logger.info("Train loader config: %s", OmegaConf.to_container(cfg.train_loader, resolve=True))
    logger.info("Validation loader config: %s", OmegaConf.to_container(cfg.validation_loader, resolve=True))

    try:
        #set seeds for replicability when using torch
        set_seeds(cfg.model.seed)

        #getting data in usable format from config paths
        train_modalities = {
            name: (paths.before, paths.after) for name, paths in cfg.train_loader.modalities.items()}
        val_modalities = {
            name: (paths.before, paths.after) for name, paths in cfg.validation_loader.modalities.items()}

    # ------------------------------Loading in data & setting up model  --------------------------------------- #
        logger.info("Building train dataset")
        train_data = Train_Val_Loader(modalities = train_modalities,
            label_dir = cfg.train_loader.label_dir,
            split = 'train',
            num_augmentations = cfg.train_loader.num_augmentations,
            patch_size = cfg.train_loader.patch_size,
            stride = cfg.train_loader.stride,
            preload = cfg.train_loader.preload)

        train_dataloader = DataLoader(train_data,
                                      batch_size = cfg.train_loader.batch_size,
                                      shuffle = cfg.train_loader.shuffle,
                                      num_workers = cfg.train_loader.num_workers)
        logger.info("Train patches: %d", len(train_data))

        logger.info("Building validation dataset")
        val_data = Train_Val_Loader(
            modalities = val_modalities,
            label_dir = cfg.validation_loader.label_dir,
            split = 'validation',
            patch_size =  cfg.validation_loader.patch_size,
            stride =  cfg.validation_loader.stride,
            preload = cfg.validation_loader.preload)
        val_dataloader = DataLoader(val_data,
                                    batch_size = cfg.validation_loader.batch_size,
                                    shuffle = cfg.validation_loader.shuffle,
                                    num_workers = cfg.validation_loader.num_workers)
        logger.info("Validation patches: %d", len(val_data))

        # Set up model configurations
        if cfg.model.apply_weight_loss:
            inverse, pixels = weights(
                train_dataloader,
                num_classes=cfg.model.num_classes,
                ignore_index=cfg.model.ignore_index,
                device=device,
            )
            if cfg.model.weight_type == 'pixels':
                criterion = nn.CrossEntropyLoss(weight=pixels, ignore_index=cfg.model.ignore_index)
            elif cfg.model.weight_type == 'inverse':
                criterion = nn.CrossEntropyLoss(weight=inverse, ignore_index=cfg.model.ignore_index)
            else:
                raise ValueError(f"Invalid weight_type: {cfg.model.weight_type}")
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
            logger.info("Fine-tuning encoder + decoder")
        else:
            optimizer = optim.Adam(decoder.parameters(), lr=cfg.model.learning_rate)
            encoder_mode = encoder.eval
            logger.info("Training decoder only (encoder frozen)")

    # ---------------------------------------- Training Loop ---------------------------------------------- #
        best_val_loss = float("inf")
        logger.info("Starting training for %d epoch(s)", cfg.model.num_epochs)
        for epoch in range(cfg.model.num_epochs):
            # set the encoder to train/eval as specified by fine-tune config above
            encoder_mode()
            decoder.train()

            running_train_loss = 0.0
            running_val_loss = 0.0
            TP = FP = FN = TN = 0.0

            for x, y in train_dataloader:
                x = move_to_device(x, device)
                y = y.to(device)

                z_before, z_after = encoder(x["before"]), encoder(x["after"])
                z_differenced = [after - before for before, after in zip(z_before, z_after)]
                logits = decoder(z_differenced)

                train_loss = criterion(logits, y)
                sz_batch = next(iter(x["before"].values())).size(0)
                running_train_loss += train_loss.item() * sz_batch

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            epoch_loss = running_train_loss / len(train_data)

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
            epoch_metrics = calc_epoch_metrics(TP, FP, FN, TN)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | IoU=%.4f | Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f",
                epoch + 1,
                cfg.model.num_epochs,
                epoch_loss,
                val_loss,
                epoch_metrics["IoU"],
                epoch_metrics["Accuracy"],
                epoch_metrics["Precision"],
                epoch_metrics["Recall"],
                epoch_metrics["F1"],
            )

            # This will ensure we save the key variables to a checkpoint file in the multirun output folders
            # It will overwrite and only save results, model checkpoints & configs for the best model (according to validation Cross Entropy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, cfg, 
                                save_dir=os.path.join(log_dir, "checkpoints")
)
                logger.info("Saved new best checkpoint at epoch %d with val_loss=%.4f", epoch + 1, val_loss)

            # Saving key metrics to look at in Tensorboard
            # To see tensorboard, type following into command line: 'tensorboard --logdir=./multirun'
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Metrics/IoU", epoch_metrics["IoU"], epoch)
            writer.add_scalar("Metrics/Accuracy", epoch_metrics["Accuracy"], epoch)
            writer.add_scalar("Metrics/Precision", epoch_metrics["Precision"], epoch)
            writer.add_scalar("Metrics/Recall", epoch_metrics["Recall"], epoch)
            writer.add_scalar("Metrics/F1", epoch_metrics["F1"], epoch)
        logger.info("Training completed successfully")
    except Exception:
        logger.exception("Trainer failed")
        raise
    finally:
        writer.close()
        logger.info("Closed TensorBoard writer")

if __name__ == "__main__":
    main()
