# #General Requirements
import os
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
import rasterio as rio
import numpy as np
import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
import shutil
from hydra.core.hydra_config import HydraConfig

# #From Corresponding files in folder
from geography.damage_mapping_terramind.V2.models.Encoder_TerraMind import TerraMindEncoder
from geography.damage_mapping_terramind.V2.models.Decoder_UNet2D import UNet2D
from geography.damage_mapping_terramind.V2.utils.utils import move_to_device, calc_test_metrics, tensor_to_color_image
from geography.damage_mapping_terramind.V2.damage_mapping.datasets.DataLoader import TestLoader


# Hydra sets configs outside of this file so we can edit params without touching this file
@hydra.main(version_base = "1.2", config_path = "configs/test" , config_name = 'config') #Will need to be at least version 1.2 for Hydra sweeping
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu" # If cuda env set up correctly and using GPU, this will speed up runs massively

    # set location for outputs of results
    hc = HydraConfig.get()
    dir, subdir = hc['sweep']['dir'], hc['sweep']['subdir']
    output_dir = os.path.join(dir, subdir)
    print(output_dir)
    writer= SummaryWriter(log_dir = output_dir) 

    # Copy both the checkpoint and original configs file from trained model to output folder
    checkpoint_copy = os.path.join(output_dir, "model_checkpoint.pt")
    # config_copy = os.path.join(output_dir, "model_configs.pt")
    shutil.copy(cfg.paths.trained_model, checkpoint_copy)
    # shutil.copy(cfg.paths.copy_configs, config_copy)


    # prepare modalities paths from configs for data loader
    test_modalities = {
        name: (paths.before, paths.after)
        for name, paths in cfg.paths.modalities.items()}

    # Load in desired data using test loader
    test_data = TestLoader(
        modalities = test_modalities,
        label_dir= cfg.paths.label_dir,
        patch_size = cfg.model.patch_size, 
        stride = cfg.model.stride)
    #torch batch loader
    test_dataloader = DataLoader(test_data, batch_size = None, num_workers = cfg.model.num_workers)

    # load in decoder + encoder based on trained model path provided
    checkpoint_path = cfg.paths.trained_model
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = TerraMindEncoder(version = "terramind_v1_base", 
                                pretrained =  True, 
                                modalities =  ["S2L2A", "S1GRD"])
    decoder = UNet2D(num_classes= 3)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.to(device)
    decoder.to(device)
    ## --------------------- run model and store outputs alongside information to reconstruct original tile from the patches ----------------------- ##
    # Set both to eval as this is a test loop
    encoder.eval()
    decoder.eval()

    # Set dicts which will help us reconstruct original image from patches
    padding = {}
    metas = {}
    tile_reconstruction = defaultdict(list)
    with torch.no_grad():
        for x , y, (i, coord_y, coord_x), (pad_left, pad_right, pad_top, pad_bottom), meta in test_dataloader:
            # Use helper function on x output (dictionary) to move to device
            x = move_to_device(x, device)

            # Normal prediction
            z_before, z_after = encoder(x["before"]), encoder(x["after"])
            z_differenced = [after - before for before, after in zip(z_before, z_after)]
            
            logits = decoder(z_differenced)
            y_hat = torch.argmax(logits, dim=1)

            #storing to dictionaries to recreate original image 
            idx = i.item() if torch.is_tensor(i) else i  # convert tensor to int
            tile_reconstruction[idx].append([y_hat, y, coord_y, coord_x])

            if idx not in padding:
                padding[idx] = tuple(p.item() if torch.is_tensor(p) else int(p)
                                    for p in (pad_left, pad_right, pad_top, pad_bottom))
            
            if idx not in metas:
                metas[idx] = meta


    # --------------- Taking information patch estimates/originals and recreating the entire images------------------ #
    max_coords = {}
    for i, patches in tile_reconstruction.items():
        # convert key if it's a tensor
        idx = int(i.item()) if isinstance(i, torch.Tensor) else i
        xs = [int(p[3].item()) if isinstance(p[3], torch.Tensor) else p[3] for p in patches]
        ys = [int(p[2].item()) if isinstance(p[2], torch.Tensor) else p[2] for p in patches]
        
        # helper dict to find size of original image (including padding)
        max_coords[idx] = {"max_coord_x": max(xs), "max_coord_y": max(ys)}

    image_tiles_true = {}
    image_tiles_pred = {}

    # Empty tiles of proper size per image (including padding)
    for idx, coords in max_coords.items():
        H = coords["max_coord_y"] + cfg.model.patch_size
        W = coords["max_coord_x"] + cfg.model.patch_size
        image_tiles_true[idx] = torch.zeros((H, W), dtype=torch.float32)
        image_tiles_pred[idx] = torch.zeros((H, W), dtype=torch.float32)

    # Fill in each image tile correctly, patch by patch
    for idx, patches in tile_reconstruction.items():
        for (y_hat, y_true, coord_y, coord_x) in patches:
            coord_y = int(coord_y)
            coord_x = int(coord_x)
            image_tiles_true[idx][coord_y:coord_y+cfg.model.patch_size, coord_x:coord_x+cfg.model.patch_size] = y_true.squeeze()
            image_tiles_pred[idx][coord_y:coord_y+cfg.model.patch_size, coord_x:coord_x+cfg.model.patch_size] = y_hat.squeeze()

    # Remove padding per image (true and predicted)
    for idx, img in image_tiles_true.items():
        H, W = img.shape[-2], img.shape[-1]
        pad_left, pad_right, pad_top, pad_bottom = padding[idx]
        image_tiles_true[idx] = img[pad_top:H-pad_bottom, pad_left:W-pad_right]
    
    for idx, img in image_tiles_pred.items():
        H, W = img.shape[-2], img.shape[-1]
        pad_left, pad_right, pad_top, pad_bottom = padding[idx]
        image_tiles_pred[idx] = img[pad_top:H-pad_bottom, pad_left:W-pad_right]

    # Mask out background non-ag land (background is ignored in metric calcs etc) 
    for idx in image_tiles_pred.keys():
        real = image_tiles_true[idx]
        pred = image_tiles_pred[idx]

        # Zero-out predictions wherever ground truth is 0
        image_tiles_pred[idx] = torch.where(real == 0, torch.zeros_like(pred), pred)


    # Save as geotiffs so we can use them later through google earth etc
    tiff_dir = os.path.join(output_dir, "geotiffs")
    os.makedirs(tiff_dir, exist_ok=True)

    for idx, pred in image_tiles_pred.items():
        meta = metas[idx]
        meta_out = meta.copy()
        meta_out.update({
            "driver": "GTiff",
            "height": pred.shape[0],
            "width": pred.shape[1],
            "count": 1,
            "dtype": "uint8",
            "nodata": 0

        })

        # Convert tensor → numpy array
        pred_np = pred.cpu().numpy().astype(np.uint8)

        # Output path
        tiff_path = os.path.join(tiff_dir, f"predicted_map_{idx}_colored.tif")

        # Define color map for google earth
        color_table = {
            0: (0, 0, 0, 255), #background)
            1: (0, 255, 0, 255), #healthy
            2: (255, 0, 0, 255) } #damaged

        # Save colorized GeoTIFF
        with rio.open(tiff_path, "w", **meta_out) as dst:
            dst.write(pred_np, 1)
            dst.write_colormap(1, color_table)



    # calculate and print metrics on an image tile level
    test_metrics = calc_test_metrics(image_tiles_pred, image_tiles_true, ignore_index=0, positive_class=2, negative_class=1)
    print(test_metrics)

    # Saving metrics to tensorboard location
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for idx, metrics in test_metrics.items():
            f.write(f"\nImage {idx} metrics:\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
                writer.add_scalar(f"metrics/{k}", v, global_step=idx)
    
    # Save images to tensorboard
    # To see tensorboard, type following into command line: 'tensorboard --logdir=./test_output'
    for i in list(image_tiles_pred.keys())[:3]:
        pred = image_tiles_pred[i]
        true = image_tiles_true[i]

        pred_rgb = tensor_to_color_image(pred)
        true_rgb = tensor_to_color_image(true)

        #save the two images together so we can see them side by side
        combined = torch.cat((true_rgb, pred_rgb), dim=2)  # (3, H, 2W)
        writer.add_image(f"comparison/{i}", combined)

    writer.close()

if __name__ == "__main__":
    main()