#patches + location tracking multimodal
from torch.utils.data import Dataset
import rasterio as rio
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
import random, math
from functools import lru_cache
import warnings
import os

from models.utils import standardize, RandomFlipPair, RandomRotationPair

INPUT_DIR = Path("/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2/data/input/")

class Train_Val_Loader(Dataset):
    def __init__(
        self,
        modalities: dict,
        label_dir: str,
        split: str = 'train',
        num_augmentations: int = 0,
        patch_size: int = 224,
        stride: int = 224,
        preload: bool = True
    ):
        """
        Loads in multimodal datasets to train the TerraMind encoder and the defined decoder model. Will be used for train & validation data.
        
        Outputs x patches per each image.
        The x data outputted is in a dictionary containing the tensor for each patch per modality.
        Y data is output as one tensor patch.

        Args:
            modalities (dict): {"S2": ("path/to/S2_before", "path/to/S2_after"), "S1": ("path/to/S1_before", "path/to/S1_after")} # can change modalities
            label_dir (str): Path to directory of label files
            split (str): "train" or "validation", test is handled elsewhere
            num_augmentations (int): if x=0, it will use the images as they are patched. If x=1, it will manipulate the patched images first. If x>1 it will randomly edit each patch x amount of times, running x * num image amount of times
            patch_size (int): Size of extracted patch. TM is trained on 224, and I recommend using that setting
            stride (int): The tile image will likely be >224x224, so when we do patching, if you choose less than 224, overlap=224-stride.
            preload (bool): If True, load all rasters into memory once. Test if machine is capable, as this will speed up process
        """
        if split not in ["train", "validation"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'validation'.") 
        
        if stride > 224:
            warnings.warn(f"Caution: with stride > 224, some pixels may not be seen by the model.",UserWarning)

        self.modalities = modalities
        print(str(INPUT_DIR / label_dir))
        self.label_files = sorted((INPUT_DIR / label_dir).glob("*.tif"))
        # self.label_files = sorted(Path(label_dir).glob("*.tif"))
        self.split = split
        self.num_augmentations = num_augmentations
        self.patch_size = patch_size
        self.stride = stride
        self.preload = preload
        self.size_helper = 1 if self.num_augmentations == 0 else self.num_augmentations

        # Verify modalities & labels have same number of images
        label_len = len(self.label_files)
        for name, (before_dir, after_dir) in modalities.items():
            before_files = sorted((INPUT_DIR / before_dir).glob("*.tif"))
            after_files = sorted((INPUT_DIR / after_dir).glob("*.tif"))
            print("After: ", str((INPUT_DIR / after_dir)))
            print("Before: ", str((INPUT_DIR / before_dir)))
            if len(before_files) != len(after_files):
                raise ValueError(f"Modality {name} before/after count mismatch")
            if len(before_files) != label_len:
                raise ValueError(f"Modality {name} count differs from labels")
        self.num_images = label_len

        print("Total labels: ", self.num_images)

        # Augmentation policy
        self.augment = None
        if (self.split == "train") & (self.num_augmentations>0):
            self.augment = transforms.Compose([
                RandomFlipPair(),
                RandomRotationPair()])
        
        # Optionally preload images into RAM (saves time if dataset fits memory)
        self.cache = {}
        if preload:
            self._preload_images()

        # Precompute patch coordinates for all images
        self.index_map = self._build_patch_index_map()
        print ("Init dataloader successfully.")

# # ------------------- utility functions
    
    # Pad overall tile image to fit multiples of 224 for H & W
    def _pad_image(self, img):
        if not img.is_floating_point():
            img = img.float()

        _, H, W = img.shape
        pad_h = (math.ceil((H - self.patch_size) / self.stride) * self.stride + self.patch_size) - H
        pad_w = (math.ceil((W - self.patch_size) / self.stride) * self.stride + self.patch_size) - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Padding using 0 (black), but reflect may be interesting, it will mirror the edges.
        # padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), padding_mode =  "reflect")
        padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode =  "constant", value = 0)

        return padded, pad_top, pad_left

    # get coordinates of top left of patches in grid
    def _extract_patch_coords(self, img):
        _, H, W = img.shape
        coords = []
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                coords.append((y, x))
        return coords

# # ------------------- Image loading & caching
    # Load in data and clean up NA values
    @lru_cache(maxsize=None)
    def _load_tif(self, path):
        with rio.open(path) as src:
            arr = src.read().astype('float32')
            nodata = src.nodata
        arr = torch.from_numpy(arr)
        # Replace NaN or NoData values
        if nodata is not None:
            arr[arr == nodata] = 0.0
        # double check to replace any NAs after accounting for rio defined non-vals
        arr = torch.nan_to_num(arr, nan=0.0)
        return arr

    # Preload images to save run time
    def _preload_images(self):
        for idx in range(self.num_images):
            image_dict = {}
            for name, (before_dir, after_dir) in self.modalities.items():
                before_file = sorted((INPUT_DIR / before_dir).glob("*.tif"))[idx]
                after_file = sorted((INPUT_DIR / after_dir).glob("*.tif"))[idx]
                image_dict[name] = {
                    "before": self._load_tif(before_file),
                    "after": self._load_tif(after_file)
                }
            label_file = self.label_files[idx]
            image_dict["label"] = self._load_tif(label_file).squeeze()
            self.cache[idx] = image_dict

# # ------------------- Patch indexing
    # create patch grid
    def _build_patch_index_map(self):
        index_map = []
        # Use the first modality per img index as reference for shape
        first_mod = next(iter(self.modalities.keys()))
        first_before_dir, _ = self.modalities[first_mod]
        for i in range(self.num_images):
            ref_path = sorted((INPUT_DIR / first_before_dir).glob("*.tif"))[i]
            with rio.open(ref_path) as src:
                #only dummy size is necessary for grid
                dummy = torch.zeros((1, src.height, src.width), dtype=torch.float32)
                dummy, _, _ = self._pad_image(dummy)
                coords = self._extract_patch_coords(dummy)
                for c in coords:
                    index_map.append((i, c))
        return index_map

# # ------------------- Dataset interface
    def __len__(self):
        return len(self.index_map) * self.size_helper #Counts each patch for each image tile. Then multiplied by number of augmentations

    def __getitem__(self, index):
        # loop over each image index x times (num augmentations)
        img_index = index // self.size_helper
        i, (y, x) = self.index_map[img_index]

        # Load data (either cached or on demand)
        if self.preload:
            data = self.cache[i]
        else:
            data = {}
            for name, (before_dir, after_dir) in self.modalities.items():
                before_file = sorted((INPUT_DIR / before_dir).glob("*.tif"))[i]
                after_file = sorted((INPUT_DIR / after_dir).glob("*.tif"))[i]
                data[name] = {
                    "before": self._load_tif(before_file),
                    "after": self._load_tif(after_file)
                }
            label_file = self.label_files[i]
            data["label"] = self._load_tif(label_file).squeeze()

        before_dict = {}
        after_dict = {}

        for name in self.modalities:
            x_before, pad_top, pad_left = self._pad_image(data[name]["before"])
            x_after, _, _ = self._pad_image(data[name]["after"])
            patch_before = x_before[:, y:y+self.patch_size, x:x+self.patch_size]
            patch_after  = x_after[:, y:y+self.patch_size, x:x+self.patch_size]

            # Standardize per modality so gradients can run cleanly
            patch_before = standardize(patch_before, dim=1)
            patch_after  = standardize(patch_after, dim=1)

            before_dict[name] = patch_before.float()
            after_dict[name]  = patch_after.float()

        # Label patch
        y_full, _, _ = self._pad_image(data["label"].unsqueeze(0))
        y_patch = y_full[0, y:y+self.patch_size, x:x+self.patch_size]

        sample = {
            "before": before_dict,
            "after": after_dict,
            "y": y_patch
        }

        # Augmentation--Randomly flip and randomly rotate patches individually (training only)
        if self.augment and self.split == "train":
            sample = self.augment(sample)

        # After augmentation, separate inputs and label for return
        y = sample.pop("y")  # extract label tensor (now possibly transformed)
        input_sample = {"before": sample["before"], "after": sample["after"]}

        # Final types/standardization already applied per modality
        return input_sample, y.long()





# # ------------------- Set up data loader for Test ------------------- # #

class TestLoader(Dataset):
    def __init__(
        self,
        modalities: dict,
        label_dir: str,
        patch_size: int = 224, 
        stride: int = 224,
    ):
        """
        Loads in multimodal datasets and a pretrained TM encoder and a decoder, of which you selected
        
        Outputs patches x per each image. 
        The x data is a dictionary containing tensors for each modality.
        The y data is a tensor.
        Records the index of the original imagge alongside top left coordinate of patch.
        Records the padding of all sides for each patch. Padding, indices & coordinates will later assist with reconstruction of original image tile.

        Args:
            modalities (dict): {"S2": ("path/to/S2_before", "path/to/S2_after"), "S1": ("path/to/S1_before", "path/to/S1_after")}. Can change modalities
            label_dir (str): Path to directory of label files
            patch_size (int): Size of extracted patch. TM is trained on 224, and I recommend using that setting. In any case it should match trained model
            stride (int): The tile image will likely be >224x224, so when we do patching, if you choose less than 224, overlap=224-stride.
        """

        self.modalities = modalities
        self.patch_size = patch_size
        self.stride = stride
    
        # Resolve relative paths from the project input root used by train/val loader.
        self.label_dir = self._resolve_dir(label_dir)
        self.modality_dirs = {
            name: (self._resolve_dir(before_dir), self._resolve_dir(after_dir))
            for name, (before_dir, after_dir) in modalities.items()
        }

        # Check counts across labels and each modality pair.
        self.label_files = sorted(self.label_dir.glob("*.tif"))
        label_len = len(self.label_files)
        if label_len == 0:
            raise ValueError(f"No test labels found in: {self.label_dir}")

        for name, (before_dir, after_dir) in self.modality_dirs.items():
            before_files = sorted(before_dir.glob("*.tif"))
            after_files = sorted(after_dir.glob("*.tif"))
            if len(before_files) != len(after_files):
                raise ValueError(f"Modality {name} before/after count mismatch")
            if len(before_files) != label_len:
                raise ValueError(f"Modality {name} count differs from labels")
        self.num_images = label_len

        # Precompute patch coordinates for all images
        self.index_map = self._build_patch_index_map()


# # ---------------- setting up utility functions
    def _resolve_dir(self, path_like):
        path = Path(path_like)
        return path if path.is_absolute() else INPUT_DIR / path
    
    def _pad_image(self, img):
        if not img.is_floating_point():
            img = img.float()

        _, H, W = img.shape
        pad_h = (math.ceil((H - self.patch_size) / self.stride) * self.stride + self.patch_size) - H
        pad_w = (math.ceil((W - self.patch_size) / self.stride) * self.stride + self.patch_size) - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # Use positional "reflect" for compatibility
        # padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), padding_mode =  "reflect")
        padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode =  "constant", value = 0)
        return padded, (pad_left, pad_right, pad_top, pad_bottom)
    

    def _extract_patch_coords(self, img):
        _, H, W = img.shape
        coords = []
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                coords.append((y, x))
        return coords


    # clean up potential NAs
    @lru_cache(maxsize=None)
    def _load_tif(self, path):
        with rio.open(path) as src:
            arr = src.read().astype('float32')
            nodata = src.nodata
        arr = torch.from_numpy(arr)
        if nodata is not None:
            arr[arr == nodata] = 0.0
        arr = torch.nan_to_num(arr, nan=0.0)
        return arr
    
    def _build_patch_index_map(self):
        index_map = []
        # Use the first modality as shape reference
        first_mod = next(iter(self.modality_dirs.keys()))
        first_before_dir, _ = self.modality_dirs[first_mod]
        for i in range(self.num_images):
            ref_path = sorted(first_before_dir.glob("*.tif"))[i]
            with rio.open(ref_path) as src:
                dummy = torch.zeros((1, src.height, src.width), dtype=torch.float32)
                dummy, _ = self._pad_image(dummy)
                coords = self._extract_patch_coords(dummy)
                for c in coords:
                    index_map.append((i, c))
        return index_map
    
    # # ---------------- Starting Dataset Creation
    def __len__(self):
        return len(self.index_map) # counts each patch for all images processed

    def __getitem__(self, index):
        img_index = index
        i, (coord_y, coord_x) = self.index_map[img_index]

        data = {}
        meta = None
        for name, (before_dir, after_dir) in self.modality_dirs.items():
            before_file = sorted(before_dir.glob("*.tif"))[i]
            after_file = sorted(after_dir.glob("*.tif"))[i]
            
            # Save metadata from the first file type (doesnt matter as long as same index)
            if meta is None:
                with rio.open(before_file) as src:
                    meta = src.meta.copy()

            data[name] = {
                "before": self._load_tif(before_file),
                "after": self._load_tif(after_file)}
            
            

        before_dict = {}
        after_dict = {}
        for name in self.modalities:
            x_before, (pad_left, pad_right, pad_top, pad_bottom) = self._pad_image(data[name]["before"])
            x_after, _ = self._pad_image(data[name]["after"])
            patch_before = x_before[:, coord_y:coord_y+self.patch_size, coord_x:coord_x+self.patch_size]
            patch_after  = x_after[:, coord_y:coord_y+self.patch_size, coord_x:coord_x+self.patch_size]

            # Standardize per modality
            patch_before = standardize(patch_before, dim=1)
            patch_after  = standardize(patch_after, dim=1)

            before_dict[name] = patch_before.unsqueeze(0).float()
            after_dict[name]  = patch_after.unsqueeze(0).float()

        input_sample = {
        "before": before_dict,
        "after": after_dict}
        
        # Label patch
        label_file = self.label_files[i]
        label_tif = self._load_tif(label_file)
        y_full, _ = self._pad_image(label_tif)
        y_patch = y_full[0, coord_y:coord_y+self.patch_size, coord_x:coord_x+self.patch_size]
        y_patch = y_patch.long()

        return input_sample, y_patch, (i, coord_y, coord_x), (pad_left, pad_right, pad_top, pad_bottom), meta
