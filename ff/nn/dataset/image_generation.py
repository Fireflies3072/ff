import torch
import numpy as np
import cv2
import random
import glob
import os
import zarr
from datasets import Image, load_dataset

from ... import cv as fcv
from .base import DatasetGeneric
from .ops import *

class ImageGenerationDatasetGeneric(DatasetGeneric):
    def __init__(self, dataset, image_size, logic_map):
        self.image_size = image_size

        # Logic registry
        self._logic_registry = {
            # Atomic operations
            'read_image': read_image,
            'bgr_to_rgb': bgr_to_rgb,
            'random_hflip': random_hflip,
            'hwc_to_chw': hwc_to_chw,
            'rescale_unit': rescale_unit,
            'rescale_signed': rescale_signed,
            'rescale_imagenet': ImagenetRescaler(),

            # Intermediate operations
            'image_to_signed': image_to_signed,

            # Advanced operations
            'image_raw': self._handle_image_raw,
            'image_file': self._handle_image_file
        }

        # Initialize dataset
        super().__init__(dataset, logic_map)
    
    def _handle_image_raw(self, image_data):
        if isinstance(image_data, dict) and 'bytes' in image_data:
            buffer = np.frombuffer(image_data['bytes'], np.uint8)
            image_data = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        else:
            image_data = np.array(image_data)
        return self._op_common_image(image_data)

    def _handle_image_file(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {path}")
        return self._op_common_image(image)
    
    def _op_common_image(self, image):
        # Convert to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # Regular operations
        image = fcv.resize_cover(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        image_tensor = image_to_signed(image)
        return image_tensor

class ImageGenerationHfDataset(ImageGenerationDatasetGeneric):
    def __init__(self, hf_dataset, image_size, logic_map=None, **kwargs):
        # Load dataset
        dataset = hf_dataset

        # Default handler map {'image': 'image_raw'}
        if logic_map is None:
            logic_map = {}
            if 'image' in dataset.column_names:
                logic_map['image'] = 'image_raw'
        
        # Disable image decoding if needed
        for key, logic in logic_map.items():
            if logic == 'image_raw':
                dataset = dataset.cast_column(key, Image(decode=False))
        
        # Initialize dataset
        super().__init__(dataset, image_size, logic_map)

class ImageGenerationOnlineHfDataset(ImageGenerationHfDataset):
    def __init__(self, hf_id, image_size, split='all', logic_map=None, **kwargs):
        self.hf_id = hf_id
        self.split = split

        # Load dataset
        dataset = load_dataset(hf_id, split=split, **kwargs)

        # Initialize dataset
        super().__init__(dataset, image_size, logic_map)

class ImageGenerationLocalDataset(ImageGenerationDatasetGeneric):
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir

        paths = glob.glob(os.path.join(data_dir, '*'))
        dataset = [{"image": path} for path in paths]
        logic_map = {'image': 'image_file'}
        super().__init__(dataset, image_size, logic_map)

class ImageGenerationLazyZarrDataset(ImageGenerationDatasetGeneric):
    def __init__(self, data_path, image_size, logic_map=None):
        self.data_path = data_path
        
        root = zarr.open(self.data_path, mode='r')
        all_keys = [name for name, _ in root.arrays()]
        if not all_keys:
            raise ValueError(f"No dataset found in {data_path}. Check if the file is valid.")
        
        self.keys = all_keys if logic_map is None else [key for key in all_keys if key in logic_map]
        if not self.keys:
            raise ValueError(f"No selected dataset found in {data_path}. Check logic_map. Available keys: {all_keys}")
        
        all_lengths = [root[key].shape[0] for key in self.keys]
        if not all(length == all_lengths[0] for length in all_lengths):
            raise ValueError(f"All selected datasets in {data_path} must have the same length. Available lengths: {all_lengths}")
        self.length = all_lengths[0]

        super().__init__(None, image_size, logic_map)
    
    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = zarr.open(self.data_path, mode='r')
        data = {key: self.dataset[key][index] for key in self.keys}
        return self._apply_handler(data)
    
    def __len__(self):
        return self.length

class ImageGenerationInMemoryZarrDataset(ImageGenerationDatasetGeneric):
    def __init__(self, data_path, image_size, logic_map=None):
        self.data_path = data_path
        
        root = zarr.open(self.data_path, mode='r')
        all_keys = [name for name, _ in root.arrays()]
        if not all_keys:
            raise ValueError(f"No dataset found in {data_path}. Check if the file is valid.")
        
        self.keys = all_keys if logic_map is None else [key for key in all_keys if key in logic_map]
        if not self.keys:
            raise ValueError(f"No selected dataset found in {data_path}. Check logic_map. Available keys: {all_keys}")
        
        all_lengths = [root[key].shape[0] for key in self.keys]
        if not all(length == all_lengths[0] for length in all_lengths):
            raise ValueError(f"All selected datasets in {data_path} must have the same length. Available lengths: {all_lengths}")
        self.length = all_lengths[0]

        dataset = {key: root[key][:] for key in all_keys}

        super().__init__(dataset, image_size, logic_map)
    
    def __getitem__(self, index):
        data = {key: self.dataset[key][index] for key in self.keys}
        return self._apply_handler(data)
    
    def __len__(self):
        return self.length
