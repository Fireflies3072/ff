import torch
import numpy as np
import cv2
import random
import glob
import os
import zarr
from datasets import Image, load_dataset
import ff.cv as fcv

from .base import DatasetGeneric

class ImageGenerationDatasetGeneric(DatasetGeneric):
    def __init__(self, dataset, image_size, logic_map):
        self.image_size = image_size

        # Logic registry
        self._logic_registry = {
            'image_raw': self._handle_image_raw,
            'image_file': self._handle_image_file,
            'latent_scale': self._handle_latent_scale,
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

    def _handle_image_file(self, filename):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {filename}")
        return self._op_common_image(image)
    
    def _handle_latent_scale(self, data):
        return torch.tensor(data, dtype=torch.float32) * self.h5_attrs['scaling_factor']
    
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
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0
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

        filenames = glob.glob(os.path.join(data_dir, '*'))
        dataset = [{"image": filename} for filename in filenames]
        logic_map = {'image': 'image_file'}
        super().__init__(dataset, image_size, logic_map)

class ImageGenerationLazyZarrDataset(ImageGenerationDatasetGeneric):
    def __init__(self, data_path, image_size, logic_map=None):
        self.data_path = data_path
        
        self.zarr_dataset = None
        root = zarr.open(self.data_path, 'r')
        self.keys = [name for name, _ in root.arrays()]
        if not self.keys:
            raise ValueError(f"No dataset found in {self.data_path}")
        self.length = root[self.keys[0]].shape[0]

        super().__init__(None, image_size, logic_map)
    
    def __getitem__(self, index):
        if self.zarr_dataset is None:
            self.zarr_dataset = zarr.open(self.data_path, 'r')
        data = {key: self.zarr_dataset[key][index] for key in self.keys}
        return self._apply_handler(data)
    
    def __len__(self):
        return self.length

class ImageGenerationInMemoryZarrDataset(ImageGenerationDatasetGeneric):
    def __init__(self, data_path, image_size, logic_map=None):
        self.data_path = data_path
        
        self.zarr_dataset = {}
        root = zarr.open(self.data_path, 'r')
        self.keys = [name for name, _ in root.arrays()]
        if not self.keys:
            raise ValueError(f"No dataset found in {self.data_path}")
        self.length = root[self.keys[0]].shape[0]
        self.zarr_dataset = {key: root[key][:] for key in self.keys}

        super().__init__(None, image_size, logic_map)
    
    def __getitem__(self, index):
        data = {key: self.zarr_dataset[key][index] for key in self.keys}
        return self._apply_handler(data)
    
    def __len__(self):
        return self.length
