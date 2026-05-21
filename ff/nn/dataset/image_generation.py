import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
import glob
import os
import h5py
from datasets import Image, load_dataset
import ff.cv as fcv

class ImageGenerationGenericDataset(Dataset):
    def __init__(self, dataset, image_size, logic_map):
        self.dataset = dataset
        self.image_size = image_size
        self.logic_map = logic_map

        # Logic registry
        self._logic_registry = {
            'image_raw': self._handle_image_raw,
            'image_file': self._handle_image_file,
            'data_tof32': self._handle_data_tof32
        }

        # Create handler map
        self.handler_map = {}
        logic_map = logic_map or {}
        for key, logic in logic_map.items():
            # Assign handler
            if logic is not None:
                self.handler_map[key] = self._logic_registry[logic]

    def __getitem__(self, index):
        data = dict(self.dataset[index])
        return self._apply_handler(data)

    def __len__(self):
        return len(self.dataset)
    
    def _apply_handler(self, data):
        for key, handler in self.handler_map.items():
            if handler is not None:
                data[key] = handler(data[key])
        return data
    
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
    
    def _handle_data_tof32(self, data):
        return torch.tensor(data, dtype=torch.float32)
    
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

class ImageGenerationHFDataset(ImageGenerationGenericDataset):
    def __init__(self, hf_id, split='train', image_size=(128, 128), logic_map=None, **kwargs):
        self.hf_id = hf_id
        self.split = split

        # Load dataset
        dataset = load_dataset(hf_id, split=split, **kwargs)

        # Default handler map {'image': 'image_raw'}
        if logic_map is None:
            logic_map = {}
            if 'image' in dataset.column_names:
                logic_map['image'] = 'image_raw'
        
        for key, logic in logic_map.items():
            if logic == 'image_raw':
                dataset = dataset.cast_column(key, Image(decode=False))
        
        # Initialize dataset
        super().__init__(dataset, image_size, logic_map)

class ImageGenerationLocalDataset(ImageGenerationGenericDataset):
    def __init__(self, data_dir, image_size=(128, 128)):
        self.data_dir = data_dir

        filenames = glob.glob(os.path.join(data_dir, '*'))
        dataset = [{"image": filename} for filename in filenames]
        logic_map = {'image': 'image_file'}
        super().__init__(dataset, image_size, logic_map)

class ImageGenerationLazyH5Dataset(ImageGenerationGenericDataset):
    def __init__(self, data_path, image_size=(128, 128), logic_map=None):
        self.data_path = data_path
        
        self.h5_dataset = None
        with h5py.File(self.data_path, 'r') as f:
            self.keys = [key for key in f if isinstance(f[key], h5py.Dataset)]
            if not self.keys:
                raise ValueError(f"No dataset found in {self.data_path}")
            self.length = len(f[self.keys[0]])
        
        super().__init__(None, image_size, logic_map)
    
    def __getitem__(self, index):
        if self.h5_dataset is None:
            self.h5_dataset = h5py.File(self.data_path, 'r')
        data = {key: self.h5_dataset[key][index] for key in self.keys}
        return self._apply_handler(data)
    
    def __len__(self):
        return self.length

class ImageGenerationInMemoryH5Dataset(ImageGenerationGenericDataset):
    def __init__(self, data_path, image_size=(128, 128), logic_map=None):
        self.data_path = data_path
        
        self.h5_dataset = {}
        with h5py.File(self.data_path, 'r') as f:
            keys = [key for key in f if isinstance(f[key], h5py.Dataset)]
            if not keys:
                raise ValueError(f"No dataset found in {self.data_path}")
            self.length = len(f[keys[0]])
            self.h5_dataset = {key: f[key][:] for key in keys}

        super().__init__(None, image_size, logic_map)
    
    def __getitem__(self, index):
        data = {key: self.h5_dataset[key][index] for key in self.h5_dataset}
        return self._apply_handler(data)
    
    def __len__(self):
        return self.length
