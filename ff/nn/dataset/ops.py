import torch
import numpy as np
import random
import cv2
from collections.abc import Sequence, Callable

def to_tensor(data: np.ndarray|torch.Tensor) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return torch.tensor(data)

def data_f32(data: np.ndarray|torch.Tensor) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=torch.float32)
    return torch.tensor(data, dtype=torch.float32)

def data_i64(data: np.ndarray|torch.Tensor) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=torch.int64)
    return torch.tensor(data, dtype=torch.int64)

def to_f32(data: torch.Tensor) -> torch.Tensor:
    return data.to(dtype=torch.float32)

def to_i64(data: torch.Tensor) -> torch.Tensor:
    return data.to(dtype=torch.int64)

def read_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_COLOR)

def bgr_to_rgb(data: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

def random_hflip(data: np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor:
    if isinstance(data, torch.Tensor):
        return torch.flip(data, dims=[-1]) if random.random() > 0.5 else data
    return cv2.flip(data, 1) if random.random() > 0.5 else data

def hwc_to_chw(data: torch.Tensor) -> torch.Tensor:
    return data.permute(2, 0, 1)

def rescale_unit(data: torch.Tensor) -> torch.Tensor:
    return data / 255.0

def rescale_signed(data: torch.Tensor) -> torch.Tensor:
    return data * 2.0 - 1.0

class ImagenetRescaler:
    def __init__(self):
        self.mean_tensor = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_tensor) / self.std_tensor

def image_to_signed(data: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(data).to(dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
