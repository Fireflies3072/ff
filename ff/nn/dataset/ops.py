import torch
import numpy as np
import random
import cv2
from collections.abc import Sequence

def random_hflip(data: np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor:
    if isinstance(data, torch.Tensor):
        return torch.flip(data, dims=[-1]) if random.random() > 0.5 else data
    return cv2.flip(data, 1) if random.random() > 0.5 else data

def create_normalizer(mean: Sequence[float], std: Sequence[float]) -> callable:
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    return lambda x: (x - mean_tensor) / std_tensor

def image_to_signed(data: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(data).to(dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
