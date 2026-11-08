import numpy as np
import torch

def get_array_module(x: np.ndarray | torch.Tensor):
    """Get the array module for a given array.

    Common usage:
    ```
    xp = get_array_module(x)
    xp.sin(x)
    ```
    
    Args:
        x: The array to get the array module for.
    
    Returns:
        The array module for the given array.
    """

    if isinstance(x, np.ndarray):
        return np
    elif isinstance(x, torch.Tensor):
        return torch
    else:
        raise TypeError(f"Unsupported array type: {type(x)}")

def to_complex(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert an array to a complex array.

    Args:
        x: The array to convert to a complex array.
    
    Returns:
        The complex array.
    """

    if isinstance(x, np.ndarray):
        dtype = np.promote_types(x.dtype, np.complex64)
        return x.astype(dtype, copy=False)
    elif isinstance(x, torch.Tensor):
        dtype = torch.promote_types(x.dtype, torch.complex64)
        return x.to(dtype)
    else:
        raise TypeError(f"Unsupported array type: {type(x)}")

def fftfreq(n: int, d: float, like: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Get the frequency bins for a given array.

    Args:
        n: The number of samples.
        d: The spacing between samples.
        like: The array to get the frequency bins for.
    """
    
    if isinstance(like, torch.Tensor):
        return torch.fft.fftfreq(n, d=d, dtype=like.real.dtype, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.fft.fftfreq(n, d=d).astype(like.real.dtype, copy=False)
    else:
        raise TypeError(f"Unsupported array type: {type(like)}")
