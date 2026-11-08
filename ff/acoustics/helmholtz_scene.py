"""Physical medium and boundary conditions for the Helmholtz solvers.

2-D maps are ``(Ny, Nx)`` and 3-D maps are ``(Nz, Ny, Nx)``.  ``c`` and ``rho``
are real, finite, and positive.  ``boundaries`` is one outer ``Boundary2d`` or
``Boundary3d`` plus any number of matching regions.  An omitted outer box is a
default PML on every face.  Later regions override earlier ones where their
masks overlap.
"""

import numpy as np
import torch

from .boundary import parse_boundaries

def _positive_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be positive") from None
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number

def _check_grid_shape(c, rho) -> None:
    if c.ndim not in (2, 3) or rho.ndim not in (2, 3):
        raise ValueError("c and rho must be 2-D or 3-D arrays")
    if tuple(c.shape) != tuple(rho.shape):
        raise ValueError("c and rho must have the same shape")
    if any(size < 1 for size in c.shape):
        raise ValueError("c and rho must have at least one cell along each axis")

def _numpy_dtype(dtype) -> np.dtype:
    try:
        resolved = np.dtype(dtype)
    except TypeError:
        raise TypeError("dtype must be a real NumPy floating type") from None
    if not np.issubdtype(resolved, np.floating):
        raise TypeError("dtype must be a real NumPy floating type")
    return resolved

def _real_positive_numpy(array, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.bool_ or np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError("c and rho must be real")
    array = np.asarray(array, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError("c and rho must be finite")
    if np.any(array <= 0.0):
        raise ValueError("c and rho must be positive")
    return array

def _torch_dtype(dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point or dtype.is_complex:
        raise TypeError("dtype must be a real torch floating type")
    return dtype

def _real_positive_torch(array, dtype: torch.dtype, device) -> torch.Tensor:
    if torch.is_tensor(array):
        if array.is_complex():
            raise ValueError("c and rho must be real")
        tensor = array.to(device=device, dtype=dtype)
    else:
        values = np.asarray(array)
        if values.dtype == np.bool_ or np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
            raise ValueError("c and rho must be real")
        tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError("c and rho must be finite")
    if torch.any(tensor <= 0):
        raise ValueError("c and rho must be positive")
    return tensor

class HelmholtzSceneNumpy:
    def __init__(self, c: np.ndarray, rho: np.ndarray, dx: float, boundaries=None, dtype=None):
        """NumPy medium.  ``c`` and ``rho`` are stored with ``dtype``.

        An omitted ``dtype`` follows a floating ``c``, otherwise ``float64``.
        """

        c = np.asarray(c)
        rho = np.asarray(rho)
        _check_grid_shape(c, rho)
        if dtype is None:
            if np.issubdtype(c.dtype, np.floating):
                dtype = c.dtype
            elif np.issubdtype(rho.dtype, np.floating):
                dtype = rho.dtype
            else:
                dtype = np.float64
        self.dtype = _numpy_dtype(dtype)
        self.c = _real_positive_numpy(c, self.dtype)
        self.rho = _real_positive_numpy(rho, self.dtype)
        self.dx = _positive_scalar(dx, "dx")
        self.ndim = self.c.ndim
        self.shape = self.c.shape
        self.boundaries, self.outer, self.regions = parse_boundaries(
            boundaries, ndim=self.ndim, shape=self.shape)

class HelmholtzSceneTorch:
    def __init__(self, c, rho, dx: float, boundaries=None, dtype=None):
        """Torch medium.  ``c`` and ``rho`` are stored with ``dtype``.

        An omitted ``dtype`` follows a floating ``c``, otherwise ``float32``.
        """

        if dtype is None:
            if torch.is_tensor(c) and c.dtype.is_floating_point and not c.is_complex():
                dtype = c.dtype
            elif torch.is_tensor(rho) and rho.dtype.is_floating_point and not rho.is_complex():
                dtype = rho.dtype
            else:
                dtype = torch.float32
        self.dtype = _torch_dtype(dtype)
        if torch.is_tensor(c):
            device = c.device
        elif torch.is_tensor(rho):
            device = rho.device
        else:
            device = torch.device("cpu")
        c = _real_positive_torch(c, self.dtype, device)
        rho = _real_positive_torch(rho, self.dtype, c.device)
        _check_grid_shape(c, rho)
        self.c = c
        self.rho = rho
        self.dx = _positive_scalar(dx, "dx")
        self.ndim = self.c.ndim
        self.shape = tuple(self.c.shape)
        self.boundaries, self.outer, self.regions = parse_boundaries(
            boundaries, ndim=self.ndim, shape=self.shape)
