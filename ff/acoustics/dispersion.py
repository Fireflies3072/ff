import numpy as np
import torch
from abc import ABC, abstractmethod
from ..common import get_array_module, to_complex

class DispersionBase2d(ABC):
    def __init__(self, accuracy_order: int, num_points: int):
        self.accuracy_order = accuracy_order
        self.num_points = num_points
    
        self.ndim = 2
    
    @abstractmethod
    def get_kz(self, k0: float, kx: np.ndarray, dx: float) -> np.ndarray:
        """Get the longitudinal wavenumber kz.
        
        Args:
            k0: Total wavenumber in rad/m.
            kx: Lateral wavenumber in rad/m.
            dx: Grid spacing in m.

        Returns:
            kz: Longitudinal wavenumber in rad/m.
        """

        pass

class ContinuousDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=0, num_points=0)
    
    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, dx: float | None = None) -> np.ndarray | torch.Tensor:
        xp = get_array_module(kx)
        kz_square = k0**2 - kx**2
        kz = xp.sqrt(to_complex(kz_square))
        kz = kz.real - 1j * xp.abs(kz.imag)
        return kz

class SecondOrderFivePointDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=2, num_points=5)
    
    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, dx: float) -> np.ndarray | torch.Tensor:
        xp = get_array_module(kx)
        cx = xp.cos(kx * dx)
        cz = 2.0 - cx - 0.5 * (k0 * dx)**2
        kz = xp.arccos(to_complex(cz)) / dx
        return kz.real - 1j * xp.abs(kz.imag)

class SecondOrderNinePointCompactDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=2, num_points=9)
    
    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, dx: float) -> np.ndarray | torch.Tensor:
        xp = get_array_module(kx)
        cx = xp.cos(kx * dx)
        k2 = (k0 * dx)**2
        cz = (20.0 - 8.0 * cx - 6.0 * k2) / (8.0 + 4.0 * cx)
        kz = xp.arccos(to_complex(cz)) / dx
        return kz.real - 1j * xp.abs(kz.imag)

class FourthOrderNinePointCrossDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=4, num_points=9)
    
    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, dx: float) -> np.ndarray | torch.Tensor:
        xp = get_array_module(kx)
        cx = xp.cos(kx * dx)
        k2 = (k0 * dx)**2
        cz = 4.0 - xp.sqrt(to_complex(2.0 + 8.0 * cx - cx**2 + 3.0 * k2))
        kz = xp.arccos(to_complex(cz)) / dx
        return kz.real - 1j * xp.abs(kz.imag)

class FourthOrderNinePointCompactDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=4, num_points=9)
    
    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, dx: float) -> np.ndarray | torch.Tensor:
        xp = get_array_module(kx)
        cx = xp.cos(kx * dx)
        k2 = (k0 * dx)**2
        cz = (20.0 - 8.0 * cx - k2 * (4.0 + cx)) / (8.0 + 4.0 * cx + k2)
        kz = xp.arccos(to_complex(cz)) / dx
        return kz.real - 1j * xp.abs(kz.imag)

class DispersionBase3d(ABC):
    def __init__(self, accuracy_order: int, num_points: int):
        self.accuracy_order = accuracy_order
        self.num_points = num_points
        
        self.ndim = 3
    
    @abstractmethod
    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, ky: np.ndarray | torch.Tensor,
               dx: float, dy: float | None = None) -> np.ndarray | torch.Tensor:
        """Get the longitudinal wavenumber kz.
        
        Args:
            k0: Total wavenumber in rad/m.
            kx: Lateral wavenumber in rad/m.
            ky: Lateral wavenumber in rad/m.
            dx: Grid spacing in m.
            dy: Grid spacing in m.

        Returns:
            kz: Longitudinal wavenumber in rad/m.
        """

        pass

class ContinuousDispersion3d(DispersionBase3d):
    def __init__(self):
        super().__init__(accuracy_order=0, num_points=0)
    
    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, ky: np.ndarray | torch.Tensor,
               dx: float | None = None, dy: float | None = None) -> np.ndarray | torch.Tensor:
        xp = get_array_module(kx)
        kx_grid = kx[None, :]
        ky_grid = ky[:, None]
        kz_square = k0**2 - (kx_grid**2 + ky_grid**2)
        kz = xp.sqrt(to_complex(kz_square))
        return kz.real - 1j * xp.abs(kz.imag)

class SecondOrderSevenPointDispersion3d(DispersionBase3d):
    def __init__(self):
        super().__init__(accuracy_order=2, num_points=7)

    def get_kz(self, k0: float, kx: np.ndarray, ky: np.ndarray,
               dx: float, dy: float | None = None) -> np.ndarray | torch.Tensor:
        if dy is None:
            dy = dx
        
        xp = get_array_module(kx)
        cx = xp.cos(kx[None, :] * dx)
        cy = xp.cos(ky[:, None] * dy)

        cz = 2.0 - cx + (dx / dy)**2 * (1.0 - cy) - 0.5 * (k0 * dx)**2
        kz = xp.arccos(to_complex(cz)) / dx
        return kz.real - 1j * xp.abs(kz.imag)

class FourthOrderThirteenPointCrossDispersion3d(DispersionBase3d):
    def __init__(self):
        super().__init__(accuracy_order=4, num_points=13)

    def get_kz(self, k0: float, kx: np.ndarray, ky: np.ndarray,
               dx: float, dy: float | None = None) -> np.ndarray | torch.Tensor:
        if dy is None:
            dy = dx
        
        xp = get_array_module(kx)
        cx = xp.cos(kx[None, :] * dx)
        cy = xp.cos(ky[:, None] * dy)

        # Fourth-order second-derivative symbols
        qx = (-cx**2 + 8.0 * cx - 7.0) / (3.0 * dx**2)
        qy = (-cy**2 + 8.0 * cy - 7.0) / (3.0 * dy**2)
        # qx + qy + qz + k0**2 = 0
        qz = -k0**2 - qx - qy
        
        cz = 4.0 - xp.sqrt(to_complex(9.0 - 3.0 * dx**2 * qz))
        kz = xp.arccos(to_complex(cz)) / dx
        return kz.real - 1j * xp.abs(kz.imag)

class SecondOrderNineteenPointCompactDispersion3d(DispersionBase3d):
    def __init__(self):
        super().__init__(accuracy_order=2, num_points=19)

    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, ky: np.ndarray | torch.Tensor,
               dx: float, dy: float | None = None) -> np.ndarray | torch.Tensor:
        if dy is None:
            dy = dx

        xp = get_array_module(kx)
        hx, hy, hz = dx, dy, dx

        cx = xp.cos(kx[None, :] * hx)
        cy = xp.cos(ky[:, None] * hy)

        qx = 2.0 * (cx - 1.0) / hx**2
        qy = 2.0 * (cy - 1.0) / hy**2

        # Pairwise mixed-difference coefficients
        axy = (hx**2 + hy**2) / 12.0
        axz = (hx**2 + hz**2) / 12.0
        ayz = (hy**2 + hz**2) / 12.0

        # Symbol:
        # qx + qy + qz
        # + axy*qx*qy + axz*qx*qz + ayz*qy*qz
        # + k0**2 = 0
        numerator = -(k0**2 + qx + qy + axy * qx * qy)
        denominator = 1.0 + axz * qx + ayz * qy

        qz = numerator / denominator
        cz = 1.0 + 0.5 * hz**2 * qz
        kz = xp.arccos(to_complex(cz)) / hz
        return kz.real - 1j * xp.abs(kz.imag)

class SecondOrderTwentySevenPointCompactDispersion3d(DispersionBase3d):
    def __init__(self):
        super().__init__(accuracy_order=2, num_points=27)

    def get_kz(self, k0: float, kx: np.ndarray | torch.Tensor, ky: np.ndarray | torch.Tensor,
               dx: float, dy: float | None = None) -> np.ndarray | torch.Tensor:
        if dy is None:
            dy = dx

        xp = get_array_module(kx)
        hx, hy, hz = dx, dy, dx

        cx = xp.cos(kx[None, :] * hx)
        cy = xp.cos(ky[:, None] * hy)

        qx = 2.0 * (cx - 1.0) / hx**2
        qy = 2.0 * (cy - 1.0) / hy**2

        axy = (hx**2 + hy**2) / 12.0
        axz = (hx**2 + hz**2) / 12.0
        ayz = (hy**2 + hz**2) / 12.0

        # Triple mixed-difference coefficient
        axyz = (
            hx**2 * hy**2
            + hy**2 * hz**2
            + hz**2 * hx**2
        ) / 90.0

        # Symbol:
        # qx + qy + qz
        # + axy*qx*qy + axz*qx*qz + ayz*qy*qz
        # + axyz*qx*qy*qz + k0**2 = 0
        numerator = -(k0**2 + qx + qy + axy * qx * qy)
        denominator = (
            1.0 + axz * qx + ayz * qy
            + axyz * qx * qy
        )

        qz = numerator / denominator
        cz = 1.0 + 0.5 * hz**2 * qz
        kz = xp.arccos(to_complex(cz)) / hz
        return kz.real - 1j * xp.abs(kz.imag)
