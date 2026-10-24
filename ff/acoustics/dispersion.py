import numpy as np
from abc import ABC, abstractmethod

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
    
    def get_kz(self, k0: float, kx: np.ndarray, dx: float | None = None) -> np.ndarray:
        kz_square = k0**2 - kx**2
        kz = np.sqrt(kz_square.astype(np.complex128))
        kz = kz.real - 1j * np.abs(kz.imag)
        return kz

class SecondOrderFivePointDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=2, num_points=5)
    
    def get_kz(self, k0: float, kx: np.ndarray, dx: float) -> np.ndarray:
        cx = np.cos(kx * dx)
        cz = 2.0 - cx - 0.5 * (k0 * dx)**2
        kz = np.arccos(cz.astype(np.complex128)) / dx
        return kz.real - 1j * np.abs(kz.imag)

class SecondOrderNinePointCompactDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=2, num_points=9)

class FourthOrderNinePointCrossDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=4, num_points=9)

class FourthOrderNinePointCompactDispersion2d(DispersionBase2d):
    def __init__(self):
        super().__init__(accuracy_order=4, num_points=9)

class DispersionBase3d(ABC):
    def __init__(self, accuracy_order: int, num_points: int):
        self.accuracy_order = accuracy_order
        self.num_points = num_points
        
        self.ndim = 3
    
    @abstractmethod
    def get_kz(self, k0: float, kx: np.ndarray, ky: np.ndarray, dx: float, dy: float) -> np.ndarray:
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
    
    def get_kz(self, k0: float, kx: np.ndarray, ky: np.ndarray,
               dx: float | None = None, dy: float | None = None) -> np.ndarray:
        kx_grid = kx[None, :]
        ky_grid = ky[:, None]
        kz_square = k0**2 - (kx_grid**2 + ky_grid**2)
        kz = np.sqrt(kz_square.astype(np.complex128))
        kz = kz.real - 1j * np.abs(kz.imag)
        return kz
