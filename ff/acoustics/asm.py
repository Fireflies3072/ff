import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq
from .dispersion import DispersionBase2d, ContinuousDispersion2d, DispersionBase3d, ContinuousDispersion3d

def propagate_asm2d(p: np.ndarray, z_dist: float, f: float, c: float,
                    dx: float, dispersion: DispersionBase2d | None = None) -> np.ndarray:
    """Propagate a 1D complex pressure field along the z-axis using the 2D Angular Spectrum Method (ASM).

    Physical geometry: 2D wave propagation in the (x, z) plane.
    Convention: Assumes an exp(j * omega * t) time-harmonic dependence,
                where forward propagation (+z) introduces a phase delay exp(-j * kz * z).

    Args:
        p: 1D complex acoustic/optical pressure field sampled along the x-axis.
        z_dist: Propagation distance in meters (+z for forward, -z for backward).
        f: Frequency in Hz.
        c: Phase speed (e.g. speed of sound) in m/s.
        dx: Spatial sampling pitch along the x-axis in meters.
        dispersion: Dispersion relation object. Defaults to ContinuousDispersion2d if None.
    
    Returns:
        1D complex pressure field at the target plane z = z_dist.
    """

    # If dispersion is not provided, use ContinuousDispersion
    if dispersion is None:
        dispersion = ContinuousDispersion2d()

    # Check if the input pressure field is 1D
    if p.ndim != 1:
        raise ValueError(f"Input pressure field must be 1D, got shape {p.shape}")

    # Total wavenumber and spatial frequencies along x
    k0 = 2.0 * np.pi * f / c
    kx = fftfreq(p.shape[0], d=dx) * 2.0 * np.pi
    kz = dispersion.get_kz(k0, kx, dx)

    # 1D FFT for angular spectrum calculation
    p_fft = fft(p)
    if z_dist < 0:
        p_fft[kz.imag < 0] = 0
    
    # Phase-shift propagation operator and 1D IFFT
    h_prop = np.exp(-1j * kz * z_dist)
    p_prop = ifft(p_fft * h_prop)
    return p_prop

def propagate_asm3d(p: np.ndarray, z_dist: float, f: float, c: float,
                    dx: float, dy: float | None = None, dispersion: DispersionBase3d | None = None) -> np.ndarray:
    """Propagate a 2D complex pressure field along the z-axis using 3D Angular Spectrum Method (ASM).

    Physical geometry: 3D wave propagation in (x, y, z) space, advancing from plane z=0 to z=z_dist.
    Convention: Assumes an exp(j * omega * t) time-harmonic dependence,
                where forward propagation (+z) introduces a phase delay exp(-j * kz * z).

    Args:
        p: 2D complex pressure field on the source plane (z=0), shape (Ny, Nx).
           Ny corresponds to rows (y-axis) and Nx corresponds to columns (x-axis).
        z_dist: Propagation distance in meters (+z for forward, -z for backward).
        f: Frequency in Hz.
        c: Phase speed (e.g. speed of sound) in m/s.
        dx: Spatial sampling pitch along the x-axis in meters (columns).
        dy: Spatial sampling pitch along the y-axis in meters (rows). Defaults to dx if None.
        dispersion: Dispersion relation object. Defaults to ContinuousDispersion3d if None.

    Returns:
        2D complex pressure field on the target plane (z=z_dist), shape (Ny, Nx).
    """

    # If dispersion is not provided, use ContinuousDispersion3d
    if dispersion is None:
        dispersion = ContinuousDispersion3d()

    # Check if the input pressure field is 2D
    if p.ndim != 2:
        raise ValueError(f"Input pressure field must be 2D, got shape {p.shape}")

    # If dy is not provided, set it to dx
    if dy is None:
        dy = dx

    # Total wavenumber and spatial frequencies along x and y
    k0 = 2.0 * np.pi * f / c
    kx = fftfreq(p.shape[1], d=dx) * (2.0 * np.pi)
    ky = fftfreq(p.shape[0], d=dy) * (2.0 * np.pi)

    # Get the longitudinal wavenumber kz
    kz = dispersion.get_kz(k0, kx, ky, dx, dy)

    # 2D FFT for angular spectrum calculation
    p_fft = fft2(p)
    p_fft[kz.imag < 0] = 0.0

    # Phase-shift propagation operator and 2D IFFT
    h_prop = np.exp(-1j * kz * z_dist)
    p_prop = ifft2(p_fft * h_prop)

    return p_prop
