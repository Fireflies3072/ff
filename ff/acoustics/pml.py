"""PML stretching profiles and grid coefficient sampling.

Profiles evaluate a normalized depth.  Grid sampling uses index coordinates, so a
half-cell stagger is ``0.5`` rather than ``dx / 2``.  Physical spacing enters
only where the profile needs a physical thickness, as in CFS-PML.

``CFSPML.stretch`` returns the coordinate factor ``s = κ + σ / (α + j ω)``.
``PolynomialPML.stretch`` returns the derivative factor used when that profile is selected,

    γ = 1 / (1 + j α_max d^m),

under ``exp(+j ω t)``.  Both have a negative imaginary part for ``d > 0``, so γ
is not ``1/s``.  That sign difference is preserved.
"""

from abc import ABC, abstractmethod
import math

import numpy as np
import torch

class PMLProfile(ABC):
    @abstractmethod
    def stretch(self, depth, omega, c_ref, thickness):
        """Complex stretching factor at normalized depth ``depth``."""

    def inverse_stretch(self, depth, omega, c_ref, thickness):
        """Reciprocal of :meth:`stretch`."""
        return 1.0 / self.stretch(depth, omega, c_ref, thickness)

class CFSPML(PMLProfile):
    """Complex-frequency-shifted PML, ``s = κ + σ / (α + j ω)``."""

    def __init__(self, order=2, log_r=8.0, kappa_max=1.0, alpha_max_hz=40.0):
        self.order = _count(order, "order")
        self.log_r = _scalar(log_r, "log_r")
        self.kappa_max = _scalar(kappa_max, "kappa_max")
        self.alpha_max_hz = _scalar(alpha_max_hz, "alpha_max_hz")
        if self.kappa_max <= 0.0:
            raise ValueError("PML kappa_max must be positive")

    def stretch(self, depth, omega, c_ref, thickness):
        sigma_max = (self.order + 1) * c_ref * self.log_r / (2.0 * thickness)
        sigma = sigma_max * depth ** self.order
        kappa = 1.0 + (self.kappa_max - 1.0) * depth ** self.order
        active = depth > 0
        alpha = (2.0 * math.pi * self.alpha_max_hz) * (1.0 - depth) * active
        return kappa + sigma / _complex_add(alpha, omega)

class PolynomialPML(PMLProfile):
    """Polynomial stretch ``γ = 1 / (1 + j α_max d^order)`` for ``exp(+j ω t)``.

    ``order`` is the exponent ``m``.  ``γ`` is the factor the GMRES operator
    multiplies into a derivative.  Its imaginary part is negative for ``d > 0``.
    """

    def __init__(self, alpha_max=2.0, order=4):
        self.alpha_max = _scalar(alpha_max, "alpha_max")
        self.order = _scalar(order, "order")

    def stretch(self, depth, omega, c_ref, thickness):
        alpha = self.alpha_max * depth ** self.order
        return 1.0 / _complex_add(1.0, alpha)

class PMLCoefficients:
    """Stretching samples on one grid.  The last axis runs over coordinate axes."""

    def __init__(self, stretch, inverse_stretch, jacobian, flux_coefficients):
        self.stretch = stretch
        self.inverse_stretch = inverse_stretch
        self.jacobian = jacobian
        self.flux_coefficients = flux_coefficients

def stretch_axis(positions, n_tot, pml_lo, pml_hi, c_ref, dx, omega):
    """CFS-style 1-D stretch used by the matrix solver, including ghost faces.

    ``positions`` are in grid-index units.  Depth is clipped to ``[0, 1]``.
    Overlapping faces accumulate ``s - 1``.
    """

    positions = np.asarray(positions, dtype=float)
    acc = np.zeros(np.shape(positions), dtype=np.complex128)
    for pml, end in ((pml_lo, "lo"), (pml_hi, "hi")):
        if pml is None or getattr(pml, "thickness", 0) <= 0:
            continue
        thickness = int(pml.thickness)
        if end == "lo":
            depth = np.clip((thickness - positions) / thickness, 0.0, 1.0)
        else:
            depth = np.clip((positions - (n_tot - 1 - thickness)) / thickness, 0.0, 1.0)
        length = thickness * float(dx)
        acc = acc + (pml.profile.stretch(depth, omega, c_ref, length) - 1.0)
    return 1.0 + acc

def build_pml_coefficients(
    boundaries,
    shape,
    spacing,
    omega,
    c_ref,
    stagger=0.0,
    backend="numpy",
    dtype=None,
    device=None,
    resolve=None,
):
    """Sample one stretching component per axis on ``shape``.

    ``stagger`` is in grid cells: ``0`` at centers, ``+0.5`` and ``-0.5`` at the
    staggered derivative locations.  ``resolve(bc)`` replaces ``bc.profile`` when
    the caller wants a profile other than the one stored on the boundary.
    """

    shape = tuple(int(n) for n in shape)
    ndim = len(shape)
    spacing = _spacing(spacing, ndim)
    faces = _faces(boundaries, ndim)
    resolved = []
    for axis in range(ndim):
        pair = {}
        for end in ("lo", "hi"):
            bc = faces[axis][end]
            if bc is None or getattr(bc, "thickness", 0) <= 0:
                pair[end] = None
            else:
                profile = resolve(bc) if resolve is not None else bc.profile
                pair[end] = (bc, profile)
        resolved.append(pair)

    if _uniform_polynomial(resolved, shape):
        thickness = resolved[0]["lo"][0].thickness
        profile = resolved[0]["lo"][1]
        samples = [
            _centered_polynomial(shape[axis], thickness, stagger, profile, backend, dtype, device)
            for axis in range(ndim)
        ]
    else:
        samples = [
            _face_stretch(
                shape[axis], resolved[axis], spacing[axis], omega, c_ref, stagger, backend, dtype, device
            )
            for axis in range(ndim)
        ]
    return _pack(samples, backend, dtype, device)

def _uniform_polynomial(resolved, shape) -> bool:
    profiles = []
    thicknesses = []
    for axis, pair in enumerate(resolved):
        for end in ("lo", "hi"):
            item = pair[end]
            if item is None:
                return False
            thicknesses.append(item[0].thickness)
            profiles.append(item[1])
    if not thicknesses or any(size != thicknesses[0] or size <= 0 for size in thicknesses):
        return False
    if any(shape[axis] <= thicknesses[0] for axis in range(len(shape))):
        return False
    first = profiles[0]
    return all(
        isinstance(profile, PolynomialPML)
        and profile.alpha_max == first.alpha_max
        and profile.order == first.order
        for profile in profiles
    )

def _centered_polynomial(n, thickness, stagger, profile, backend, dtype, device):
    """j-Wave centered depth.  ``stagger`` is added in index units."""
    coord = _centered_index(n, stagger, backend, dtype, device)
    delta = n / 2.0 - float(thickness)
    raw = (abs(coord) - delta) / float(thickness)
    depth = _where_positive(raw)
    return _as_complex(profile.stretch(depth, 0.0, 1.0, 1.0), backend, dtype, device)

def _centered_index(n, stagger, backend, dtype, device):
    if backend == "torch":
        real = _torch_real(dtype)
        if n % 2 == 0:
            axis = torch.arange(n, device=device, dtype=real) - (n / 2.0)
        else:
            axis = torch.arange(n, device=device, dtype=real) - ((n - 1) / 2.0)
        return axis - torch.mean(axis) + float(stagger)
    if n % 2 == 0:
        axis = np.arange(n, dtype=np.float64) - (n / 2.0)
    else:
        axis = np.arange(n, dtype=np.float64) - ((n - 1) / 2.0)
    return axis - np.mean(axis) + float(stagger)

def _face_stretch(n, pair, dx, omega, c_ref, stagger, backend, dtype, device):
    index = _index(n, stagger, backend, dtype, device)
    lo = pair["lo"]
    hi = pair["hi"]
    if _polynomial_only(lo, hi):
        alpha = _zeros_like(index)
        for item, end in ((lo, "lo"), (hi, "hi")):
            if item is None:
                continue
            _, profile = item
            depth = _where_positive(_side_depth(index, n, item[0].thickness, end))
            alpha = alpha + profile.alpha_max * depth ** profile.order
        return _as_complex(1.0 / _complex_add(1.0, alpha), backend, dtype, device)

    acc = _zeros_like(index, complex=True)
    for item, end in ((lo, "lo"), (hi, "hi")):
        if item is None:
            continue
        bc, profile = item
        depth = _clip01(_side_depth(index, n, bc.thickness, end))
        length = bc.thickness * float(dx)
        acc = acc + (profile.stretch(depth, omega, c_ref, length) - 1.0)
    return _as_complex(1.0 + acc, backend, dtype, device)

def _polynomial_only(lo, hi) -> bool:
    items = [item for item in (lo, hi) if item is not None]
    return bool(items) and all(isinstance(item[1], PolynomialPML) for item in items)

def _side_depth(index, n, thickness, end):
    if end == "lo":
        return (float(thickness) - index) / float(thickness)
    return (index - (n - 1 - thickness)) / float(thickness)

def _index(n, stagger, backend, dtype, device):
    if backend == "torch":
        return torch.arange(n, device=device, dtype=_torch_real(dtype)) + float(stagger)
    return np.arange(n, dtype=np.float64) + float(stagger)

def _where_positive(depth):
    if torch.is_tensor(depth):
        return torch.where(depth > 0, depth, torch.zeros_like(depth))
    return np.where(depth > 0, depth, 0.0)

def _clip01(depth):
    if torch.is_tensor(depth):
        return torch.clamp(depth, 0.0, 1.0)
    return np.clip(depth, 0.0, 1.0)

def _complex_add(real_part, imag_part):
    """``real_part + j * imag_part`` without promoting a float32 tensor.

    A Python ``1j`` multiplied into a float32 tensor is promoted by PyTorch.
    Building the imaginary unit at the tensor dtype keeps complex64 arithmetic.
    """

    if torch.is_tensor(real_part) or torch.is_tensor(imag_part):
        reference = real_part if torch.is_tensor(real_part) else imag_part
        real = reference.dtype if reference.dtype.is_floating_point else (
            torch.float32 if reference.dtype == torch.complex64 else torch.float64
        )
        complex_dtype = torch.complex64 if real == torch.float32 else torch.complex128
        if not torch.is_tensor(real_part):
            real_part = torch.as_tensor(real_part, dtype=real, device=reference.device)
        eye = torch.tensor(1j, dtype=complex_dtype, device=reference.device)
        return real_part.to(complex_dtype) + eye * imag_part
    return real_part + 1j * imag_part

def _as_complex(value, backend, dtype, device):
    """Keep float32 profiles in complex64 and float64 profiles in complex128."""
    if backend == "torch":
        complex_dtype = torch.complex64 if _torch_real(dtype) == torch.float32 else torch.complex128
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=device)
        if value.dtype != complex_dtype:
            value = value.to(dtype=complex_dtype)
        return value
    return np.asarray(value, dtype=np.complex128)

def _zeros_like(index, complex=False):
    if torch.is_tensor(index):
        if complex:
            dtype = torch.complex64 if index.dtype == torch.float32 else torch.complex128
            return torch.zeros(index.shape, dtype=dtype, device=index.device)
        return torch.zeros_like(index)
    if complex:
        return np.zeros(np.shape(index), dtype=np.complex128)
    return np.zeros(np.shape(index), dtype=np.float64)

def _pack(samples, backend, dtype, device):
    if backend == "torch":
        grids = torch.meshgrid(*samples, indexing="ij")
        stretch = torch.stack(grids, dim=-1)
    else:
        grids = np.meshgrid(*samples, indexing="ij")
        stretch = np.stack(grids, axis=-1)
    inverse = 1.0 / stretch
    jacobian = stretch.prod(dim=-1) if torch.is_tensor(stretch) else np.prod(stretch, axis=-1)
    flux = jacobian.unsqueeze(-1) / stretch ** 2 if torch.is_tensor(stretch) else jacobian[..., None] / stretch ** 2
    return PMLCoefficients(stretch, inverse, jacobian, flux)

def _faces(boundaries, ndim):
    from .boundary import PMLBC, iter_sides

    faces = [{"lo": None, "hi": None} for _ in range(ndim)]
    for _, axis, end, bc in iter_sides(boundaries):
        faces[axis][end] = bc if isinstance(bc, PMLBC) else None
    return faces

def _spacing(spacing, ndim):
    if isinstance(spacing, (int, float)):
        return (float(spacing),) * ndim
    values = tuple(float(value) for value in spacing)
    if len(values) != ndim:
        raise ValueError(f"spacing must have length {ndim}")
    return values

def _torch_real(dtype):
    if dtype in (torch.float32, torch.complex64, None):
        return torch.float32
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    raise TypeError(f"unsupported dtype {dtype}")

def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"PML {name} must be an integer")
    if int(value) < 0:
        raise ValueError(f"PML {name} must be non-negative")
    return int(value)

def _scalar(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"PML {name} must be a real scalar") from None
    if isinstance(value, (complex, np.complexfloating)) and float(np.imag(value)) != 0.0:
        raise TypeError(f"PML {name} must be a real scalar")
    if not math.isfinite(number):
        raise ValueError(f"PML {name} must be finite")
    return number
