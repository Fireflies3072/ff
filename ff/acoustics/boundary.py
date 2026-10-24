"""Boundary conditions for the frequency-domain Helmholtz solvers.

The unknown is the complex pressure amplitude under ``exp(+j ω t)``.  On every
boundary the normal ``n`` points out of the active fluid.

Array layout follows NumPy.  2-D fields are ``(Ny, Nx)`` and 3-D fields are
``(Nz, Ny, Nx)``, so the last axis is always ``x``.

* ``left`` / ``right``: low / high ``x``, outward normals ``-x`` / ``+x``.
* ``top`` / ``bottom``: low / high ``y``, outward normals ``-y`` / ``+y``.
* ``front`` / ``back``: low / high ``z``, outward normals ``-z`` / ``+z``.

A boundary value may be a scalar, the tangential face, or the full physical
grid.  Face shapes are ``(Ny,)`` and ``(Nx,)`` in 2-D, and ``(Nz, Ny)``,
``(Nz, Nx)``, and ``(Ny, Nx)`` for the ``x``, ``y``, and ``z`` faces in 3-D.
Values continue into a tangential PML by edge replication.

A Dirichlet node replaces the stencil and takes precedence over Neumann or
Robin data on any other face of that node.  Neumann and Robin data on different
faces of one node both enter the flux balance.  Dirichlet values must agree
where faces meet.  Periodic conditions are allowed only as an opposite pair.

``BoundaryRegion2d`` and ``BoundaryRegion3d`` mark obstacle cells.  The
condition acts on the fluid/obstacle faces, not by assigning a pressure to the
obstacle cell.  Array data use the physical grid shape and are read from the
obstacle cell, with ``n`` pointing out of the fluid.  Region Dirichlet data are
the pressure on the midway face.  Region Robin data use the fluid-cell pressure,
``a * p_fluid + b * dp/dn = g``.  Outer Neumann and Robin data instead use the
boundary-cell pressure and a one-sided derivative.

Later regions overwrite earlier ones where masks overlap, and a region overrides
the outer condition on every cell it masks.  An obstacle face on a periodic
seam uses the region condition instead of the periodic flux.
"""

from collections.abc import Sequence

import numpy as np

_SIDES_2D = (
    ("top", 0, "lo"),
    ("bottom", 0, "hi"),
    ("left", 1, "lo"),
    ("right", 1, "hi"),
)
_SIDES_3D = (
    ("front", 0, "lo"),
    ("back", 0, "hi"),
    ("top", 1, "lo"),
    ("bottom", 1, "hi"),
    ("left", 2, "lo"),
    ("right", 2, "hi"),
)
_PAIRS_2D = (("left", "right"), ("top", "bottom"))
_PAIRS_3D = (("left", "right"), ("top", "bottom"), ("front", "back"))

def _as_numeric(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array

class DirichletBC:
    def __init__(self, value):
        """Prescribed pressure ``p = value``."""
        self.value = _as_numeric(value, "DirichletBC value")

class NeumannBC:
    def __init__(self, value):
        """Prescribed outward derivative ``dp/dn = value``."""
        self.value = _as_numeric(value, "NeumannBC value")

class RobinBC:
    def __init__(self, a, b, g):
        """Impedance condition ``a * p + b * dp/dn = g`` with outward ``n``.

        ``b = 0`` is the Dirichlet condition ``a * p = g``.  Enforced locations
        must have ``a`` or ``b`` nonzero.
        """

        self.a = _as_numeric(a, "RobinBC a")
        self.b = _as_numeric(b, "RobinBC b")
        self.g = _as_numeric(g, "RobinBC g")

class PeriodicBC:
    def __init__(self):
        """Periodic identification with the opposite face."""

class PMLBC:
    def __init__(self, thickness=16, order=2, log_r=8.0, kappa_max=1.0, alpha_max_hz=40.0):
        """Complex-frequency-shifted PML on one face.

        The stretch is ``s = κ + σ / (α + j ω)`` for ``exp(+j ω t)``.  ``thickness``
        is the number of cells added outside this face.  The profile and the
        homogeneous Dirichlet ghost beyond the outer PML cell match the previous
        uniform-PML discretization.  ``log_r`` sets the nominal normal-incidence
        reflection to ``exp(-log_r)``.
        """

        self.thickness = _pml_count(thickness, "thickness")
        self.order = _pml_count(order, "order")
        self.log_r = _pml_scalar(log_r, "log_r")
        self.kappa_max = _pml_scalar(kappa_max, "kappa_max")
        self.alpha_max_hz = _pml_scalar(alpha_max_hz, "alpha_max_hz")
        if self.kappa_max <= 0.0:
            raise ValueError("PML kappa_max must be positive")

def _pml_count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"PML {name} must be an integer")
    if int(value) < 0:
        raise ValueError(f"PML {name} must be non-negative")
    return int(value)

def _pml_scalar(value, name: str) -> float:
    value = _as_numeric(value, f"PML {name}")
    if value.shape != ():
        raise ValueError(f"PML {name} must be a scalar")
    if np.iscomplexobj(value) and np.any(np.imag(value) != 0.0):
        raise TypeError(f"PML {name} must be real")
    return float(np.real(value))

def _assign_faces(owner, faces, pairs) -> None:
    allowed = (DirichletBC, NeumannBC, RobinBC, PeriodicBC, PMLBC)
    for name, value in faces:
        stored = PMLBC() if value is None else value
        if not isinstance(stored, allowed):
            raise TypeError(f"{name} must be a Dirichlet, Neumann, Robin, periodic, or PML condition")
        setattr(owner, name, stored)
    for low_name, high_name in pairs:
        low = getattr(owner, low_name)
        high = getattr(owner, high_name)
        if isinstance(low, PeriodicBC) != isinstance(high, PeriodicBC):
            raise ValueError(
                f"{low_name} and {high_name} must be a periodic pair: "
                "both PeriodicBC or neither"
            )

class Boundary2d:
    def __init__(self, left=None, right=None, top=None, bottom=None):
        """Outer conditions on the four faces of a 2-D physical grid.

        ``top`` is ``y_min`` and ``bottom`` is ``y_max``.
        """

        _assign_faces(
            self,
            (("left", left), ("right", right), ("top", top), ("bottom", bottom)),
            _PAIRS_2D,
        )

class Boundary3d:
    def __init__(self, left=None, right=None, top=None, bottom=None, front=None, back=None):
        """Outer conditions on the six faces of a 3-D physical grid.

        ``front`` is ``z_min`` and ``back`` is ``z_max``.  ``top`` is ``y_min``
        and ``bottom`` is ``y_max``.
        """

        _assign_faces(
            self,
            (
                ("left", left), ("right", right),
                ("top", top), ("bottom", bottom),
                ("front", front), ("back", back),
            ),
            _PAIRS_3D,
        )

def _region(ndim: int, bc, mask):
    label = "2-D" if ndim == 2 else "3-D"
    if not isinstance(bc, (DirichletBC, NeumannBC, RobinBC)):
        raise TypeError(f"a {label} boundary region accepts DirichletBC, NeumannBC, or RobinBC")
    mask = np.asarray(mask)
    if mask.ndim != ndim or mask.dtype != np.bool_:
        name = "BoundaryRegion2d" if ndim == 2 else "BoundaryRegion3d"
        raise ValueError(
            f"{name} mask must be a {label} boolean array, "
            f"got shape {mask.shape} and dtype {mask.dtype}"
        )
    return bc, mask

class BoundaryRegion2d:
    def __init__(self, bc, mask):
        """Obstacle cells on a 2-D physical grid and the interface condition."""
        self.bc, self.mask = _region(2, bc, mask)

class BoundaryRegion3d:
    def __init__(self, bc, mask):
        """Obstacle cells on a 3-D physical grid and the interface condition."""
        self.bc, self.mask = _region(3, bc, mask)

def iter_sides(outer: Boundary2d | Boundary3d):
    """Yield ``(name, axis, "lo"|"hi", condition)`` for each outer face.

    ``"lo"`` is the index-0 end of that axis and ``"hi"`` is the last index.
    """

    table = _SIDES_2D if isinstance(outer, Boundary2d) else _SIDES_3D
    for name, axis, end in table:
        yield name, axis, end, getattr(outer, name)

def _sample_face(value, shape: tuple[int, ...], axis: int, end: str) -> np.ndarray:
    """Sample one physical face. Accepts a scalar, the tangential face, or the full grid."""
    array = np.asarray(value, dtype=np.complex128)
    tangential = tuple(length for index, length in enumerate(shape) if index != axis)
    if array.shape == ():
        return np.full(tangential, array)
    if array.shape == tangential:
        return np.array(array, dtype=np.complex128, copy=True)
    if array.shape == shape:
        sl = [slice(None)] * len(shape)
        sl[axis] = 0 if end == "lo" else -1
        return np.array(array[tuple(sl)], dtype=np.complex128, copy=True)
    raise ValueError(
        f"boundary data shape {array.shape} does not match scalar, "
        f"face {tangential}, or grid {shape}"
    )

def _check_region_data(bc, mask: np.ndarray, shape: tuple[int, ...]) -> None:
    if isinstance(bc, (DirichletBC, NeumannBC)):
        array = np.asarray(bc.value)
        if array.shape not in ((), shape):
            raise ValueError(
                f"region data must be a scalar or an array of shape {shape}, got {array.shape}"
            )
        return
    for coeff_name, coeff in (("a", bc.a), ("b", bc.b), ("g", bc.g)):
        array = np.asarray(coeff)
        if array.shape not in ((), shape):
            raise ValueError(
                f"region RobinBC {coeff_name} must be a scalar or an array of "
                f"shape {shape}, got {array.shape}"
            )
    aa, bb = np.broadcast_arrays(
        np.asarray(bc.a, dtype=np.complex128),
        np.asarray(bc.b, dtype=np.complex128),
    )
    if not np.any(mask):
        singular = np.all((aa == 0.0) & (bb == 0.0))
    else:
        try:
            aa = np.broadcast_to(aa, mask.shape)
            bb = np.broadcast_to(bb, mask.shape)
        except ValueError as exc:
            raise ValueError("boundary region coefficients a and b are not broadcastable") from exc
        singular = np.any((aa[mask] == 0.0) & (bb[mask] == 0.0))
    if singular:
        raise ValueError("boundary region requires a or b to be nonzero on every enforced location")

def parse_boundaries(boundaries, *, ndim: int, shape: tuple[int, ...]):
    """Validate a scene boundary list and insert the default PML box if needed.

    Returns ``(stored_boundaries, outer, regions)``.  ``stored_boundaries``
    always contains exactly one outer configuration, followed by the regions in
    caller order.
    """

    if boundaries is None:
        boundaries = []
    if isinstance(boundaries, (str, bytes)) or not isinstance(boundaries, Sequence):
        raise TypeError("boundaries must be a sequence of boundary specifications")

    shape = tuple(shape)
    outer = None
    regions = []
    stored = []
    for item in boundaries:
        if ndim == 2 and isinstance(item, (Boundary3d, BoundaryRegion3d)):
            raise ValueError("a 2-D scene cannot use 3-D boundary specifications")
        if ndim == 3 and isinstance(item, (Boundary2d, BoundaryRegion2d)):
            raise ValueError("a 3-D scene cannot use 2-D boundary specifications")
        if isinstance(item, (Boundary2d, Boundary3d)):
            if outer is not None:
                raise ValueError("a scene accepts only one outer Boundary2d or Boundary3d")
            outer = item
            stored.append(item)
        elif isinstance(item, (BoundaryRegion2d, BoundaryRegion3d)):
            if item.mask.shape != shape:
                raise ValueError(
                    f"boundary region mask shape {item.mask.shape} does not match "
                    f"the physical grid {shape}"
                )
            _check_region_data(item.bc, item.mask, shape)
            regions.append(item)
            stored.append(item)
        else:
            raise TypeError(f"unsupported boundary specification {type(item).__name__}")

    if outer is None:
        outer = Boundary2d() if ndim == 2 else Boundary3d()
        stored.insert(0, outer)
    for name, axis, end, bc in iter_sides(outer):
        if isinstance(bc, (DirichletBC, NeumannBC)):
            _sample_face(bc.value, shape, axis, end)
        elif isinstance(bc, RobinBC):
            a = _sample_face(bc.a, shape, axis, end)
            b = _sample_face(bc.b, shape, axis, end)
            _sample_face(bc.g, shape, axis, end)
            if np.any((a == 0.0) & (b == 0.0)):
                raise ValueError(f"{name} boundary requires a or b to be nonzero on every enforced location")
    return tuple(stored), outer, tuple(regions)
