"""Conservative finite-difference Helmholtz solvers in two and three dimensions.

The discrete equation on the active fluid is

    div((1 / rho) grad p) + ω² / (rho c²) p = -source

with the ``exp(+j ω t)`` convention.  The factorized system uses ``scene.dtype``:
``float32`` is complex64 and ``float64`` is complex128.  Boundary conditions, including per-face
PML padding, come from ``scene.boundaries``.  See ``boundary.py`` for the
outward normals, array shapes, and precedence rules.  2-D arrays are
``(Ny, Nx)`` and 3-D arrays are ``(Nz, Ny, Nx)``.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .boundary import (
    DirichletBC,
    NeumannBC,
    PMLBC,
    PeriodicBC,
    RobinBC,
    _sample_face,
    iter_sides,
)
from .helmholtz_scene import HelmholtzSceneNumpy
from .pml import stretch_axis

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

@dataclass
class _System:
    A: sp.csc_matrix
    rhs_bc: np.ndarray
    source_weight: np.ndarray
    shape: tuple
    padding: tuple
    origin: tuple
    c: np.ndarray
    rho: np.ndarray
    c_ref: float
    n: int

@dataclass
class _RegionField:
    kind: str
    a: np.ndarray
    b: np.ndarray
    g: np.ndarray

@dataclass
class _Assembly:
    idx: np.ndarray
    active: np.ndarray
    obstacle: np.ndarray
    dirichlet: np.ndarray
    region_id: np.ndarray
    regions: list
    buoy: np.ndarray
    phys_shape: tuple
    padding: tuple
    c_ref: float
    omega: float
    dx: float
    diag: np.ndarray
    rhs_bc: np.ndarray
    off_rows: list = field(default_factory=list)
    off_cols: list = field(default_factory=list)
    off_vals: list = field(default_factory=list)

def _stretch_1d(positions, n_tot: int, pml_lo: PMLBC | None, pml_hi: PMLBC | None,
                c_ref: float, dx: float, omega: float) -> np.ndarray:
    """CFS stretch along one axis.  The stretch is 1 outside that axis's PML."""
    return stretch_axis(positions, n_tot, pml_lo, pml_hi, c_ref, dx, omega)

def _extend_face(value, phys_shape, axis: int, end: str, padding) -> np.ndarray:
    """Physical-face samples, edge-replicated through tangential PML layers."""
    face = _sample_face(value, tuple(phys_shape), axis, end)
    tangential = [index for index in range(len(phys_shape)) if index != axis]
    pad_width = [padding[index] for index in tangential]
    if any(low or high for low, high in pad_width):
        face = np.pad(face, pad_width, mode="edge")
    return face

def _add_diag(diag: np.ndarray, index, values) -> None:
    index = np.asarray(index, dtype=np.int64).ravel()
    values = np.asarray(values, dtype=np.complex128).ravel()
    if index.size:
        np.add.at(diag, index, values)

def _add_off(off_rows, off_cols, off_vals, rows, cols, vals) -> None:
    rows = np.asarray(rows, dtype=np.int64).ravel()
    if rows.size == 0:
        return
    off_rows.append(rows)
    off_cols.append(np.asarray(cols, dtype=np.int64).ravel())
    off_vals.append(np.asarray(vals, dtype=np.complex128).ravel())

def _flux_deltas(kind: str, coeff, a, b, g, dx: float):
    """Diagonal and RHS updates that replace one two-point flux."""
    coeff = np.asarray(coeff, dtype=np.complex128)
    diag = np.zeros(coeff.shape, dtype=np.complex128)
    rhs = np.zeros(coeff.shape, dtype=np.complex128)
    if kind == "neumann":
        rhs[...] = -coeff * dx * np.asarray(g, dtype=np.complex128)
        return diag, rhs
    if kind == "dirichlet":
        diag[...] = -2.0 * coeff
        rhs[...] = -2.0 * coeff * np.asarray(g, dtype=np.complex128)
        return diag, rhs
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    g = np.asarray(g, dtype=np.complex128)
    pure = b == 0.0
    soft = ~pure
    if np.any(pure):
        diag[pure] = -2.0 * coeff[pure]
        rhs[pure] = -2.0 * coeff[pure] * (g[pure] / a[pure])
    if np.any(soft):
        scale = coeff[soft] * dx / b[soft]
        diag[soft] = -scale * a[soft]
        rhs[soft] = -scale * g[soft]
    return diag, rhs

def _plane(ndim: int, axis: int, end: str) -> tuple:
    sl = [slice(None)] * ndim
    sl[axis] = 0 if end == "lo" else -1
    return tuple(sl)

def _interior(ndim: int, axis: int) -> tuple[tuple, tuple]:
    lo = [slice(None)] * ndim
    hi = [slice(None)] * ndim
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    return tuple(lo), tuple(hi)

def _along(values, ndim: int, axis: int, shape) -> np.ndarray:
    view = [1] * ndim
    view[axis] = np.size(values)
    return np.broadcast_to(np.reshape(np.asarray(values), view), shape)

def _padding_from(sides, ndim: int) -> tuple[tuple[int, int], ...]:
    pads = [[0, 0] for _ in range(ndim)]
    for _, axis, end, bc in sides:
        if isinstance(bc, PMLBC) and bc.thickness:
            pads[axis][0 if end == "lo" else 1] = bc.thickness
    return tuple(tuple(item) for item in pads)

def _faces_by_axis(sides, ndim: int) -> list[dict]:
    faces = [{"lo": None, "hi": None} for _ in range(ndim)]
    for _, axis, end, bc in sides:
        faces[axis][end] = bc
    return faces

def _embed(value, ext_shape, phys_slice) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.shape == ():
        return np.full(ext_shape, array, dtype=np.complex128)
    field = np.zeros(ext_shape, dtype=np.complex128)
    field[phys_slice] = array
    return field

def _region_fields(regions, ext_shape, phys_slice) -> list[_RegionField]:
    fields = []
    for region in regions:
        bc = region.bc
        if isinstance(bc, DirichletBC):
            kind, a, b, g = "dirichlet", 1.0, 0.0, bc.value
        elif isinstance(bc, NeumannBC):
            kind, a, b, g = "neumann", 0.0, 1.0, bc.value
        else:
            kind, a, b, g = "robin", bc.a, bc.b, bc.g
        fields.append(_RegionField(
            kind,
            _embed(a, ext_shape, phys_slice),
            _embed(b, ext_shape, phys_slice),
            _embed(g, ext_shape, phys_slice),
        ))
    return fields

def _mark_outer_dirichlet(asm: _Assembly, sides, dirichlet_value: np.ndarray) -> None:
    """Replace outer Dirichlet nodes, including Robin conditions with ``b = 0``."""
    ndim = asm.idx.ndim
    for _, axis, end, bc in sides:
        plane = _plane(ndim, axis, end)
        blocked = asm.obstacle[plane]
        if isinstance(bc, DirichletBC):
            face = _extend_face(bc.value, asm.phys_shape, axis, end, asm.padding)
        elif isinstance(bc, RobinBC):
            face_b = _extend_face(bc.b, asm.phys_shape, axis, end, asm.padding)
            pure = face_b == 0.0
            if not np.any(pure):
                continue
            face_a = _extend_face(bc.a, asm.phys_shape, axis, end, asm.padding)
            face_g = _extend_face(bc.g, asm.phys_shape, axis, end, asm.padding)
            face = np.zeros(face_g.shape, dtype=np.complex128)
            face[pure] = face_g[pure] / face_a[pure]
            blocked = blocked | ~pure
        else:
            continue
        free = ~blocked
        conflict = (
            asm.dirichlet[plane] & free
            & ~np.isclose(dirichlet_value[plane], face, rtol=1e-8, atol=1e-10)
        )
        if np.any(conflict):
            raise ValueError(
                "Dirichlet values disagree where boundary faces meet. "
                "Corners and edges need a consistent prescribed pressure."
            )
        asm.dirichlet[plane][free] = True
        dirichlet_value[plane][free] = face[free]

def _contribute_pair(asm: _Assembly, slice_a, slice_b, coeff_a: np.ndarray, coeff_b: np.ndarray) -> None:
    """Flux between two neighboring locations.

    ``coeff_a`` is the face coefficient in the equation at ``slice_a``, and
    ``coeff_b`` is the coefficient in the equation at ``slice_b``.  An obstacle
    neighbor is replaced by that region's interface condition.
    """

    idx_a = asm.idx[slice_a]
    idx_b = asm.idx[slice_b]
    coeff_at_a = coeff_a[slice_a]
    coeff_at_b = coeff_b[slice_b]
    active_a = asm.active[slice_a]
    active_b = asm.active[slice_b]
    obstacle_a = asm.obstacle[slice_a]
    obstacle_b = asm.obstacle[slice_b]
    row_a = active_a & ~asm.dirichlet[slice_a]
    row_b = active_b & ~asm.dirichlet[slice_b]
    couple_a = row_a & active_b & ~obstacle_b
    couple_b = row_b & active_a & ~obstacle_a
    if np.any(couple_a):
        _add_off(asm.off_rows, asm.off_cols, asm.off_vals, idx_a[couple_a], idx_b[couple_a], coeff_at_a[couple_a])
        _add_diag(asm.diag, idx_a[couple_a], -coeff_at_a[couple_a])
    if np.any(couple_b):
        _add_off(asm.off_rows, asm.off_cols, asm.off_vals, idx_b[couple_b], idx_a[couple_b], coeff_at_b[couple_b])
        _add_diag(asm.diag, idx_b[couple_b], -coeff_at_b[couple_b])

    region_a = asm.region_id[slice_a]
    region_b = asm.region_id[slice_b]
    for reg_i, region in enumerate(asm.regions):
        into_b = row_a & obstacle_b & (region_b == reg_i)
        if np.any(into_b):
            flat_b = np.asarray(idx_b[into_b], dtype=np.int64).ravel()
            ddiag, drhs = _flux_deltas(
                region.kind, coeff_at_a[into_b],
                region.a.ravel()[flat_b], region.b.ravel()[flat_b], region.g.ravel()[flat_b],
                asm.dx,
            )
            _add_diag(asm.diag, idx_a[into_b], ddiag)
            _add_diag(asm.rhs_bc, idx_a[into_b], drhs)
        into_a = row_b & obstacle_a & (region_a == reg_i)
        if np.any(into_a):
            flat_a = np.asarray(idx_a[into_a], dtype=np.int64).ravel()
            ddiag, drhs = _flux_deltas(
                region.kind, coeff_at_b[into_a],
                region.a.ravel()[flat_a], region.b.ravel()[flat_a], region.g.ravel()[flat_a],
                asm.dx,
            )
            _add_diag(asm.diag, idx_b[into_a], ddiag)
            _add_diag(asm.rhs_bc, idx_b[into_a], drhs)

def _assemble_axis(asm: _Assembly, faces: dict, axis: int) -> None:
    """Coefficients and fluxes along one coordinate axis."""
    ndim = asm.idx.ndim
    n_axis = asm.idx.shape[axis]
    pml_lo = faces["lo"] if isinstance(faces["lo"], PMLBC) else None
    pml_hi = faces["hi"] if isinstance(faces["hi"], PMLBC) else None
    nodes = np.arange(n_axis, dtype=float)
    s_node = _stretch_1d(nodes, n_axis, pml_lo, pml_hi, asm.c_ref, asm.dx, asm.omega)
    s_plus = _stretch_1d(nodes + 0.5, n_axis, pml_lo, pml_hi, asm.c_ref, asm.dx, asm.omega)
    s_outer_lo = _stretch_1d(np.array([-0.5]), n_axis, pml_lo, pml_hi, asm.c_ref, asm.dx, asm.omega)[0]
    s_outer_hi = _stretch_1d(np.array([n_axis - 0.5]), n_axis, pml_lo, pml_hi, asm.c_ref, asm.dx, asm.omega)[0]

    coeff_plus = np.zeros(asm.idx.shape, dtype=np.complex128)
    coeff_minus = np.zeros(asm.idx.shape, dtype=np.complex128)
    lo_slice, hi_slice = _interior(ndim, axis)
    lo_plane = _plane(ndim, axis, "lo")
    hi_plane = _plane(ndim, axis, "hi")
    dx2 = asm.dx ** 2
    if n_axis > 1:
        b_face = 0.5 * (asm.buoy[lo_slice] + asm.buoy[hi_slice])
        s_node_lo = _along(s_node[:-1], ndim, axis, b_face.shape)
        s_node_hi = _along(s_node[1:], ndim, axis, b_face.shape)
        s_face = _along(s_plus[:-1], ndim, axis, b_face.shape)
        coeff_plus[lo_slice] = b_face / (s_node_lo * s_face) / dx2
        coeff_minus[hi_slice] = b_face / (s_node_hi * s_face) / dx2

    coeff_minus[lo_plane] = asm.buoy[lo_plane] / (s_node[0] * s_outer_lo) / dx2
    coeff_plus[hi_plane] = asm.buoy[hi_plane] / (s_node[-1] * s_outer_hi) / dx2

    periodic = isinstance(faces["lo"], PeriodicBC)
    if periodic:
        b_face = 0.5 * (asm.buoy[lo_plane] + asm.buoy[hi_plane])
        coeff_plus[hi_plane] = b_face / dx2
        coeff_minus[lo_plane] = b_face / dx2

    if n_axis > 1:
        _contribute_pair(asm, lo_slice, hi_slice, coeff_plus, coeff_minus)
    if periodic:
        _contribute_pair(asm, hi_plane, lo_plane, coeff_plus, coeff_minus)
        return

    for end, coeff in (("lo", coeff_minus), ("hi", coeff_plus)):
        plane = lo_plane if end == "lo" else hi_plane
        bc = faces[end]
        fluid = asm.active[plane] & ~asm.dirichlet[plane]
        coeff_face = coeff[plane]
        if isinstance(bc, PMLBC):
            # Homogeneous Dirichlet ghost just outside the outer PML cell.
            _add_diag(asm.diag, asm.idx[plane][fluid], -coeff_face[fluid])
        elif isinstance(bc, NeumannBC) and np.any(fluid):
            g = _extend_face(bc.value, asm.phys_shape, axis, end, asm.padding)
            ddiag, drhs = _flux_deltas("neumann", coeff_face[fluid], 0.0, 1.0, g[fluid], asm.dx)
            _add_diag(asm.diag, asm.idx[plane][fluid], ddiag)
            _add_diag(asm.rhs_bc, asm.idx[plane][fluid], drhs)
        elif isinstance(bc, RobinBC) and np.any(fluid):
            a = _extend_face(bc.a, asm.phys_shape, axis, end, asm.padding)
            b = _extend_face(bc.b, asm.phys_shape, axis, end, asm.padding)
            g = _extend_face(bc.g, asm.phys_shape, axis, end, asm.padding)
            ddiag, drhs = _flux_deltas("robin", coeff_face[fluid], a[fluid], b[fluid], g[fluid], asm.dx)
            _add_diag(asm.diag, asm.idx[plane][fluid], ddiag)
            _add_diag(asm.rhs_bc, asm.idx[plane][fluid], drhs)

def _finalize(asm: _Assembly, scene: HelmholtzSceneNumpy, c, rho, dirichlet_value, padding, origin) -> _System:
    ext_shape = asm.idx.shape
    n = int(np.prod(ext_shape))
    mass = asm.omega ** 2 / (rho * c ** 2)
    fluid = (asm.active & ~asm.dirichlet).ravel()
    asm.diag[fluid] += mass.ravel()[fluid]

    source_weight = np.ones(n, dtype=np.float64)
    obstacle = asm.obstacle.ravel()
    dirichlet = asm.dirichlet.ravel()
    asm.diag[obstacle] = 1.0
    asm.rhs_bc[obstacle] = 0.0
    source_weight[obstacle] = 0.0
    asm.diag[dirichlet] = 1.0
    asm.rhs_bc[dirichlet] = dirichlet_value.ravel()[dirichlet]
    source_weight[dirichlet] = 0.0

    if asm.off_rows:
        rows = np.concatenate(asm.off_rows)
        cols = np.concatenate(asm.off_cols)
        vals = np.concatenate(asm.off_vals)
        keep = ~(obstacle[rows] | dirichlet[rows])
        if np.any(keep):
            off = sp.coo_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n), dtype=np.complex128)
        else:
            off = sp.coo_matrix((n, n), dtype=np.complex128)
    else:
        off = sp.coo_matrix((n, n), dtype=np.complex128)
    return _System(
        A=(sp.diags(asm.diag) + off).tocsc(),
        rhs_bc=asm.rhs_bc,
        source_weight=source_weight,
        shape=ext_shape,
        padding=padding,
        origin=origin,
        c=c,
        rho=rho,
        c_ref=asm.c_ref,
        n=n,
    )

def _assemble_system(scene: HelmholtzSceneNumpy, omega: float) -> _System:
    sides = tuple(iter_sides(scene.outer))
    ndim = scene.ndim
    phys_shape = tuple(scene.shape)
    padding = _padding_from(sides, ndim)
    origin = tuple(item[0] for item in padding)
    c = np.pad(scene.c, padding, mode="edge")
    rho = np.pad(scene.rho, padding, mode="edge")
    ext_shape = c.shape
    n = int(np.prod(ext_shape))
    phys_slice = tuple(slice(origin[axis], origin[axis] + phys_shape[axis]) for axis in range(ndim))

    region_id = np.full(ext_shape, -1, dtype=np.int32)
    phys_ids = np.full(phys_shape, -1, dtype=np.int32)
    for reg_i, region in enumerate(scene.regions):
        phys_ids[region.mask] = reg_i
    region_id[phys_slice] = phys_ids

    asm = _Assembly(
        idx=np.arange(n).reshape(ext_shape),
        active=~(region_id >= 0),
        obstacle=region_id >= 0,
        dirichlet=np.zeros(ext_shape, dtype=bool),
        region_id=region_id,
        regions=_region_fields(scene.regions, ext_shape, phys_slice),
        buoy=1.0 / rho,
        phys_shape=phys_shape,
        padding=padding,
        c_ref=float(np.mean(scene.c)),
        omega=omega,
        dx=scene.dx,
        diag=np.zeros(n, dtype=np.complex128),
        rhs_bc=np.zeros(n, dtype=np.complex128),
    )
    dirichlet_value = np.zeros(ext_shape, dtype=np.complex128)
    _mark_outer_dirichlet(asm, sides, dirichlet_value)
    for axis, faces in enumerate(_faces_by_axis(sides, ndim)):
        _assemble_axis(asm, faces, axis)
    return _finalize(asm, scene, c, rho, dirichlet_value, padding, origin)

class _HelmholtzSolver:
    def _setup(self, scene: HelmholtzSceneNumpy, f: float) -> None:
        self.scene = scene
        self.f = _positive_scalar(f, "f")
        self.omega = 2.0 * np.pi * self.f
        system = _assemble_system(scene, self.omega)
        self.c = system.c
        self.rho = system.rho
        self.shape = system.shape
        self.padding = system.padding
        self._origin = system.origin
        self.n = system.n
        self.c_ref = system.c_ref
        self.dtype = np.dtype(scene.dtype)
        if self.dtype == np.dtype(np.float32):
            self.complex_dtype = np.dtype(np.complex64)
        elif self.dtype == np.dtype(np.float64):
            self.complex_dtype = np.dtype(np.complex128)
        else:
            raise TypeError(f"scene dtype {self.dtype} is not supported")
        self.A = system.A.astype(self.complex_dtype, copy=False)
        self.lu = spla.splu(self.A)
        self._rhs_bc = np.asarray(system.rhs_bc, dtype=self.complex_dtype)
        self._source_weight = np.asarray(system.source_weight, dtype=self.dtype)

    def assemble(self) -> sp.csc_matrix:
        """Sparse matrix of the operator at ``self.f``, including boundary rows."""
        return self.A

    def solve(self, source: np.ndarray | None = None) -> np.ndarray:
        """Pressure on the physical grid.  An omitted source is zero."""
        if source is None:
            src = np.zeros(self.scene.shape, dtype=self.complex_dtype)
        else:
            src = np.asarray(source)
            label = "2-D" if self.scene.ndim == 2 else "3-D"
            if src.ndim != self.scene.ndim or src.shape != self.scene.shape:
                raise ValueError(f"source must be a {label} array with the same shape as the scene")
            src = np.asarray(src, dtype=self.complex_dtype)
        padded = np.pad(src, self.padding, mode="constant")
        rhs = -padded.reshape(-1) * self._source_weight + self._rhs_bc
        return self.physical_field(self.lu.solve(rhs))

    def physical_field(self, field: np.ndarray) -> np.ndarray:
        """Drop the PML frame.  Extra trailing axes are preserved."""
        field = np.asarray(field)
        ndim = self.scene.ndim
        grid = field.reshape(self.shape) if field.ndim == 1 else field.reshape(*self.shape, -1)
        sl = tuple(
            slice(self._origin[axis], self._origin[axis] + self.scene.shape[axis])
            for axis in range(ndim)
        )
        extra = (slice(None),) * (grid.ndim - ndim)
        return grid[sl + extra]

class HelmholtzMatrixSolverNumpy2d(_HelmholtzSolver):
    def __init__(self, scene: HelmholtzSceneNumpy, f: float):
        """Five-point Helmholtz operator at one frequency.

        The operator is factorized once and reused for every source.  ``solve()``
        with no source is the boundary-driven problem.  The CFS-PML stretch is
        ``κ + σ / (α + j ω)``.  ``padding`` is ``((top, bottom), (left, right))``.
        """

        if scene.ndim != 2:
            raise ValueError("HelmholtzMatrixSolverNumpy2d requires a 2-D scene")
        self._setup(scene, f)

class HelmholtzMatrixSolverNumpy3d(_HelmholtzSolver):
    def __init__(self, scene: HelmholtzSceneNumpy, f: float):
        """Seven-point Helmholtz operator at one frequency.

        The operator is factorized once and reused for every source.  ``solve()``
        with no source is the boundary-driven problem.  The CFS-PML stretch is
        ``κ + σ / (α + j ω)``.  ``padding`` is
        ``((front, back), (top, bottom), (left, right))``.  ``front`` is
        ``z_min``, ``back`` is ``z_max``, ``top`` is ``y_min``, and ``bottom``
        is ``y_max``.
        """

        if scene.ndim != 3:
            raise ValueError("HelmholtzMatrixSolverNumpy3d requires a 3-D scene")
        self._setup(scene, f)
