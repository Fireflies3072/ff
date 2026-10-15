from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class PMLSettings:
    """Complex-frequency-shifted PML placed around the physical grid.

    ``thickness`` is the number of grid cells on each side.  ``log_r`` sets the
    theoretical normal-incidence reflection to exp(-log_r).
    """

    thickness: int = 16
    order: int = 2
    log_r: float = 8.0
    kappa_max: float = 1.0
    alpha_max_hz: float = 40.0


class HelmholtzScene:
    def __init__(self, c: np.ndarray, rho: np.ndarray, dx: float,
                    pml: PMLSettings | None = None):
        """Physical medium, source map, and PML on a uniform Cartesian grid.

        Args:
            c: Map of sound speed (m/s)
            rho: Map of density (kg/m^3)
            dx: Grid spacing (m)
            pml: PML settings
        """

        # Validate input arrays
        if c.ndim not in (2, 3) or rho.ndim not in (2, 3):
            raise ValueError("c and rho must be 2-D or 3-D arrays")
        if c.shape != rho.shape:
            raise ValueError("c and rho must have the same shape")
        if np.any(c <= 0.0) or np.any(rho <= 0.0):
            raise ValueError("c and rho must be positive")
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        
        # Store input variables
        self.c = c
        self.rho = rho
        self.dx = dx

        # Build PML settings
        self.pml = PMLSettings() if pml is None else pml
        if self.pml.thickness < 0:
            raise ValueError("PML thickness must be non-negative")
        
        # Build derived variables
        self.ndim = c.ndim
        self.shape = c.shape

class HelmholtzMatrixSolver2d:
    def __init__(self, scene: HelmholtzScene, f: float):
        """Five-point conservative finite-difference Helmholtz operator at one frequency.

        The physical grid is surrounded by ``scene.pml.thickness`` cells of
        edge-replicated material.  The source map is placed on the physical cells
        only.  ``solve`` assembles the operator and factorizes it with a sparse LU.
        
        Args:
            scene: HelmholtzScene
            f: Frequency (Hz)
        """

        # Validate input variables
        if scene.ndim != 2:
            raise ValueError("HelmholtzMatrixSolver2d requires a 2-D scene")
        if f <= 0.0:
            raise ValueError("f must be positive")

        # Store input variables
        self.scene = scene
        self.f = f

        # Build derived variables
        self.omega = 2.0 * np.pi * f
        pml_thickness = scene.pml.thickness
        
        self.c = np.pad(scene.c, pml_thickness, mode="edge")
        self.rho = np.pad(scene.rho, pml_thickness, mode="edge")

        self.shape = self.c.shape
        self.n = int(np.prod(self.shape))
        self.c_ref = float(np.mean(scene.c))

        self.A = self.assemble()
        self.lu = spla.splu(self.A)

    def assemble(self) -> sp.csc_matrix:
        """Sparse matrix of the stretched operator at ``self.f``."""
        omega = self.omega
        nxt, nyt = self.shape
        dx = self.scene.dx
        buoy = 1.0 / self.rho
        bxp = 0.5 * (buoy + np.pad(buoy, ((0, 1), (0, 0)), mode="edge")[1:, :])
        bxm = 0.5 * (buoy + np.pad(buoy, ((1, 0), (0, 0)), mode="edge")[:-1, :])
        byp = 0.5 * (buoy + np.pad(buoy, ((0, 0), (0, 1)), mode="edge")[:, 1:])
        bym = 0.5 * (buoy + np.pad(buoy, ((0, 0), (1, 0)), mode="edge")[:, :-1])

        ix = np.arange(nxt, dtype=float)
        iy = np.arange(nyt, dtype=float)
        sx_n = self._stretch(nxt, ix)
        sx_p = self._stretch(nxt, ix + 0.5)
        sx_m = self._stretch(nxt, ix - 0.5)
        sy_n = self._stretch(nyt, iy)
        sy_p = self._stretch(nyt, iy + 0.5)
        sy_m = self._stretch(nyt, iy - 0.5)

        ce = bxp / (sx_n[:, None] * sx_p[:, None]) / dx ** 2
        cw = bxm / (sx_n[:, None] * sx_m[:, None]) / dx ** 2
        cn = byp / (sy_n[None, :] * sy_p[None, :]) / dx ** 2
        cs = bym / (sy_n[None, :] * sy_m[None, :]) / dx ** 2
        diag = -(ce + cw + cn + cs) + omega ** 2 / (self.rho * self.c ** 2)

        idx = np.arange(self.n).reshape(nxt, nyt)
        rows = [idx.ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel(),
                idx[:, :-1].ravel(), idx[:, 1:].ravel()]
        cols = [idx.ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel(),
                idx[:, 1:].ravel(), idx[:, :-1].ravel()]
        vals = [diag.ravel(), ce[:-1, :].ravel(), cw[1:, :].ravel(),
                cn[:, :-1].ravel(), cs[:, 1:].ravel()]
        matrix = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n, self.n), dtype=complex)
        return matrix.tocsc()

    def solve(self, source: np.ndarray) -> np.ndarray:
        """Pressure on the physical grid for the given source map at ``self.f``.
        
        Args:
            source: Source map (kg/m^3 s)
        
        Returns:
            Pressure on the physical grid
        """

        # Validate input variables
        if source.ndim != 2 or source.shape != self.scene.shape:
            raise ValueError("source must be a 2-D array with the same shape as the scene")

        source = np.pad(source, self.scene.pml.thickness, mode="constant")
        rhs = -source.reshape(-1)
        raw_p = self.lu.solve(rhs)
        return self.physical_field(raw_p)

    def physical_field(self, field: np.ndarray) -> np.ndarray:
        """Drop the PML frame and return the field on the physical grid."""
        w = self.scene.pml.thickness
        nx, ny = self.scene.shape
        if np.asarray(field).ndim == 1:
            grid = np.asarray(field).reshape(*self.shape)
            return grid[w:w + nx, w:w + ny]
        grid = np.asarray(field).reshape(*self.shape, -1)
        return grid[w:w + nx, w:w + ny, :]

    def _stretch(self, n_tot: int, positions: np.ndarray) -> np.ndarray:
        """CFS stretch s = kappa + sigma / (alpha - i omega) at grid positions."""
        npml = self.scene.pml.thickness
        if npml == 0:
            return np.ones(np.shape(positions), dtype=complex)
        pml = self.scene.pml
        length = npml * self.scene.dx
        left = np.clip((npml - positions) / npml, 0.0, 1.0)
        right = np.clip((positions - (n_tot - 1 - npml)) / npml, 0.0, 1.0)
        depth = np.maximum(left, right)
        sigma_max = (pml.order + 1) * self.c_ref * pml.log_r / (2.0 * length)
        sigma = sigma_max * depth ** pml.order
        kappa = 1.0 + (pml.kappa_max - 1.0) * depth ** pml.order
        alpha = 2.0 * np.pi * pml.alpha_max_hz * (1.0 - depth) * (depth > 0.0)
        return kappa + sigma / (alpha - 1j * self.omega)
