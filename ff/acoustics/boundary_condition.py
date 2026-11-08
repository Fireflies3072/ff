"""Boundary-condition types.  Geometry and parsing live in ``boundary.py``.

``PMLBC`` stores a face thickness and a :class:`~ff.acoustics.pml.PMLProfile`.
It does not store grid-sized coefficient arrays.  Omitting ``profile`` selects
CFS-PML with the historical parameter names.  Both Helmholtz solvers use that
profile.
"""

import numpy as np

from .pml import CFSPML, PMLProfile

def _as_numeric(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array

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
    def __init__(self, thickness=16, order=2, log_r=8.0, kappa_max=1.0, alpha_max_hz=40.0, profile=None):
        """PML on one face.  ``thickness`` is the number of exterior cells.

        With no ``profile``, the face uses :class:`~ff.acoustics.pml.CFSPML` and
        the historical parameter names.  Pass ``profile`` to select another
        stretching function, such as :class:`~ff.acoustics.pml.PolynomialPML`.
        """

        self.thickness = _pml_count(thickness, "thickness")
        if profile is None:
            self.profile = CFSPML(order=order, log_r=log_r, kappa_max=kappa_max, alpha_max_hz=alpha_max_hz)
            self._explicit_profile = False
        else:
            if not isinstance(profile, PMLProfile):
                raise TypeError("PMLBC profile must be a PMLProfile")
            self.profile = profile
            self._explicit_profile = True
        if isinstance(self.profile, CFSPML):
            self.order = self.profile.order
            self.log_r = self.profile.log_r
            self.kappa_max = self.profile.kappa_max
            self.alpha_max_hz = self.profile.alpha_max_hz
        else:
            self.order = None
            self.log_r = None
            self.kappa_max = None
            self.alpha_max_hz = None
