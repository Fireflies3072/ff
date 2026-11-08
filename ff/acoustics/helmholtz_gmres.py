"""PyTorch GMRES Helmholtz solvers for a ``HelmholtzSceneTorch``.

The discrete pressure satisfies

    A p = -source

with the same source sign as ``HelmholtzMatrixSolverNumpy2d``.  The PML profile
is the one stored on each face.  An omitted profile is CFS-PML.  CFS-PML with
the five- or seven-point stencil uses the matrix flux and a zero ghost.  A
polynomial profile, and spectral CFS-PML, use the j-Wave Helmholtz operator: a
PML-modified Laplacian, the density gradient term, and ``omega**2 / c**2``.
The scene has no absorption, so that term is zero.  ``omega = 2 pi f``.
Derivatives are Fourier spectral when ``dispersion`` is omitted or continuous,
or the second-order staggered stencil for ``SecondOrderFivePointDispersion2d``
and ``SecondOrderSevenPointDispersion3d``.  PML cells are added outside the
scene, using the outer PML thickness on each face, and ``solve`` removes them.
Array layout follows the scene:
``(Ny, Nx)`` or ``(Nz, Ny, Nx)``, with one ``dx``.

The solver is an ``nn.Module``.  ``c``, ``rho``, and ``omega`` are buffers, so
``to(device)`` moves them.  Gradients of ``solve`` use the adjoint of the
converged system instead of the GMRES iterations.  The real dtype is
``scene.dtype``: ``float32`` uses ``complex64`` and ``float64`` uses ``complex128``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .boundary import PMLBC, iter_sides
from .dispersion import (
    ContinuousDispersion2d,
    ContinuousDispersion3d,
    SecondOrderFivePointDispersion2d,
    SecondOrderSevenPointDispersion3d,
)
from .helmholtz_scene import HelmholtzSceneTorch
from .pml import CFSPML, PolynomialPML, build_pml_coefficients

# Second-order Fornberg coefficients on the unit grid, before dividing by dx.
# stagger +1/2, -1/2, and 0.  A trailing or leading zero keeps the stencil
# length odd so circular correlation stays centered (same padding as j-Wave).
_FD_COEFF = {
    0.5: ( -1.0, 1.0, 0.0),
    -0.5: (0.0, -1.0, 1.0),
    0.0: (-0.5, 0.0, 0.5),
}

def _pad_tensor(field: torch.Tensor, padding: tuple, mode: str) -> torch.Tensor:
    """Pad ``field`` outside the physical grid.  ``mode`` is ``edge`` or ``constant``."""
    result = field
    for axis, (lo, hi) in enumerate(padding):
        lo = int(lo)
        hi = int(hi)
        if lo == 0 and hi == 0:
            continue
        pieces = []
        if lo:
            pieces.append(_edge_slab(result, axis, lo, "lo") if mode == "edge" else _zero_slab(result, axis, lo))
        pieces.append(result)
        if hi:
            pieces.append(_edge_slab(result, axis, hi, "hi") if mode == "edge" else _zero_slab(result, axis, hi))
        result = torch.cat(pieces, dim=axis)
    return result

def _edge_slab(field: torch.Tensor, axis: int, count: int, end: str) -> torch.Tensor:
    index = [slice(None)] * field.ndim
    index[axis] = slice(0, 1) if end == "lo" else slice(-1, None)
    edge = field[tuple(index)]
    repeats = [1] * field.ndim
    repeats[axis] = count
    return edge.repeat(*repeats)

def _zero_slab(field: torch.Tensor, axis: int, count: int) -> torch.Tensor:
    shape = list(field.shape)
    shape[axis] = count
    return field.new_zeros(shape)

def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float32, torch.complex64):
        return torch.float32
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    raise TypeError(f"unsupported dtype {dtype}")

def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if _real_dtype(dtype) == torch.float32 else torch.complex128

def _prepare_source(source, shape: tuple, device, complex_dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(source, device=device)
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"source must have shape {shape}")
    if tensor.is_complex():
        return tensor.to(dtype=complex_dtype)
    return tensor.to(dtype=_real_dtype(complex_dtype)).to(dtype=complex_dtype)

def _k_axis(n: int, dx: float, *, real: bool, device, dtype: torch.dtype) -> torch.Tensor:
    if real:
        freq = torch.fft.rfftfreq(n, d=dx, device=device, dtype=dtype)
    else:
        freq = torch.fft.fftfreq(n, d=dx, device=device, dtype=dtype)
    return freq * (2.0 * math.pi)

def _apply_multiplier(spectrum: torch.Tensor, multiplier: torch.Tensor, axis: int) -> torch.Tensor:
    view = [1] * spectrum.ndim
    view[axis] = multiplier.shape[0]
    return spectrum * multiplier.view(view)

def _spectral_deriv(field: torch.Tensor, axis: int, dx: float, stagger: float, correct_nyquist: bool) -> torch.Tensor:
    real = not field.is_complex()
    n = field.shape[axis]
    complex_dtype = _complex_dtype(field.dtype)
    k = _k_axis(n, dx, real=real, device=field.device, dtype=_real_dtype(field.dtype)).to(complex_dtype)
    multiplier = (1j * k) * torch.exp(1j * k * (stagger * dx))
    if correct_nyquist and n % 2 == 0:
        multiplier = multiplier.clone()
        multiplier[n // 2] = 0
    multiplier = multiplier.to(complex_dtype)
    if real:
        spectrum = torch.fft.rfft(field, dim=axis)
        filtered = _apply_multiplier(spectrum, multiplier, axis)
        return torch.fft.irfft(filtered, n=n, dim=axis)
    spectrum = torch.fft.fft(field, dim=axis)
    filtered = _apply_multiplier(spectrum, multiplier, axis)
    return torch.fft.ifft(filtered, dim=axis)

def _spectral_gradient(field: torch.Tensor, dx: tuple, stagger: float, correct_nyquist: bool) -> torch.Tensor:
    parts = [
        _spectral_deriv(field, axis, dx[axis], stagger, correct_nyquist)
        for axis in range(field.ndim)
    ]
    return torch.stack(parts, dim=-1)

def _fd_deriv(field: torch.Tensor, axis: int, dx: float, stagger: float) -> torch.Tensor:
    """Centered circular correlation, matching j-Wave's wrapped ``reflection_conv``."""
    kernel = field.new_tensor(_FD_COEFF[stagger], dtype=_real_dtype(field.dtype)) / dx
    kernel = kernel.to(dtype=field.dtype)
    pad = (kernel.numel() - 1) // 2
    out = torch.zeros_like(field)
    for index in range(kernel.numel()):
        offset = index - pad
        out = out + kernel[index] * torch.roll(field, shifts=-offset, dims=axis)
    return out

def _fd_gradient(field: torch.Tensor, dx: tuple, stagger: float) -> torch.Tensor:
    parts = [_fd_deriv(field, axis, dx[axis], stagger) for axis in range(field.ndim)]
    return torch.stack(parts, dim=-1)

def helmholtz_operator(
    field: torch.Tensor,
    sound_speed: torch.Tensor,
    density: torch.Tensor,
    attenuation: torch.Tensor,
    omega: torch.Tensor,
    *,
    dx: tuple,
    pml: tuple,
    kind: str,
    density_is_field: bool,
) -> torch.Tensor:
    """Apply the j-Wave Helmholtz operator.  ``field`` is complex and shaped like the grid."""
    if kind == "spectral":
        gradient = _spectral_gradient(field, dx, stagger=0.5, correct_nyquist=False)
        modified = gradient * pml[0]
        diagonal = torch.stack([
            _spectral_deriv(modified[..., axis], axis, dx[axis], -0.5, False)
            for axis in range(modified.shape[-1])
        ], dim=-1)
    elif kind == "fd":
        gradient = _fd_gradient(field, dx, stagger=0.5)
        modified = gradient * pml[0]
        diagonal = torch.stack([
            _fd_deriv(modified[..., axis], axis, dx[axis], -0.5)
            for axis in range(modified.shape[-1])
        ], dim=-1)
    else:
        raise ValueError(f"unknown dispersion kind {kind}")
    laplacian = (diagonal * pml[1]).sum(dim=-1)

    if density_is_field:
        if kind == "spectral":
            density_gradient = _spectral_gradient(density, dx, stagger=0.5, correct_nyquist=True)
            product = modified * density_gradient
            shifted = []
            for axis, step in enumerate(dx):
                component = product[..., axis]
                n = component.shape[axis]
                wave = _k_axis(
                    n, step, real=False, device=component.device, dtype=_real_dtype(component.dtype)
                ).to(_complex_dtype(component.dtype))
                phase = torch.exp(1j * wave * (-0.5 * step))
                spectrum = torch.fft.fft(component, dim=axis)
                shifted.append(torch.fft.ifft(_apply_multiplier(spectrum, phase, axis), dim=axis))
            density_term = torch.stack(shifted, dim=-1).sum(dim=-1) / density
        else:
            density_gradient = _fd_gradient(density, dx, stagger=0.0)
            density_term = (modified * density_gradient).sum(dim=-1) / density
    else:
        density_term = 0.0

    complex_dtype = _complex_dtype(field.dtype)
    two_pi = attenuation.new_tensor(2.0 * math.pi)
    neper = attenuation.new_tensor(20.0 * math.log10(math.e))
    scale = attenuation.new_tensor(100.0) * ((attenuation.new_tensor(1e-6) / two_pi) ** 2) / neper
    alpha = (attenuation * scale).to(complex_dtype)
    omega_c = omega.to(complex_dtype)
    speed = sound_speed.to(complex_dtype)
    two_j = field.new_tensor(2j, dtype=complex_dtype)
    wavenumber = (omega_c / speed) ** 2 + two_j * (omega_c ** 3) * alpha / speed
    return ((laplacian - density_term) + wavenumber * field).to(complex_dtype)

def _pad_one(body: torch.Tensor, axis: int, where: str) -> torch.Tensor:
    """Append or prepend one zero plane along ``axis``."""
    shape = list(body.shape)
    shape[axis] = 1
    zero = body.new_zeros(shape)
    if where == "end":
        return torch.cat((body, zero), dim=axis)
    return torch.cat((zero, body), dim=axis)

def _place_plane(plane: torch.Tensor, axis: int, index: int, length: int) -> torch.Tensor:
    """Embed a one-cell plane at ``index`` along ``axis`` in a length-``length`` tensor."""
    parts = []
    shape = list(plane.shape)
    if index:
        shape[axis] = index
        parts.append(plane.new_zeros(shape))
    parts.append(plane)
    after = length - index - 1
    if after:
        shape[axis] = after
        parts.append(plane.new_zeros(shape))
    return torch.cat(parts, dim=axis)

def _vdot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sum(left.conj() * right)

def _norm(field: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.real(_vdot(field, field)))

def _gmres(matvec, rhs: torch.Tensor, guess: Optional[torch.Tensor], tol: float, atol: float, restart: int, maxiter: int) -> torch.Tensor:
    """Restarted GMRES.  Stops on ``||Ax - b|| <= max(tol ||b||, atol)``."""
    solution = torch.zeros_like(rhs) if guess is None else guess.to(dtype=rhs.dtype)
    rhs_norm = _norm(rhs)
    threshold = torch.maximum(rhs_norm * tol, rhs.new_tensor(atol, dtype=rhs_norm.dtype))
    restart = max(1, min(int(restart), rhs.numel()))
    for _cycle in range(int(maxiter)):
        residual = rhs - matvec(solution)
        beta = _norm(residual)
        if beta <= threshold or not torch.isfinite(beta):
            break
        basis = [residual / beta]
        hessenberg = rhs.new_zeros((restart + 1, restart))
        steps = 0
        for column in range(restart):
            worked = matvec(basis[column])
            for row in range(column + 1):
                coeff = _vdot(basis[row], worked)
                hessenberg[row, column] = coeff
                worked = worked - coeff * basis[row]
            next_norm = _norm(worked)
            hessenberg[column + 1, column] = next_norm
            steps = column + 1
            if next_norm <= torch.finfo(_real_dtype(rhs.dtype)).eps * beta:
                break
            basis.append(worked / next_norm)
        block = hessenberg[: steps + 1, :steps]
        right = rhs.new_zeros(steps + 1)
        right[0] = beta.to(rhs.dtype)
        update = torch.linalg.lstsq(block, right.unsqueeze(1)).solution.squeeze(1)
        for column in range(steps):
            solution = solution + update[column] * basis[column]
    return solution

class _HelmholtzSolve(torch.autograd.Function):
    """GMRES solve with an implicit adjoint.  Iterations are not differentiated."""

    @staticmethod
    def forward(ctx, source, sound_speed, density, omega, guess, solver):
        ctx.solver = solver
        with torch.no_grad():
            pressure = solver._solve_forward(source, sound_speed, density, omega, guess)
        ctx.save_for_backward(
            pressure.detach(),
            source.detach(),
            sound_speed.detach(),
            density.detach(),
            omega.detach(),
        )
        return pressure

    @staticmethod
    def backward(ctx, grad_pressure):
        if grad_pressure is None:
            return None, None, None, None, None, None
        solver = ctx.solver
        pressure, source, sound_speed, density, omega = ctx.saved_tensors
        grad_pressure = grad_pressure.to(dtype=pressure.dtype)
        adjoint = solver._adjoint_unknown(grad_pressure, sound_speed, density, omega)

        with torch.enable_grad():
            speed = sound_speed.detach().requires_grad_(True)
            rho = density.detach().requires_grad_(True)
            omega_s = omega.detach().requires_grad_(True)
            src = source.detach().requires_grad_(True)
            residual = -src - solver._operator(pressure.detach(), speed, rho, omega_s)
            grads = torch.autograd.grad(
                residual,
                (src, speed, rho, omega_s),
                adjoint.detach(),
                allow_unused=True,
            )

        def _or_zero(grad, like):
            if grad is None:
                return torch.zeros_like(like)
            return grad

        return (
            _or_zero(grads[0], source),
            _or_zero(grads[1], sound_speed),
            _or_zero(grads[2], density),
            _or_zero(grads[3], omega),
            None,
            None,
        )

class _HelmholtzGMRESSolver(nn.Module):
    def __init__(
        self,
        scene: HelmholtzSceneTorch,
        f,
        dispersion=None,
        *,
        tol: float = 1e-5,
        atol: float = 0.0,
        restart: int = 20,
        maxiter: int = 500
    ):
        super().__init__()
        if not isinstance(scene, HelmholtzSceneTorch):
            raise TypeError("the GMRES solver requires a HelmholtzSceneTorch")
        if scene.ndim != self._ndim:
            raise ValueError(f"scene.ndim must be {self._ndim}")
        self.scene = scene
        self.shape = tuple(scene.shape)
        pads = [[0, 0] for _ in range(scene.ndim)]
        for _, axis, end, bc in iter_sides(scene.outer):
            if isinstance(bc, PMLBC) and bc.thickness:
                pads[axis][0 if end == "lo" else 1] = int(bc.thickness)
        self.padding = tuple(tuple(item) for item in pads)
        self.extended_shape = tuple(
            length + lo + hi for length, (lo, hi) in zip(self.shape, self.padding)
        )
        thicknesses = [side for pair in self.padding for side in pair]
        self.pml_size = float(thicknesses[0]) if all(side == thicknesses[0] for side in thicknesses) else None
        self.dx = (float(scene.dx),) * scene.ndim
        if isinstance(f, bool):
            raise ValueError("f must be positive")
        try:
            frequency = float(f)
        except (TypeError, ValueError):
            raise ValueError("f must be positive") from None
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("f must be positive")
        self.f = frequency
        if dispersion is None:
            dispersion = ContinuousDispersion2d() if self._ndim == 2 else ContinuousDispersion3d()
        allowed = (
            (ContinuousDispersion2d, SecondOrderFivePointDispersion2d)
            if self._ndim == 2
            else (ContinuousDispersion3d, SecondOrderSevenPointDispersion3d)
        )
        if not isinstance(dispersion, allowed):
            raise TypeError("dispersion must be a dispersion class for this solver")
        self.dispersion = dispersion
        self.kind = "fd" if isinstance(
            dispersion, (SecondOrderFivePointDispersion2d, SecondOrderSevenPointDispersion3d)
        ) else "spectral"
        self.tol = float(tol)
        self.atol = float(atol)
        self.restart = int(restart)
        self.maxiter = int(maxiter)
        self._pml_cache = {}
        pml_faces = [bc for _, _, _, bc in iter_sides(scene.outer) if isinstance(bc, PMLBC) and bc.thickness]
        self._pml_uses_cfs = bool(pml_faces) and all(isinstance(bc.profile, CFSPML) for bc in pml_faces)
        self.dtype = scene.dtype
        if self.dtype == torch.float32:
            self.complex_dtype = torch.complex64
        elif self.dtype == torch.float64:
            self.complex_dtype = torch.complex128
        else:
            raise TypeError(f"scene dtype {self.dtype} is not supported")
        self.register_buffer("c", scene.c.detach().clone())
        self.register_buffer("rho", scene.rho.detach().clone())
        omega = torch.tensor(2.0 * math.pi * self.f, dtype=self.dtype, device=scene.c.device)
        self.register_buffer("omega", omega)
        if scene.c.requires_grad:
            self.c.requires_grad_(True)
        if scene.rho.requires_grad:
            self.rho.requires_grad_(True)

    def _pad_edge(self, field: torch.Tensor) -> torch.Tensor:
        return _pad_tensor(field, self.padding, "edge")

    def _pad_constant(self, field: torch.Tensor) -> torch.Tensor:
        return _pad_tensor(field, self.padding, "constant")

    def _pml(self, device):
        real_dtype = self.dtype
        # Stagger is half a cell in index units.  ``dx / 2`` would mix a physical
        # length into that index and change the PML when ``dx != 1``.
        medium = ()
        if self._pml_uses_cfs:
            medium = (float(self.omega.detach().item()), float(self.c.detach().mean().item()))
        key = (str(device), str(real_dtype), self.padding, self._profile_key(), medium)
        if key not in self._pml_cache:
            omega = medium[0] if medium else 0.0
            c_ref = medium[1] if medium else 1.0
            common = dict(
                boundaries=self.scene.outer,
                shape=self.extended_shape,
                spacing=self.dx,
                omega=omega,
                c_ref=c_ref,
                backend="torch",
                dtype=real_dtype,
                device=device,
            )
            if self._pml_uses_cfs and self.kind == "fd":
                # Node and face stretch for ``1 / (s(j) s(j+1/2))``.  The sample at
                # ``-1/2`` is the zero-ghost face just outside the first cell.
                self._pml_cache[key] = (
                    build_pml_coefficients(**common, stagger=0.0).stretch,
                    build_pml_coefficients(**common, stagger=0.5).stretch,
                    build_pml_coefficients(**common, stagger=-0.5).stretch,
                )
            else:
                positive = build_pml_coefficients(**common, stagger=0.5)
                negative = build_pml_coefficients(**common, stagger=-0.5)
                attribute = "inverse_stretch" if self._pml_uses_cfs else "stretch"
                self._pml_cache[key] = (getattr(positive, attribute), getattr(negative, attribute))
        return self._pml_cache[key]

    def _profile_key(self):
        parts = []
        for _, _, _, bc in iter_sides(self.scene.outer):
            if not isinstance(bc, PMLBC):
                parts.append(("other", type(bc).__name__))
                continue
            profile = bc.profile
            if isinstance(profile, PolynomialPML):
                parts.append(("poly", bc.thickness, profile.alpha_max, profile.order, bc._explicit_profile))
            else:
                parts.append((
                    "cfs", bc.thickness, profile.order, profile.log_r,
                    profile.kappa_max, profile.alpha_max_hz, bc._explicit_profile,
                ))
        return tuple(parts)

    def _operator(self, field, sound_speed, density, omega) -> torch.Tensor:
        if self._pml_uses_cfs and self.kind == "fd":
            # Face j to j+1 carries b / (s(j) s(j+1/2)) / dx^2.  The outer neighbor is a zero ghost.
            s_node, s_plus, s_minus = self._pml(field.device)
            complex_dtype = field.dtype
            buoy = (1.0 / density).to(complex_dtype)
            out = field.new_zeros(field.shape)
            for axis, step in enumerate(self.dx):
                length = field.shape[axis]
                dx2 = step * step
                node = s_node[..., axis]
                plus = s_plus[..., axis]
                minus = s_minus[..., axis]
                if length > 1:
                    left = field.narrow(axis, 0, length - 1)
                    right = field.narrow(axis, 1, length - 1)
                    averaged = 0.5 * (buoy.narrow(axis, 0, length - 1) + buoy.narrow(axis, 1, length - 1))
                    face = plus.narrow(axis, 0, length - 1)
                    forward = averaged / (node.narrow(axis, 0, length - 1) * face) / dx2
                    backward = averaged / (node.narrow(axis, 1, length - 1) * face) / dx2
                    out = out + _pad_one(forward * (right - left), axis, "end")
                    out = out + _pad_one(backward * (left - right), axis, "start")
                ghost_lo = buoy.narrow(axis, 0, 1) / (node.narrow(axis, 0, 1) * minus.narrow(axis, 0, 1)) / dx2
                ghost_hi = buoy.narrow(axis, length - 1, 1) / (
                    node.narrow(axis, length - 1, 1) * plus.narrow(axis, length - 1, 1)
                ) / dx2
                out = out - _place_plane(ghost_lo * field.narrow(axis, 0, 1), axis, 0, length)
                out = out - _place_plane(ghost_hi * field.narrow(axis, length - 1, 1), axis, length - 1, length)
            speed = sound_speed.to(complex_dtype)
            rho = density.to(complex_dtype)
            return out + (omega.to(complex_dtype) ** 2) / (rho * speed ** 2) * field
        attenuation = sound_speed.new_zeros(())
        return helmholtz_operator(
            field,
            sound_speed,
            density,
            attenuation,
            omega,
            dx=self.dx,
            pml=self._pml(field.device),
            kind=self.kind,
            density_is_field=True,
        )

    def _solve_forward(self, source, sound_speed, density, omega, guess):
        def matvec(vector):
            return self._operator(vector, sound_speed, density, omega)

        initial = None if guess is None else guess.detach()
        return _gmres(matvec, -source, initial, self.tol, self.atol, self.restart, self.maxiter)

    def _adjoint_unknown(self, grad_pressure, sound_speed, density, omega):
        speed = sound_speed.detach()
        rho = density.detach()
        frequency = omega.detach()

        def adjoint_matvec(vector):
            with torch.enable_grad():
                unknown = torch.zeros_like(vector, requires_grad=True)
                image = self._operator(unknown, speed, rho, frequency)
                (transposed,) = torch.autograd.grad(image, unknown, vector)
            return transposed.detach()

        return _gmres(
            adjoint_matvec,
            grad_pressure,
            None,
            self.tol,
            self.atol,
            self.restart,
            self.maxiter,
        )

    def apply(self, field) -> torch.Tensor:
        """Apply ``A`` on the extended grid, including the exterior PML."""
        field_t = _prepare_source(
            field,
            self.extended_shape,
            field.device if torch.is_tensor(field) else self.c.device,
            self.complex_dtype,
        )
        return self._operator(field_t, self._pad_edge(self.c), self._pad_edge(self.rho), self.omega)

    def solve(self, source=None, guess=None) -> torch.Tensor:
        """Physical pressure.  The exterior PML is removed.  An omitted source is zero."""
        complex_dtype = self.complex_dtype
        if source is None:
            source_t = torch.zeros(self.shape, dtype=complex_dtype, device=self.c.device)
        else:
            source_t = _prepare_source(source, self.shape, self.c.device, complex_dtype)
        source_ext = self._pad_constant(source_t)
        if guess is None:
            guess_ext = torch.zeros_like(source_ext)
        else:
            guess_t = _prepare_source(guess, self.shape, source_t.device, complex_dtype).detach()
            guess_ext = self._pad_constant(guess_t)
        pressure_ext = _HelmholtzSolve.apply(
            source_ext, self._pad_edge(self.c), self._pad_edge(self.rho), self.omega, guess_ext, self
        )
        ranges = [
            slice(int(lo), pressure_ext.shape[axis] - int(hi))
            for axis, (lo, hi) in enumerate(self.padding)
        ]
        return pressure_ext[tuple(ranges)]

class HelmholtzGMRESSolverTorch2d(_HelmholtzGMRESSolver):
    """2-D Helmholtz GMRES solver.

    ``dispersion=None`` selects continuous dispersion and Fourier spectral
    derivatives.  ``SecondOrderFivePointDispersion2d`` selects the second-order
    staggered stencil.
    """

    _ndim = 2

class HelmholtzGMRESSolverTorch3d(_HelmholtzGMRESSolver):
    """3-D Helmholtz GMRES solver.

    ``dispersion=None`` selects continuous dispersion and Fourier spectral
    derivatives.  ``SecondOrderSevenPointDispersion3d`` selects the second-order
    staggered stencil.
    """

    _ndim = 3
