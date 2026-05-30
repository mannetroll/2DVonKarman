"""2D decaying Navier-Stokes pseudo-spectral solver.

Vorticity-streamfunction formulation on a doubly-periodic square domain
``[0, 2*pi)^2``::

    d(omega)/dt + (u . grad) omega = nu * laplacian(omega)
    laplacian(psi) = -omega,   u = d(psi)/dy,   v = -d(psi)/dx

* The nonlinear advection term is computed pseudo-spectrally with **3/2 zero
  padding** de-aliasing.
* Time stepping is a **low-storage IMEX Runge-Kutta 3** scheme: the stiff linear
  viscous term is integrated implicitly (Crank-Nicolson inside each substage),
  the nonlinear term explicitly.  Only the previous substage's nonlinear
  evaluation is stored (low storage).
* A **cylindrical rod** (a disk of radius ``R``) is imposed through volume
  penalization, applied as an exact, unconditionally-stable operator-splitting
  velocity relaxation after each full step.  We work in the **rod's reference
  frame**: the rod is held fixed at the domain centre and a uniform
  **horizontal** free-stream ``(vr, 0)`` flows past it, which is exactly
  equivalent to the rod translating horizontally through otherwise still fluid
  (the classic von Karman flow-past-cylinder).  No turbulent forcing is added.
* All FFTs use ``scipy.fft`` with ``workers=-1`` (multithreaded).
"""

from __future__ import annotations

import numpy as np
from scipy import fft

TWO_PI = 2.0 * np.pi

# Low-storage IMEX-RK3 (Spalart / Kim-Moin-Moser RK3-CN) coefficients.
_ALPHA = (4.0 / 15.0, 1.0 / 15.0, 1.0 / 6.0)   # implicit, old level
_BETA = (4.0 / 15.0, 1.0 / 15.0, 1.0 / 6.0)    # implicit, new level
_GAMMA = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)   # explicit, this substage
_ZETA = (0.0, -17.0 / 60.0, -5.0 / 12.0)       # explicit, previous substage

FIELD_NAMES = ("Vorticity", "Energy", "U-Velocity", "V-Velocity", "Stream function")


class Solver:
    """A single 2D turbulence simulation instance.

    Parameters mirror the GUI controls.  ``cfl`` and ``vr`` may be changed live
    while the simulation runs; the others define the grid / initial state and
    require a fresh :class:`Solver`.
    """

    def __init__(
        self,
        N: int = 512,
        Re: float = 10_000.0,
        cfl: float = 2.5,
        k0: int = 5,
        NR: float = 10.0,
        vr: float = 10.0,
        L: float = TWO_PI,
        seed: int | None = 1234,
        workers: int = -1,
    ) -> None:
        if N % 2 != 0:
            raise ValueError("N must be even")
        self.N = int(N)
        self.Re = float(Re)
        self.nu = 1.0 / float(Re)
        self.cfl = float(cfl)
        self.k0 = int(k0)
        self.NR = float(NR)
        self.vr = float(vr)         # free-stream speed (horizontal, +x) in (2*pi/sec)
        self.L = float(L)
        self.workers = int(workers)

        self.R = self.L / self.NR           # rod radius: 2*pi = NR * R
        self.dxg = self.L / self.N          # physical grid spacing
        self.delta = 2.0 * self.dxg         # mask smoothing width (a few cells)
        self.pen_strength = 30.0            # penalization: ~hard solid inside rod
        self.dt_cap = 0.05

        # Padded ("3/2 rule") grid size used for de-aliased products.
        self.M = (3 * self.N) // 2

        self.time = 0.0
        self.step_count = 0
        self.last_dt = 0.0

        # Rod is held fixed in the rod reference frame; the free-stream (vr, 0)
        # flows horizontally past it.  Placed 1/4 in from the upstream (left)
        # edge so ~3/4 of the periodic domain downstream is free for the wake.
        self.xc = self.L / 4.0
        self.yc = self.L / 2.0

        self._setup_grid()
        self._init_vorticity(seed)

    # ------------------------------------------------------------------ setup
    def _setup_grid(self) -> None:
        N, L = self.N, self.L
        self.nh = N // 2 + 1

        # Integer wavenumbers (L = 2*pi -> k = 0, +/-1, ...).
        kx = np.fft.fftfreq(N, d=1.0 / N) * (TWO_PI / L)
        ky = np.fft.rfftfreq(N, d=1.0 / N) * (TWO_PI / L)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2inv = np.zeros_like(self.K2)
        nz = self.K2 > 0
        self.K2inv[nz] = 1.0 / self.K2[nz]

        # Physical grid (for the moving-rod mask).
        x = np.arange(N) * self.dxg
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")

    def _init_vorticity(self, seed: int | None) -> None:
        """Random initial vorticity with energy peaked near wavenumber ``k0``."""
        rng = np.random.default_rng(seed)
        kmag = np.sqrt(self.K2)
        t = (kmag / self.k0) ** 2
        amp = t * np.exp(-t)                       # spectral envelope, peaks at k0
        phase = rng.uniform(0.0, TWO_PI, size=self.KX.shape)
        wh = amp * np.exp(1j * phase)
        wh[0, 0] = 0.0
        # Project onto a valid real field, then normalize to unit-ish energy.
        w = fft.irfft2(wh, s=(self.N, self.N), workers=self.workers)
        self.omega_hat = fft.rfft2(w, workers=self.workers)
        self.omega_hat[0, 0] = 0.0
        self._normalize_energy(target=0.5)

    def _normalize_energy(self, target: float) -> None:
        u_hat, v_hat = self._velocities(self.omega_hat)
        u = fft.irfft2(u_hat, s=(self.N, self.N), workers=self.workers)
        v = fft.irfft2(v_hat, s=(self.N, self.N), workers=self.workers)
        e = 0.5 * float(np.mean(u * u + v * v))
        if e > 0:
            self.omega_hat *= np.sqrt(target / e)

    # ------------------------------------------------------ spectral helpers
    def _velocities(self, omega_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi_hat = self.K2inv * omega_hat
        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat
        return u_hat, v_hat

    def _pad(self, fh: np.ndarray) -> np.ndarray:
        """Zero-pad an ``(N, nh)`` rfft2 spectrum to the ``(M, Mh)`` grid."""
        M, Mh, n2, nh = self.M, self.M // 2 + 1, self.N // 2, self.nh
        out = np.zeros((M, Mh), dtype=complex)
        out[:n2, :nh] = fh[:n2, :]
        out[M - n2:, :nh] = fh[n2:, :]
        return out

    def _unpad(self, F: np.ndarray) -> np.ndarray:
        """Truncate an ``(M, Mh)`` rfft2 spectrum back to ``(N, nh)``."""
        n2, nh = self.N // 2, self.nh
        out = np.empty((self.N, nh), dtype=complex)
        out[:n2, :] = F[:n2, :nh]
        out[n2:, :] = F[self.M - n2:, :nh]
        return out

    def _to_phys_padded(self, fh: np.ndarray) -> np.ndarray:
        """Spectrum -> de-aliased physical field on the padded ``M x M`` grid."""
        P = self._pad(fh) * (self.M / self.N) ** 2
        return fft.irfft2(P, s=(self.M, self.M), workers=self.workers)

    def _to_spec_trunc(self, phys: np.ndarray) -> np.ndarray:
        """Padded physical field -> truncated ``(N, nh)`` spectrum."""
        F = fft.rfft2(phys, workers=self.workers)
        return self._unpad(F) * (self.N / self.M) ** 2

    # --------------------------------------------------------------- dynamics
    def _nonlinear(self, omega_hat: np.ndarray) -> np.ndarray:
        """Explicit RHS: ``-(u . grad) omega`` with 3/2 de-aliasing.

        The advecting velocity is the *total* velocity, i.e. the vortical part
        plus the uniform horizontal free-stream ``(vr, 0)``.
        """
        u_hat, v_hat = self._velocities(omega_hat)
        u = self._to_phys_padded(u_hat) + self.vr
        v = self._to_phys_padded(v_hat)
        ox = self._to_phys_padded(1j * self.KX * omega_hat)
        oy = self._to_phys_padded(1j * self.KY * omega_hat)
        n_hat = -self._to_spec_trunc(u * ox + v * oy)
        n_hat[0, 0] = 0.0
        return n_hat

    def _rod_mask(self) -> np.ndarray:
        """Smoothed indicator of the disk centered at ``(xc, yc)`` (periodic)."""
        dx = (self.X - self.xc + self.L / 2) % self.L - self.L / 2
        dy = (self.Y - self.yc + self.L / 2) % self.L - self.L / 2
        r = np.sqrt(dx * dx + dy * dy)
        return 0.5 * (1.0 - np.tanh((r - self.R) / self.delta))

    def _compute_dt(self, u: np.ndarray, v: np.ndarray) -> float:
        """Advective CFL from the *total* velocity (``u``, ``v`` already include
        the free-stream)."""
        vel = float(np.abs(u).max() + np.abs(v).max()) + 1e-9
        dt = self.cfl * self.dxg / (np.pi * vel)
        return min(dt, self.dt_cap)

    def step(self) -> None:
        """Advance one full time step."""
        nu, K2 = self.nu, self.K2

        # Time step from the current total velocity field (advective CFL).
        u_hat, v_hat = self._velocities(self.omega_hat)
        u = fft.irfft2(u_hat, s=(self.N, self.N), workers=self.workers) + self.vr
        v = fft.irfft2(v_hat, s=(self.N, self.N), workers=self.workers)
        dt = self._compute_dt(u, v)

        # Rod is fixed at the domain centre (rod reference frame).
        chi = self._rod_mask()

        # --- LS-IMEX-RK3 advance of the pure decaying Navier-Stokes -------
        w = self.omega_hat
        n_prev = None
        for k in range(3):
            n_k = self._nonlinear(w)
            if n_prev is None:
                n_prev = n_k
            num = (1.0 - _ALPHA[k] * dt * nu * K2) * w \
                + dt * (_GAMMA[k] * n_k + _ZETA[k] * n_prev)
            w = num / (1.0 + _BETA[k] * dt * nu * K2)
            n_prev = n_k
        self.omega_hat = w

        # --- Volume penalization (exact relaxation, operator split) -------
        # In the rod frame the rod is stationary, so the *total* velocity
        # (vortical part + free-stream) relaxes to zero inside the disk:
        #   u_tot <- u_tot * exp(-pen * chi),  then subtract the free-stream
        #   back out to recover the vortical part the spectral code carries.
        u_hat, v_hat = self._velocities(self.omega_hat)
        u = fft.irfft2(u_hat, s=(self.N, self.N), workers=self.workers) + self.vr
        v = fft.irfft2(v_hat, s=(self.N, self.N), workers=self.workers)
        f = np.exp(-self.pen_strength * chi)
        u = u * f - self.vr
        v *= f
        u_hat = fft.rfft2(u, workers=self.workers)
        v_hat = fft.rfft2(v, workers=self.workers)
        # Recover vorticity (curl) -- discards the penalization divergence.
        self.omega_hat = 1j * self.KX * v_hat - 1j * self.KY * u_hat
        self.omega_hat[0, 0] = 0.0

        self.time += dt
        self.step_count += 1
        self.last_dt = dt

    # ----------------------------------------------------------- diagnostics
    def get_field(self, name: str) -> np.ndarray:
        """Return the requested physical field as an ``(N, N)`` float array."""
        if name == "Vorticity":
            return fft.irfft2(self.omega_hat, s=(self.N, self.N), workers=self.workers)
        if name == "Stream function":
            return fft.irfft2(self.K2inv * self.omega_hat, s=(self.N, self.N),
                              workers=self.workers)
        # U-Velocity and Energy show the *total* flow (incl. the free-stream);
        # V-Velocity has no free-stream component.
        u_hat, v_hat = self._velocities(self.omega_hat)
        if name == "U-Velocity":
            return fft.irfft2(u_hat, s=(self.N, self.N), workers=self.workers) + self.vr
        if name == "V-Velocity":
            return fft.irfft2(v_hat, s=(self.N, self.N), workers=self.workers)
        if name == "Energy":
            u = fft.irfft2(u_hat, s=(self.N, self.N), workers=self.workers) + self.vr
            v = fft.irfft2(v_hat, s=(self.N, self.N), workers=self.workers)
            return 0.5 * (u * u + v * v)
        raise ValueError(f"unknown field {name!r}")

    def stats(self) -> dict[str, float]:
        u_hat, v_hat = self._velocities(self.omega_hat)
        u = fft.irfft2(u_hat, s=(self.N, self.N), workers=self.workers)
        v = fft.irfft2(v_hat, s=(self.N, self.N), workers=self.workers)
        omega = fft.irfft2(self.omega_hat, s=(self.N, self.N), workers=self.workers)
        return {
            "time": self.time,
            "step": float(self.step_count),
            "dt": self.last_dt,
            "energy": 0.5 * float(np.mean(u * u + v * v)),
            "enstrophy": 0.5 * float(np.mean(omega * omega)),
            "max_omega": float(np.abs(omega).max()),
            "xc": self.xc,
            "yc": self.yc,
            "R": self.R,
            "L": self.L,
        }
