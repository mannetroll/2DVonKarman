"""2D decaying Navier-Stokes pseudo-spectral solver.

Vorticity-streamfunction formulation on a doubly-periodic square domain
``[0, 2*pi)^2``::

    d(omega)/dt + (u . grad) omega = nu * laplacian(omega)
    laplacian(psi) = -omega,   u = d(psi)/dy,   v = -d(psi)/dx

* The nonlinear advection term is computed pseudo-spectrally with **3/2 zero
  padding** de-aliasing.
* Time stepping is a **low-storage IMEX Runge-Kutta 3** scheme: the stiff linear
  viscous term and uniform free-stream advection are integrated implicitly
  (Crank-Nicolson inside each substage), while only fluctuation advection is
  explicit.  Only the previous substage's nonlinear evaluation is stored
  (low storage).
* A **cylindrical rod** (a disk of radius ``R``) is imposed through volume
  penalization, applied as an exact, unconditionally-stable operator-splitting
  velocity relaxation after each full step.  We work in the **rod's reference
  frame**: the rod is held fixed at the domain centre and a uniform
  **horizontal** free-stream ``(vr, 0)`` flows past it, which is exactly
  equivalent to the rod translating horizontally through otherwise still fluid
  (the classic von Karman flow-past-cylinder).  No turbulent forcing is added.

Performance (techniques ported from ``fast/turbo_simulator.py``):

* A **NumPy/CuPy backend abstraction** (``self.xp`` + ``self.fft``) lets the
  identical code run on the CPU (``scipy.fft``, multithreaded) or on an NVIDIA
  GPU such as an **RTX 3090** (``cupyx.scipy.fft``); pass ``backend="gpu"`` /
  ``"cpu"`` / ``"auto"``.
* All state is **float32 / complex64**.
* Spectral operators (``i*k``, the velocity projector ``i*k/|k|^2``, the viscous
  symbol ``nu*|k|^2``) are **precomputed once** so each step is a handful of
  fused array multiplies plus FFTs.
"""

from __future__ import annotations

import numpy as np

from turbosim.backend import gpu_name, select_backend, to_host

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

    ``backend`` selects the compute device: ``"auto"`` (GPU if available, else
    CPU), ``"cpu"`` or ``"gpu"``.
    """

    def __init__(
        self,
        N: int = 512,
        Re: float = 10_000.0,
        cfl: float = 2.5,
        k0: int = 1,
        NR: float = 25.0,
        vr: float = 1.0,
        L: float = TWO_PI,
        seed: int | None = 1234,
        workers: int = -1,
        backend: str = "auto",
    ) -> None:
        if N % 2 != 0:
            raise ValueError("N must be even")

        # --- compute backend (NumPy+scipy.fft or CuPy+cupyx.scipy.fft) ------
        self.backend, self.xp, self.fft = select_backend(backend)
        self.on_gpu = self.backend == "gpu"
        self.rdt = self.xp.float32        # real dtype
        self.cdt = self.xp.complex64      # complex dtype
        print(self._backend_banner(workers))

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
        self.pad_scale = float((self.M / self.N) ** 2)
        self.unpad_scale = float((self.N / self.M) ** 2)

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

    def _backend_banner(self, workers: int) -> str:
        if self.on_gpu:
            name = gpu_name() or "CUDA device"
            return f"turbosim: GPU backend — {name} (CuPy float32, cupyx.scipy.fft)"
        return f"turbosim: CPU backend — NumPy float32, scipy.fft (workers={workers})"

    # ------------------------------------------------------------------ setup
    def _setup_grid(self) -> None:
        N, L = self.N, self.L
        xp = self.xp
        self.nh = N // 2 + 1

        # Integer wavenumbers (L = 2*pi -> k = 0, +/-1, ...). Built on the host
        # then moved to the backend; complex32 doesn't exist, so wavenumbers are
        # float32 and the i*k operators are complex64.
        kx = (np.fft.fftfreq(N, d=1.0 / N) * (TWO_PI / L)).astype(np.float32)
        ky = (np.fft.rfftfreq(N, d=1.0 / N) * (TWO_PI / L)).astype(np.float32)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K2 = (KX * KX + KY * KY).astype(np.float32)
        K2inv = np.zeros_like(K2)
        nz = K2 > 0
        K2inv[nz] = 1.0 / K2[nz]

        self.KX = xp.asarray(KX)
        self.KY = xp.asarray(KY)
        self.K2 = xp.asarray(K2)

        # Precomputed spectral operators (one fused multiply per use):
        #   iKX, iKY            : spectral d/dx, d/dy
        #   opU =  i*KY / |k|^2 : u_hat from omega_hat
        #   opV = -i*KX / |k|^2 : v_hat from omega_hat
        #   nuK2 = nu * |k|^2   : viscous symbol for the IMEX update
        iKX = (1j * KX).astype(np.complex64)
        iKY = (1j * KY).astype(np.complex64)
        self.iKX = xp.asarray(iKX)
        self.iKY = xp.asarray(iKY)
        self.opU = xp.asarray((iKY * K2inv).astype(np.complex64))
        self.opV = xp.asarray((-iKX * K2inv).astype(np.complex64))
        self.K2inv = xp.asarray(K2inv)
        self.nuK2 = xp.asarray((self.nu * K2).astype(np.float32))

        # Physical grid (for the rod mask).
        x = (np.arange(N) * self.dxg).astype(np.float32)
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.X = xp.asarray(X)
        self.Y = xp.asarray(Y)

    def _init_vorticity(self, seed: int | None) -> None:
        """Random initial vorticity with energy peaked near wavenumber ``k0``.

        ``k0 == 0`` is a special case: the flow starts from rest (zero vorticity)
        so the only structure comes from the free-stream sweeping past the rod.
        """
        if self.k0 <= 0:
            self.omega_hat = self.xp.zeros((self.N, self.nh), dtype=self.cdt)
            return

        # Phases are drawn on the host (deterministic NumPy RNG, seed-stable
        # across backends), then the field is moved to the backend and projected.
        rng = np.random.default_rng(seed)
        kx = np.fft.fftfreq(self.N, d=1.0 / self.N) * (TWO_PI / self.L)
        ky = np.fft.rfftfreq(self.N, d=1.0 / self.N) * (TWO_PI / self.L)
        KXh, KYh = np.meshgrid(kx, ky, indexing="ij")
        t = (np.sqrt(KXh * KXh + KYh * KYh) / self.k0) ** 2
        amp = (t * np.exp(-t)).astype(np.float32)        # envelope, peaks at k0
        phase = rng.uniform(0.0, TWO_PI, size=amp.shape).astype(np.float32)
        wh = (amp * np.exp(1j * phase)).astype(np.complex64)
        wh[0, 0] = 0.0

        wh_xp = self.xp.asarray(wh)
        # Project onto a valid real field, then normalize to unit-ish energy.
        w = self._irfft2(wh_xp, (self.N, self.N))
        self.omega_hat = self._rfft2(w).astype(self.cdt)
        self.omega_hat[0, 0] = 0.0
        self._normalize_energy(target=0.5)

    def _normalize_energy(self, target: float) -> None:
        u_hat, v_hat = self._velocities(self.omega_hat)
        u = self._irfft2(u_hat, (self.N, self.N))
        v = self._irfft2(v_hat, (self.N, self.N))
        e = 0.5 * float(self.xp.mean(u * u + v * v))
        if e > 0:
            self.omega_hat *= np.float32(np.sqrt(target / e))

    # ----------------------------------------------------------- FFT wrappers
    def _rfft2(self, a, overwrite: bool = False):
        """Real 2D forward FFT (multithreaded on CPU, cuFFT on GPU)."""
        if self.on_gpu:
            return self.fft.rfft2(a)
        return self.fft.rfft2(a, workers=self.workers, overwrite_x=overwrite)

    def _irfft2(self, a, s):
        """Real 2D inverse FFT onto an ``s``-shaped grid."""
        if self.on_gpu:
            return self.fft.irfft2(a, s=s)
        return self.fft.irfft2(a, s=s, workers=self.workers)

    # ------------------------------------------------------ spectral helpers
    def _velocities(self, omega_hat):
        """(u_hat, v_hat) from omega_hat via the precomputed projectors."""
        return self.opU * omega_hat, self.opV * omega_hat

    def _pad(self, fh):
        """Zero-pad an ``(N, nh)`` rfft2 spectrum to the ``(M, Mh)`` grid."""
        M, Mh, n2, nh = self.M, self.M // 2 + 1, self.N // 2, self.nh
        out = self.xp.zeros((M, Mh), dtype=self.cdt)
        out[:n2, :nh] = fh[:n2, :]
        out[M - n2:, :nh] = fh[n2:, :]
        return out

    def _unpad(self, F):
        """Truncate an ``(M, Mh)`` rfft2 spectrum back to ``(N, nh)``."""
        n2, nh = self.N // 2, self.nh
        out = self.xp.empty((self.N, nh), dtype=self.cdt)
        out[:n2, :] = F[:n2, :nh]
        out[n2:, :] = F[self.M - n2:, :nh]
        return out

    def _to_phys_padded(self, fh):
        """Spectrum -> de-aliased physical field on the padded ``M x M`` grid."""
        P = self._pad(fh) * self.pad_scale
        return self._irfft2(P, (self.M, self.M))

    def _to_spec_trunc(self, phys):
        """Padded physical field -> truncated ``(N, nh)`` spectrum."""
        F = self._rfft2(phys, overwrite=True)
        return self._unpad(F) * self.unpad_scale

    # --------------------------------------------------------------- dynamics
    def _nonlinear(self, omega_hat):
        """Explicit RHS: ``-(u' . grad) omega`` with 3/2 de-aliasing.

        The uniform free-stream contribution ``vr * d(omega)/dx`` is linear and
        is handled implicitly with viscosity in :meth:`step`.
        """
        u_hat, v_hat = self._velocities(omega_hat)
        u = self._to_phys_padded(u_hat)
        v = self._to_phys_padded(v_hat)
        ox = self._to_phys_padded(self.iKX * omega_hat)
        oy = self._to_phys_padded(self.iKY * omega_hat)
        n_hat = -self._to_spec_trunc(u * ox + v * oy)
        n_hat[0, 0] = 0.0
        return n_hat

    def _rod_mask(self):
        """Smoothed indicator of the disk centered at ``(xc, yc)`` (periodic)."""
        dx = (self.X - self.xc + self.L / 2) % self.L - self.L / 2
        dy = (self.Y - self.yc + self.L / 2) % self.L - self.L / 2
        r = self.xp.sqrt(dx * dx + dy * dy)
        return 0.5 * (1.0 - self.xp.tanh((r - self.R) / self.delta))

    def _compute_dt(self, u, v) -> float:
        """Advective CFL from the *total* velocity (``u``, ``v`` already include
        the free-stream)."""
        vel = float(self.xp.abs(u).max() + self.xp.abs(v).max()) + 1e-9
        dt = self.cfl * self.dxg / (np.pi * vel)
        return min(dt, self.dt_cap)

    def step(self) -> None:
        """Advance one full time step."""
        nuK2 = self.nuK2

        # Time step from the current total velocity field (advective CFL).
        u_hat, v_hat = self._velocities(self.omega_hat)
        u = self._irfft2(u_hat, (self.N, self.N)) + self.vr
        v = self._irfft2(v_hat, (self.N, self.N))
        dt = self._compute_dt(u, v)

        # Rod is fixed at the domain centre (rod reference frame).
        chi = self._rod_mask()

        # --- LS-IMEX-RK3 advance of Navier-Stokes ------------------------
        # Linear implicit symbol for d(w)/dt = -S*w + N(w), where S contains
        # viscosity plus uniform free-stream advection in the rod frame.
        linear = (nuK2 + 1j * np.float32(self.vr) * self.KX).astype(self.cdt)
        w = self.omega_hat
        n_prev = None
        for k in range(3):
            n_k = self._nonlinear(w)
            if n_prev is None:
                n_prev = n_k
            num = (1.0 - _ALPHA[k] * dt * linear) * w \
                + dt * (_GAMMA[k] * n_k + _ZETA[k] * n_prev)
            w = num / (1.0 + _BETA[k] * dt * linear)
            n_prev = n_k
        self.omega_hat = w

        # --- Volume penalization (exact relaxation, operator split) -------
        # In the rod frame the rod is stationary, so the *total* velocity
        # (vortical part + free-stream) relaxes to zero inside the disk:
        #   u_tot <- u_tot * exp(-pen * chi),  then subtract the free-stream
        #   back out to recover the vortical part the spectral code carries.
        u_hat, v_hat = self._velocities(self.omega_hat)
        u = self._irfft2(u_hat, (self.N, self.N)) + self.vr
        v = self._irfft2(v_hat, (self.N, self.N))
        f = self.xp.exp(-self.pen_strength * chi)
        u = u * f - self.vr
        v = v * f
        u_hat = self._rfft2(u, overwrite=True)
        v_hat = self._rfft2(v, overwrite=True)
        # Recover vorticity (curl) -- discards the penalization divergence.
        self.omega_hat = self.iKX * v_hat - self.iKY * u_hat
        self.omega_hat[0, 0] = 0.0

        self.time += dt
        self.step_count += 1
        self.last_dt = dt

    # ----------------------------------------------------------- diagnostics
    def get_field(self, name: str) -> np.ndarray:
        """Return the requested physical field as a host ``(N, N)`` float array."""
        s = (self.N, self.N)
        if name == "Vorticity":
            return to_host(self._irfft2(self.omega_hat, s))
        if name == "Stream function":
            return to_host(self._irfft2(self.K2inv * self.omega_hat, s))
        # U-Velocity and Energy show the *total* flow (incl. the free-stream);
        # V-Velocity has no free-stream component.
        u_hat, v_hat = self._velocities(self.omega_hat)
        if name == "U-Velocity":
            return to_host(self._irfft2(u_hat, s) + self.vr)
        if name == "V-Velocity":
            return to_host(self._irfft2(v_hat, s))
        if name == "Energy":
            u = self._irfft2(u_hat, s) + self.vr
            v = self._irfft2(v_hat, s)
            return to_host(0.5 * (u * u + v * v))
        raise ValueError(f"unknown field {name!r}")

    def stats(self) -> dict[str, float]:
        xp = self.xp
        s = (self.N, self.N)
        u_hat, v_hat = self._velocities(self.omega_hat)
        u = self._irfft2(u_hat, s)
        v = self._irfft2(v_hat, s)
        omega = self._irfft2(self.omega_hat, s)
        return {
            "time": self.time,
            "step": float(self.step_count),
            "dt": self.last_dt,
            "energy": 0.5 * float(xp.mean(u * u + v * v)),
            "enstrophy": 0.5 * float(xp.mean(omega * omega)),
            "max_omega": float(xp.abs(omega).max()),
            "xc": self.xc,
            "yc": self.yc,
            "R": self.R,
            "L": self.L,
            "backend": self.backend,
        }
