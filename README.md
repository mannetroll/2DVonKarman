# turbosim

A 2D **decaying turbulence** pseudo-spectral solver with a fancy black PySide6 GUI,
simulating Navier-Stokes flow decaying around a **moving cylindrical rod** (a
von Karman-style obstacle).

## Run

```bash
uv run simulation
```

## Numerics

- Doubly-periodic domain `[0, 2*pi)^2`, vorticity-streamfunction formulation.
- Pure decaying Navier-Stokes (`Visc = 1/Re`) — no forcing, no extra terms.
- Pseudo-spectral nonlinear term with **3/2 zero-padding** de-aliasing.
- **LS-IMEX-RK3** time stepping: viscous term implicit (Crank-Nicolson per
  substage), advection explicit, low storage.
- Moving rod via **volume penalization** (exact relaxation, operator-split).
- All FFTs use `scipy.fft` with `workers=-1` (multithreaded).

## Fast rendering

Each frame the selected field is normalized to a contiguous `uint8` 2D array,
that memory is wrapped directly in a `QImage` (`Format_Indexed8`), a 256-entry
color table is applied, and a `QPixmap` is built from it.

## Controls

| Control | Values |
|---|---|
| `N` | 512, 1024, 2048 spectral nodes (3/2 padding) |
| `Re` | default 10000 (`Visc = 1/Re`) |
| `CFL` | 0.5 – 2.5 (default 1.5) |
| `K0` | 1 – 25, initial field peak wavenumber (default 5) |
| `NR` | rod radius `R` from `2*pi = NR * R` (default 5) |
| `VR` | 0.1 – 10, rod vertical speed in `2*pi/sec` (default 0.1) |
| Field | Vorticity, Energy, U-Velocity, V-Velocity, Stream function |
| Colors | Inferno, Gray, Ocean |
| Frame / n steps | 2, 5, 10, 20, 50 |

`CFL`, `VR`, field, colors and frame skip update live; `N`, `Re`, `K0`, `NR`
take effect on **Restart**.
