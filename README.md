# turbosim

A 2D **decaying turbulence** pseudo-spectral solver with a fancy black PySide6 GUI,
simulating Navier-Stokes flow around a **cylindrical rod** held fixed at the
domain centre, with a uniform **horizontal free-stream** flowing past it — i.e.
the rod's own reference frame as it translates horizontally (a von Karman-style
obstacle / vortex street).

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
- Centred rod via **volume penalization** (exact relaxation, operator-split);
  solved in the rod frame with a uniform horizontal free-stream `(VR, 0)`.
- All state is **float32 / complex64**; spectral operators (`i*k`, the velocity
  projector `i*k/|k|^2`, the viscous symbol `nu*|k|^2`) are precomputed once.
- **CPU / GPU backend** (technique from `fast/turbo_simulator.py`): the same code
  runs on the CPU via `scipy.fft` (multithreaded, `workers=-1`) or on an NVIDIA
  GPU via CuPy + `cupyx.scipy.fft`.

## GPU (NVIDIA, e.g. RTX 3090)

The solver auto-detects CuPy and a CUDA device. On a GPU box install the extra:

```bash
uv sync --extra gpu          # CUDA 12.x; use cupy-cuda11x for CUDA 11
```

Then **Auto** uses the GPU (the `Compute` dropdown also offers explicit `CPU` /
`GPU`; the window title shows the active device). With no CUDA device present the
GPU option is greyed out and everything runs on the CPU.

## Fast rendering

Each frame the selected field is normalized to a contiguous `uint8` 2D array,
that memory is wrapped directly in a `QImage` (`Format_Indexed8`), a 256-entry
color table is applied, and a `QPixmap` is built from it.

## Controls

| Control | Values |
|---|---|
| `N` | 512, 1024, 2048 spectral nodes (3/2 padding) |
| `Re` | default 10000 (`Visc = 1/Re`) |
| `CFL` | 0.5 – 3.5 (default 2.5) |
| `K0` | 1 – 25, initial field peak wavenumber (default 5) |
| `NR` | rod radius `R` from `2*pi = NR * R` (default 10) |
| `VR` | 1 – 100, horizontal free-stream speed in `2*pi/sec` (default 10) |
| Field | Vorticity, Energy, U-Velocity, V-Velocity, Stream function |
| Colors | Inferno, Gray, Ocean |
| Frame / n steps | 2, 5, 10, 20, 50 |
| Compute | Auto, CPU, GPU (GPU needs CuPy + a CUDA device) |

`CFL`, `VR`, field, colors and frame skip update live; `N`, `Re`, `K0`, `NR`
take effect on **Restart**.
