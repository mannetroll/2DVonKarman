# Von Karman Vortex Street

A 2D **decaying turbulence** pseudo-spectral solver, 
simulating Navier-Stokes flow around a **cylindrical rod** held fixed at the
domain centre, with a uniform **horizontal free-stream** flowing past it — i.e.
the rod's own reference frame as it translates horizontally (a von Karman-style
obstacle / vortex street).

## Screenshots

The dark PySide6 GUI:

![PySide6 GUI](https://raw.githubusercontent.com/mannetroll/2DVonKarman/v0.1.0/simulation_gui.png)

The pure-ASCII terminal front-end:

![ASCII vortex street](https://raw.githubusercontent.com/mannetroll/2DVonKarman/v0.1.0/simulation_ascii.png)

## Install

```bash
pip install mannetroll-vonkarman      # from PyPI
```

This installs two console scripts, `simulation` and `simulation_ascii`.

## Run

```bash
uv run simulation          # the dark PySide6 GUI
uv run simulation_ascii    # pure-ASCII vortex street in the terminal (no Qt)

# or, once installed:
simulation                 # the dark PySide6 GUI
simulation_ascii           # pure-ASCII vortex street in the terminal (no Qt)
```

Run straight from PyPI without installing anything:

```bash
uv run --python 3.13 --with "mannetroll-vonkarman==0.1.0" simulation
uv run --python 3.13 --with "mannetroll-vonkarman==0.1.0" simulation_ascii
```

Run the GPU build

```bash
uv run --python 3.13 --with "mannetroll-vonkarman[cuda]==0.1.0" simulation
uv run --python 3.13 --with "mannetroll-vonkarman[cuda]==0.1.0" simulation_ascii
```

The ASCII front-end runs the same `N=512` solver and draws the vorticity field
as a live 128×32 character grid (`" .:-=+*#%@"` brightness ramp, 256-color, the
rod marked `o`), with a small diagnostics header. Flags: `--mono`, `--no-diff`,
`--cmap {Inferno,Ocean,Gray}`, `--backend {auto,cpu,gpu}`, `--nsteps`,
`--frames`, and the usual `--re/--vr/--nr/--cfl/--k0/--n`.

## Numerics

- Doubly-periodic domain `[0, 2*pi)^2`, vorticity-streamfunction formulation.
- Pure decaying Navier-Stokes (`Visc = 1/Re`) — no forcing, no extra terms.
- Pseudo-spectral nonlinear term with **3/2 zero-padding** de-aliasing.
- **LS-IMEX-RK3** time stepping: viscous term and uniform free-stream advection
  implicit (Crank-Nicolson per substage), fluctuation advection explicit,
  low storage.
- Centred rod via **volume penalization** (exact relaxation, operator-split);
  solved in the rod frame with a uniform horizontal free-stream `(VR, 0)`.
- All state is **float32 / complex64**; spectral operators (`i*k`, the velocity
  projector `i*k/|k|^2`, the viscous symbol `nu*|k|^2`) are precomputed once.
- **CPU / GPU backend** (technique from `cuda/turbo_simulator.py`): the same code
  runs on the CPU via `scipy.fft` (multithreaded, `workers=-1`) or on an NVIDIA
  GPU via CuPy + `cupyx.scipy.fft`.

## GPU (NVIDIA, e.g. RTX 3090)

The solver auto-detects CuPy and a CUDA device. On a GPU box install the extra:

```bash
uv sync --extra cuda         # CUDA 13.x (cupy-cuda13x)
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
| `K0` | 1 – 25, initial field peak wavenumber (default 1) |
| `NR` | 2 – 100, rod radius `R` from `2*pi = NR * R` (default 25) |
| `VR` | 1 – 100, horizontal free-stream speed in `2*pi/sec` (default 2) |
| Field | Vorticity, Energy, U-Velocity, V-Velocity, Stream function |
| Colors | Inferno, Gray, Ocean |
| Frame / n steps | 2, 5, 10, 20, 50 |
| Compute | Auto, CPU, GPU (GPU needs CuPy + a CUDA device) |

`CFL`, `VR`, field, colors and frame skip update live; `N`, `Re`, `K0`, `NR`
take effect on **Restart**.
