"""Pure-ASCII renderer for the turbosim solver -- no Qt, no PySide6.

Runs the N=512 :class:`~turbosim.solver.Solver` and draws the vorticity field as
a live 128x32 ASCII-art von Karman vortex street in the terminal (ANSI cursor
moves, per-row diffing, 256-color codes), with a small diagnostics header.

    uv run simulation_ascii

Geometry (see :data:`COLS` / :data:`ROWS`): the solver runs on an ``N x N`` grid
(default ``N=512``).  We render a 128 x 32 view of the vorticity:

* **width**  -- the *full* domain in x (streamwise), block-averaged ``N/128`` px
  -> 128 columns
* **height** -- a ``32 * (N/128)``-px band centred on the rod (``y = L/2``),
  block-averaged the same factor -> 32 rows (square pixel blocks, wake-focused)

At ``N=512`` that is 4-px square blocks over the full 512-px width and a 128-px
tall band around the rod, so the rod sits a quarter in from the left (column
~32, since ``xc = L/4``) just as in the GUI.

Note the solver lays arrays out ``[x, y]`` (axis 0 = streamwise x); we transpose
to ``[y, x]`` so image rows are y and columns are x.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque

import numpy as np

from turbosim import colormaps
from turbosim.backend import to_host
from turbosim.solver import Solver

# Brightness ramp -- the classic ASCII density ramp; blank background lets the
# free-stream read as empty space so only vortices light up.
PALETTE = " .:-=+*#%@"

# ASCII canvas size.
COLS = 128
ROWS = 32

# Terminal-row (1-indexed) where the field block starts; everything above is the
# diagnostics header (4 lines + blank + caption + separator = 7 lines).
_FIELD_ROW0 = 8


# --------------------------------------------------------------------- color
def _rgb_to_xterm256(r: int, g: int, b: int) -> int:
    """Map an 8-bit RGB triple to the nearest xterm-256 palette index.

    Uses the 6x6x6 color cube (indices 16..231); near-grey values snap to the
    232..255 grey ramp, which renders smoother gradients in most terminals.
    """
    if abs(r - g) < 12 and abs(g - b) < 12:
        level = int(round((max(r, g, b) - 8) / 230 * 23))
        return 232 + min(23, max(0, level))
    to6 = lambda v: int(round(v / 255 * 5))
    return 16 + 36 * to6(r) + 6 * to6(g) + to6(b)


def _build_color_lut(cmap_name: str) -> list[int]:
    """256-entry lookup: brightness level (0..255) -> xterm-256 color index."""
    table = colormaps.colormap(cmap_name)  # (256, 3) uint8, Qt-free
    return [_rgb_to_xterm256(int(r), int(g), int(b)) for r, g, b in table]


# --------------------------------------------------------------------- render
class AsciiRenderer:
    """Turns solver state into ASCII frames and diffs them onto the terminal."""

    def __init__(self, solver: Solver, cmap: str = "Inferno", color: bool = True,
                 diff: bool = True) -> None:
        self.solver = solver
        self.color = color
        self.diff = diff
        self.lut = _build_color_lut(cmap)
        self._prev_rows: list[str | None] = [None] * ROWS
        self._umax_hist: deque[str] = deque(maxlen=14)
        self._build_geometry()

    def _build_geometry(self) -> None:
        """Precompute the downsample block size, y-band and rod overlay mask."""
        s = self.solver
        N = s.N
        self.block = max(1, N // COLS)            # px per character cell (4 at N=512)
        band = ROWS * self.block                  # tall band kept around the rod
        cy = int(round(s.yc / s.L * N))           # rod centre row (y = L/2)
        self.y0 = max(0, min(N - band, cy - band // 2))
        self.band = band

        # Rod overlay: cells whose centre falls inside the rod radius -> 'o'.
        R_px = s.R / s.L * N
        cx_px = s.xc / s.L * N
        col_px = (np.arange(COLS) + 0.5) * self.block
        row_px = self.y0 + (np.arange(ROWS) + 0.5) * self.block
        dx = col_px[np.newaxis, :] - cx_px
        dy = row_px[:, np.newaxis] - cy
        self.rod = (dx * dx + dy * dy) < (R_px * R_px)

    # -- field -> chars/colors --
    def _vorticity(self) -> np.ndarray:
        """Vorticity as a host ``[y, x]`` array (solver stores ``[x, y]``)."""
        return self.solver.get_field("Vorticity").T

    def _downsample(self, W: np.ndarray) -> np.ndarray:
        """Block-average the full-width / banded vorticity to (ROWS, COLS)."""
        b = self.block
        sub = W[self.y0:self.y0 + self.band, :COLS * b]
        return sub.reshape(ROWS, b, COLS, b).mean(axis=(1, 3))

    def _cells(self, small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map a (ROWS, COLS) field to character indices and color indices.

        Brightness is the magnitude normalized by the 98th percentile of |field|,
        so a quiescent free-stream reads as blank space and only vortices light up.
        """
        amax = float(np.percentile(np.abs(small), 98.0))
        if amax < 1e-6:
            amax = 1e-6
        mag = np.clip(np.abs(small) / amax, 0.0, 1.0)
        char_idx = np.rint(mag * (len(PALETTE) - 1)).astype(np.intp)
        color_idx = np.rint(mag * 255.0).astype(np.intp)
        return char_idx, color_idx

    # -- diagnostics header --
    def _diagnostics(self, W: np.ndarray) -> list[str]:
        s = self.solver
        N = s.N

        # Enstrophy fraction on the kx=0 / ky=0 spectral axes (a cross artifact of
        # the periodic box; high values mean the wake is railing).  omega_hat is
        # (N, nh) with axis 0 = kx (full), axis 1 = ky (rfft half).
        ens = np.abs(to_host(s.omega_hat)) ** 2
        total = float(ens.sum()) + 1e-30
        axes = float(ens[0, :].sum() + ens[:, 0].sum() - ens[0, 0])
        ens_pct = 100.0 * axes / total

        # Vorticity rms upstream vs. in the downstream wake, rod interior excluded.
        chi = to_host(s._rod_mask()).T          # [y, x]
        out = chi < 0.5
        col = np.arange(N)[np.newaxis, :]        # x index (columns of [y, x])
        cx = int(round(s.xc / s.L * N))
        up = out & (col < cx)
        down = out & (col >= cx)
        rms = lambda m: float(np.sqrt(np.mean(W[m] ** 2))) if m.any() else 0.0
        rms_up, rms_down = rms(up), rms(down)
        ratio = rms_down / rms_up if rms_up > 0 else 0.0

        u = s.get_field("U-Velocity")
        v = s.get_field("V-Velocity")
        umax = float(np.sqrt(np.max(u * u + v * v)))
        self._umax_hist.append(f"{umax:.1f}")
        wmax = float(np.max(np.abs(W)))

        return [
            f"enstrophy on spectral axes = {ens_pct:.2f}%  (cross artifact if high)",
            f"vorticity rms  upstream={rms_up:.3f}  "
            f"downstream(wake)={rms_down:.3f}  ratio={ratio:.2f}",
            f"umax over run: {list(self._umax_hist)}",
            f"live: step={s.step_count}  t={s.time:.3f}  "
            f"dt={s.last_dt:.2e}  wmax={wmax:.2f}  [{s.backend}]",
        ]

    # -- frame assembly --
    def frame(self) -> str:
        """Build the ANSI string for the current solver state (header + field)."""
        W = self._vorticity()
        header = self._diagnostics(W)
        char_idx, color_idx = self._cells(self._downsample(W))

        buf: list[str] = []

        # Header: always redrawn (it changes every frame). \033[K clears leftovers.
        for i, line in enumerate(header):
            buf.append(f"\033[{i + 1};1H{line}\033[K")
        buf.append(f"\033[6;1HVORTICITY  (flow ->,  o = rod)\033[K")
        buf.append(f"\033[7;1H{'-' * COLS}")

        # Field rows, optionally diffed (skip rows unchanged since last frame).
        for r in range(ROWS):
            row = self._render_row(char_idx[r], color_idx[r], self.rod[r])
            line = f"\033[{_FIELD_ROW0 + r};1H{row}"
            if self.diff and self._prev_rows[r] == line:
                continue
            self._prev_rows[r] = line
            buf.append(line)

        return "".join(buf)

    def _render_row(self, chars: np.ndarray, colors: np.ndarray,
                    rod: np.ndarray) -> str:
        out: list[str] = []
        last = -1
        for c in range(COLS):
            if rod[c]:
                ch, col = "o", 231  # bright white rod
            else:
                ch = PALETTE[chars[c]]
                col = self.lut[colors[c]]
            if self.color and ch != " " and col != last:
                out.append(f"\033[38;5;{col}m")
                last = col
            out.append(ch)
        if self.color:
            out.append("\033[0m")
            last = -1
        return "".join(out)


# --------------------------------------------------------------------- driver
def _run(renderer: AsciiRenderer, nsteps: int, frames: int) -> None:
    out, flush = sys.stdout.write, sys.stdout.flush
    out("\033[2J\033[?25l")  # clear screen, hide cursor
    flush()
    solver = renderer.solver
    start = time.perf_counter()
    drawn = 0
    try:
        while frames == 0 or drawn < frames:
            for _ in range(nsteps):
                solver.step()
            out(renderer.frame())
            flush()
            drawn += 1
    finally:
        out("\033[?25h\033[0m\n")  # show cursor, reset
        flush()
        elapsed = time.perf_counter() - start
        if drawn and elapsed > 0:
            sys.stderr.write(
                f"{COLS}x{ROWS}  {drawn} frames  "
                f"[{nsteps} steps/frame]\nelapsed: {elapsed:.3f}s   "
                f"~{drawn / elapsed:.1f} FPS\n"
            )


def main() -> int:
    p = argparse.ArgumentParser(description="ASCII von Karman vortex street.")
    p.add_argument("--frames", type=int, default=0, help="0 = run until Ctrl-C")
    p.add_argument("--nsteps", type=int, default=2, help="solver steps per frame")
    p.add_argument("--re", type=float, default=200.0, help="Reynolds number")
    p.add_argument("--vr", type=float, default=2.0, help="free-stream speed")
    p.add_argument("--nr", type=float, default=25.0, help="rod radius (2*pi/NR)")
    p.add_argument("--cfl", type=float, default=2.5)
    p.add_argument("--k0", type=int, default=1, help="initial peak wavenumber")
    p.add_argument("--n", type=int, default=512, help="spectral grid size")
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--cmap", default="Inferno", choices=list(colormaps.COLORMAP_NAMES))
    p.add_argument("--mono", action="store_true", help="no color codes")
    p.add_argument("--no-diff", action="store_true", help="redraw every row")
    args = p.parse_args()

    solver = Solver(N=args.n, Re=args.re, cfl=args.cfl, k0=args.k0,
                    NR=args.nr, vr=args.vr, backend=args.backend)
    renderer = AsciiRenderer(solver, cmap=args.cmap, color=not args.mono,
                             diff=not args.no_diff)

    # Restore the cursor on Ctrl-C even though we also do it in the finally block.
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    _run(renderer, nsteps=args.nsteps, frames=args.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
