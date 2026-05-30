"""PySide6 GUI for the turbosim 2D turbulence solver.

The simulation runs in a :class:`SimWorker` background thread and emits ready
``uint8`` frames; the GUI thread wraps each frame's memory directly in a
``QImage``, applies a 256-entry color table and builds a ``QPixmap`` (fast
rendering).  Numerical statistics use fixed-width, fixed-format labels so the
digits never reflow or flicker.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from turbosim.backend import gpu_available
from turbosim.colormaps import COLORMAP_NAMES, colormap
from turbosim.render import display_limits, to_uint8
from turbosim.solver import FIELD_NAMES, Solver

N_OPTIONS = (512, 1024, 2048)
FRAMESKIP_OPTIONS = (2, 5, 10, 20, 50)


# --------------------------------------------------------------------- worker
class SimWorker(QThread):
    """Runs the solver and emits a uint8 frame + statistics every N steps."""

    frameReady = Signal(object, dict)

    def __init__(self, solver: Solver) -> None:
        super().__init__()
        self.solver = solver
        self.frameskip = 2
        self.field_name = "Vorticity"
        self._running = True
        self._paused = False
        self._fps = 0.0
        self._last_emit = time.perf_counter()

    def run(self) -> None:
        s = self.solver
        while self._running:
            if self._paused:
                self.msleep(40)
                continue
            for _ in range(self.frameskip):
                if not self._running:
                    return
                s.step()
            field = s.get_field(self.field_name)
            vmin, vmax = display_limits(field, self.field_name)
            u8 = to_uint8(field, vmin, vmax)

            now = time.perf_counter()
            dt_wall = now - self._last_emit
            self._last_emit = now
            if dt_wall > 0:
                steps_per_sec = self.frameskip / dt_wall
                self._fps = 0.9 * self._fps + 0.1 * steps_per_sec

            stats = s.stats()
            stats["fps"] = self._fps
            stats["field"] = self.field_name
            self.frameReady.emit(u8, stats)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def stop(self) -> None:
        self._running = False
        self.wait()


# ----------------------------------------------------------------- main window
DARK_QSS = """
QWidget { background: #0b0b0d; color: #d8d8de;
          font-family: 'Menlo','Consolas','DejaVu Sans Mono',monospace; font-size: 12px; }
QGroupBox { border: 1px solid #26262e; border-radius: 8px; margin-top: 16px;
            padding: 10px 8px 8px 8px; background: #101015; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px;
                   color: #8a8cff; font-weight: bold; }
QPushButton { background: #1a1a22; border: 1px solid #34343e; border-radius: 6px;
              padding: 7px 14px; }
QPushButton:hover { background: #262630; border-color: #7a7cff; }
QPushButton:pressed { background: #14141a; }
QComboBox, QSpinBox { background: #15151c; border: 1px solid #34343e;
                      border-radius: 5px; padding: 4px 6px; }
QComboBox QAbstractItemView { background: #15151c; selection-background-color: #3a3a8a; }
QSlider::groove:horizontal { height: 4px; background: #2a2a32; border-radius: 2px; }
QSlider::handle:horizontal { background: #7a7cff; width: 14px; height: 14px;
                             margin: -6px 0; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #4a4ca0; border-radius: 2px; }
QLabel#statval { color: #56d364; }
QLabel#statname { color: #8a8a96; }
QLabel#hint { color: #6a6a76; }
"""


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("turbosim - 2D decaying turbulence")
        self.setStyleSheet(DARK_QSS)
        self.resize(1180, 820)

        self.worker: SimWorker | None = None
        self._table = self._qt_table(colormap("Ocean"))
        self._last_stats: dict = {}

        self._build_ui()
        self.build_and_start()

    # ----- color table -----
    @staticmethod
    def _qt_table(rgb: np.ndarray) -> list[int]:
        return [(0xFF << 24) | (int(r) << 16) | (int(g) << 8) | int(b)
                for r, g, b in rgb]

    # ----- UI -----
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Image canvas.
        self.canvas = QLabel("starting...")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumSize(560, 560)
        self.canvas.setFrameShape(QFrame.NoFrame)
        self.canvas.setStyleSheet("background: #000000; border: 1px solid #26262e;"
                                  " border-radius: 8px;")
        root.addWidget(self.canvas, stretch=1)

        # Side panel.
        panel = QVBoxLayout()
        panel.setSpacing(10)
        panel.addWidget(self._physics_group())
        panel.addWidget(self._rod_group())
        panel.addWidget(self._display_group())
        panel.addWidget(self._buttons())
        panel.addWidget(self._stats_group())
        panel.addStretch(1)
        side = QWidget()
        side.setLayout(panel)
        side.setFixedWidth(320)
        root.addWidget(side)

    def _physics_group(self) -> QGroupBox:
        g = QGroupBox("Grid && Physics")
        lay = QGridLayout(g)
        lay.setVerticalSpacing(8)

        self.cb_N = QComboBox()
        self.cb_N.addItems([str(n) for n in N_OPTIONS])
        self.cb_N.setCurrentText("512")

        self.sp_Re = QSpinBox()
        self.sp_Re.setRange(1, 10_000_000)
        self.sp_Re.setSingleStep(1000)
        self.sp_Re.setValue(10_000)

        self.sl_cfl, self.lb_cfl = self._slider(50, 350, 250, self._on_cfl)
        self.sl_k0, self.lb_k0 = self._slider(1, 25, 15, self._on_k0)

        # Compute backend: Auto picks the GPU (CuPy) if one is present, else CPU.
        self.cb_backend = QComboBox()
        self.cb_backend.addItems(["Auto", "CPU", "GPU"])
        if not gpu_available():
            self.cb_backend.model().item(2).setEnabled(False)  # grey out GPU

        lay.addWidget(QLabel("N (nodes)"), 0, 0)
        lay.addWidget(self.cb_N, 0, 1, 1, 2)
        lay.addWidget(QLabel("Re"), 1, 0)
        lay.addWidget(self.sp_Re, 1, 1, 1, 2)
        lay.addWidget(QLabel("CFL"), 2, 0)
        lay.addWidget(self.sl_cfl, 2, 1)
        lay.addWidget(self.lb_cfl, 2, 2)
        lay.addWidget(QLabel("K0 (init)"), 3, 0)
        lay.addWidget(self.sl_k0, 3, 1)
        lay.addWidget(self.lb_k0, 3, 2)
        lay.addWidget(QLabel("Compute"), 4, 0)
        lay.addWidget(self.cb_backend, 4, 1, 1, 2)
        return g

    def _rod_group(self) -> QGroupBox:
        g = QGroupBox("Moving Rod")
        lay = QGridLayout(g)
        lay.setVerticalSpacing(8)

        self.sp_NR = QSpinBox()
        self.sp_NR.setRange(2, 100)
        self.sp_NR.setValue(15)
        self.sp_NR.valueChanged.connect(self._on_nr)

        self.sl_vr, self.lb_vr = self._slider(1, 100, 10, self._on_vr)

        self.lb_R = QLabel("-")
        self.lb_R.setObjectName("statval")

        lay.addWidget(QLabel("NR (2pi=NR*R)"), 0, 0)
        lay.addWidget(self.sp_NR, 0, 1, 1, 2)
        lay.addWidget(QLabel("VR (2pi/s)"), 1, 0)
        lay.addWidget(self.sl_vr, 1, 1)
        lay.addWidget(self.lb_vr, 1, 2)
        lay.addWidget(QLabel("R (radius)"), 2, 0)
        lay.addWidget(self.lb_R, 2, 1, 1, 2)
        return g

    def _display_group(self) -> QGroupBox:
        g = QGroupBox("Display")
        lay = QGridLayout(g)
        lay.setVerticalSpacing(8)

        self.cb_field = QComboBox()
        self.cb_field.addItems(FIELD_NAMES)
        self.cb_field.currentTextChanged.connect(self._on_field)

        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(COLORMAP_NAMES)
        self.cb_cmap.setCurrentText("Ocean")
        self.cb_cmap.currentTextChanged.connect(self._on_cmap)

        self.cb_skip = QComboBox()
        self.cb_skip.addItems([str(s) for s in FRAMESKIP_OPTIONS])
        self.cb_skip.currentTextChanged.connect(self._on_skip)

        lay.addWidget(QLabel("Field"), 0, 0)
        lay.addWidget(self.cb_field, 0, 1)
        lay.addWidget(QLabel("Colors"), 1, 0)
        lay.addWidget(self.cb_cmap, 1, 1)
        lay.addWidget(QLabel("Frame / n steps"), 2, 0)
        lay.addWidget(self.cb_skip, 2, 1)
        return g

    def _buttons(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_restart = QPushButton("Restart")
        self.btn_restart.clicked.connect(self.build_and_start)
        lay.addWidget(self.btn_pause)
        lay.addWidget(self.btn_restart)
        return w

    def _stats_group(self) -> QGroupBox:
        g = QGroupBox("Statistics")
        lay = QGridLayout(g)
        lay.setVerticalSpacing(6)
        self._stat_labels: dict[str, QLabel] = {}
        rows = [
            ("step", "step"), ("time", "time"), ("dt", "dt"),
            ("energy", "energy"), ("enstrophy", "enstrophy"),
            ("max_omega", "max|omega|"), ("fps", "steps/s"),
        ]
        for i, (key, text) in enumerate(rows):
            name = QLabel(text)
            name.setObjectName("statname")
            val = QLabel("-")
            val.setObjectName("statval")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setMinimumWidth(120)
            self._stat_labels[key] = val
            lay.addWidget(name, i, 0)
            lay.addWidget(val, i, 1)
        return g

    def _slider(self, lo, hi, val, cb):
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        lab = QLabel("-")
        lab.setObjectName("statval")
        lab.setMinimumWidth(44)
        lab.setAlignment(Qt.AlignRight)
        s.valueChanged.connect(cb)
        return s, lab

    # ----- live control slots -----
    def _on_cfl(self, v: int) -> None:
        self.lb_cfl.setText(f"{v / 100:.2f}")
        if self.worker:
            self.worker.solver.cfl = v / 100.0

    def _on_k0(self, v: int) -> None:
        self.lb_k0.setText(str(v))  # applied on Restart

    def _on_nr(self, v: int) -> None:
        self.lb_R.setText(f"{2 * np.pi / v:.3f}")  # applied on Restart

    def _on_vr(self, v: int) -> None:
        self.lb_vr.setText(str(v))
        if self.worker:
            self.worker.solver.vr = float(v)

    def _on_field(self, name: str) -> None:
        if self.worker:
            self.worker.field_name = name

    def _on_cmap(self, name: str) -> None:
        self._table = self._qt_table(colormap(name))

    def _on_skip(self, text: str) -> None:
        if self.worker:
            self.worker.frameskip = int(text)

    def _toggle_pause(self) -> None:
        if not self.worker:
            return
        paused = self.btn_pause.text() == "Pause"
        self.worker.set_paused(paused)
        self.btn_pause.setText("Resume" if paused else "Pause")

    # ----- lifecycle -----
    def _make_solver(self, backend: str) -> Solver:
        return Solver(
            N=int(self.cb_N.currentText()),
            Re=float(self.sp_Re.value()),
            cfl=self.sl_cfl.value() / 100.0,
            k0=self.sl_k0.value(),
            NR=float(self.sp_NR.value()),
            vr=float(self.sl_vr.value()),
            backend=backend,
        )

    def build_and_start(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        backend = self.cb_backend.currentText().lower()  # "auto" / "cpu" / "gpu"
        try:
            solver = self._make_solver(backend)
        except Exception as exc:  # e.g. GPU requested but CuPy/CUDA missing
            print(f"[turbosim] {backend!r} backend failed ({exc}); using CPU.")
            self.cb_backend.setCurrentText("CPU")
            solver = self._make_solver("cpu")

        # Refresh derived labels.
        self._on_cfl(self.sl_cfl.value())
        self._on_k0(self.sl_k0.value())
        self._on_vr(self.sl_vr.value())
        self.lb_R.setText(f"{solver.R:.3f}")
        self.setWindowTitle(f"turbosim - 2D decaying turbulence [{solver.backend.upper()}]")

        self.worker = SimWorker(solver)
        self.worker.frameskip = int(self.cb_skip.currentText())
        self.worker.field_name = self.cb_field.currentText()
        self.worker.frameReady.connect(self._on_frame, Qt.QueuedConnection)
        self.btn_pause.setText("Pause")
        self.worker.start()

    def _on_frame(self, u8: np.ndarray, stats: dict) -> None:
        self._last_stats = stats
        h, w = u8.shape
        # Wrap the contiguous buffer directly; QPixmap.fromImage copies it.
        img = QImage(u8.data, w, h, w, QImage.Format_Indexed8)
        img.setColorTable(self._table)
        pix = QPixmap.fromImage(img)

        target = self.canvas.size()
        scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.FastTransformation)
        self._draw_rod(scaled, stats)
        self.canvas.setPixmap(scaled)
        self._update_stats(stats)

    def _draw_rod(self, pix: QPixmap, stats: dict) -> None:
        L = stats.get("L", 2 * np.pi)
        w = pix.width()
        s = w / L  # pixmap is square, same scale both axes
        r = stats["R"] * s
        cx = stats["xc"] * s
        cy = stats["yc"] * s
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(255, 255, 255, 150))
        pen.setWidthF(1.4)
        painter.setPen(pen)
        for dy in (-L * s, 0.0, L * s):  # periodic copies for the wrap
            painter.drawEllipse(int(cx - r), int(cy + dy - r), int(2 * r), int(2 * r))
        painter.end()

    def _update_stats(self, stats: dict) -> None:
        fmt = {
            "step": f"{int(stats['step']):>10d}",
            "time": f"{stats['time']:>10.4f}",
            "dt": f"{stats['dt']:>10.2e}",
            "energy": f"{stats['energy']:>10.4e}",
            "enstrophy": f"{stats['enstrophy']:>10.4e}",
            "max_omega": f"{stats['max_omega']:>10.3f}",
            "fps": f"{stats['fps']:>10.1f}",
        }
        for key, label in self._stat_labels.items():
            label.setText(fmt[key])

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop()
        super().closeEvent(event)


def _mono_font(size: int = 10) -> QFont:
    """First installed monospace family from our preference list.

    Picking a family that actually exists avoids Qt's costly font-alias scan
    (and the "missing font family" warning) for names like 'SF Mono'.
    """
    installed = set(QFontDatabase.families())
    font = QFont()
    font.setPointSize(size)
    for fam in ("SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono"):
        if fam in installed:
            font.setFamily(fam)
            break
    font.setStyleHint(QFont.Monospace)
    return font


def run() -> int:
    app = QApplication.instance() or QApplication([])
    app.setFont(_mono_font(10))
    win = MainWindow()
    win.show()
    return app.exec()
