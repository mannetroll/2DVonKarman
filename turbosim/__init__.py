"""turbosim - 2D decaying turbulence pseudo-spectral solver with a PySide6 GUI.

The numerical core (:mod:`turbosim.solver`) depends only on NumPy/SciPy and can
be imported and exercised head-less.  The GUI lives in :mod:`turbosim.gui`.
"""

__version__ = "0.1.0"

from turbosim.solver import Solver

__all__ = ["Solver", "__version__"]
