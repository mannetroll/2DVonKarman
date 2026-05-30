"""Entry point: ``uv run simulation`` launches the GUI."""

from __future__ import annotations

import sys


def main() -> int:
    from turbosim.gui import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
