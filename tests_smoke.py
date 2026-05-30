"""Head-less smoke tests for the numerical core + render pipeline (no GUI)."""
import numpy as np
import pytest

from turbosim.solver import Solver, FIELD_NAMES
from turbosim.render import display_limits, to_uint8
from turbosim.colormaps import colormap, COLORMAP_NAMES


@pytest.fixture(scope="module")
def stepped():
    """A small solver advanced 40 steps, with its initial energy recorded."""
    s = Solver(N=128, Re=10000, cfl=1.5, k0=5, NR=5, vr=0.1, seed=1)
    e0 = s.stats()["energy"]
    for _ in range(40):
        s.step()
    return s, e0


def test_energy_decays(stepped):
    s, e0 = stepped
    e = s.stats()["energy"]
    assert np.isfinite(e) and e > 0, "energy blew up / vanished"
    assert e <= e0 + 1e-9, "decaying flow gained energy"


def test_rod_moves(stepped):
    s, _ = stepped
    assert abs(s.stats()["yc"] - s.yc0) > 0, "rod did not move"


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_field_render_pipeline(stepped, name):
    s, _ = stepped
    f = s.get_field(name)
    assert f.shape == (s.N, s.N) and np.all(np.isfinite(f)), name
    vmin, vmax = display_limits(f, name)
    u8 = to_uint8(f, vmin, vmax)
    assert u8.dtype == np.uint8 and u8.flags["C_CONTIGUOUS"]
    assert u8.shape == (s.N, s.N)


@pytest.mark.parametrize("cm", COLORMAP_NAMES)
def test_colormap_table(cm):
    t = colormap(cm)
    assert t.shape == (256, 3) and t.dtype == np.uint8
