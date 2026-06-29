"""Simple smoke tests for the analysis pipeline.

Run with:  python -m pytest tests/   (or just  python tests/test_pipeline.py)

The tests check that the core components fit together and produce reasonable
numbers - not that the exact results are "correct" (that is an empirical matter).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the project root importable when the test is run directly (not via pytest).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.diebold_li import estimate_factors, nelson_siegel_loadings
from src.evaluation import dm_test


def test_loadings_shape_and_bounds():
    X = nelson_siegel_loadings(config.ROLL_MATURITY_MONTHS, lam=0.23)
    assert X.shape == (11, 3)  # maturities 2..12
    # The level loading is always 1; the slope loading falls from ~1 towards 0.
    assert np.allclose(X[:, 0], 1.0)
    assert X[0, 1] > X[-1, 1] > 0


def test_factor_fit_is_tight():
    """NS should fit the roll-return curve well (small RMSE in roll-return units)."""
    from src.data_acquisition import load_dataset, roll_return_curve

    roll = roll_return_curve(load_dataset())
    factors, rmse = estimate_factors(roll, config.ROLL_MATURITY_MONTHS, lam=0.23)
    assert factors.shape[1] == 3
    assert rmse < 0.05  # roll returns are ~0.0-0.15 in magnitude


def test_dm_test_symmetry():
    """The DM statistic should flip sign when the arguments are swapped."""
    rng = np.random.default_rng(0)
    e1 = rng.normal(size=120)
    e2 = rng.normal(size=120) * 1.5
    s_ab, p_ab = dm_test(e1, e2)
    s_ba, p_ba = dm_test(e2, e1)
    assert np.isclose(s_ab, -s_ba)
    assert np.isclose(p_ab, p_ba)
    assert 0.0 <= p_ab <= 1.0


if __name__ == "__main__":
    test_loadings_shape_and_bounds()
    test_factor_fit_is_tight()
    test_dm_test_symmetry()
    print("All smoke tests passed.")
