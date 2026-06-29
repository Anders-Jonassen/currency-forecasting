"""Enkle røyktester for analyse-pipelinen.

Kjør med:  python -m pytest tests/  (eller bare python tests/test_pipeline.py)

Testene sjekker at kjernekomponentene henger sammen og gir rimelige tall – ikke
at de eksakte resultatene er "riktige" (det er en empirisk vurdering).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Gjør prosjektroten importerbar når testen kjøres direkte (ikke via pytest).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.diebold_li import estimate_factors, nelson_siegel_loadings
from src.evaluation import dm_test


def test_loadings_shape_and_bounds():
    X = nelson_siegel_loadings(config.MATURITY_MONTHS, lam=0.23)
    assert X.shape == (12, 3)
    # Nivå-loading er alltid 1; helning faller fra ~1 mot 0.
    assert np.allclose(X[:, 0], 1.0)
    assert X[0, 1] > X[-1, 1] > 0


def test_factor_fit_is_tight():
    """NS skal tilpasse den faktiske Brent-kurven svært godt (lav RMSE)."""
    from src.data_acquisition import load_dataset

    prices = load_dataset()[[f"M{m}" for m in config.MATURITY_MONTHS]]
    factors, rmse = estimate_factors(prices, lam=0.23)
    assert factors.shape[1] == 3
    assert rmse < 1.0  # USD/fat, mot et snitt på ~66


def test_dm_test_symmetry():
    """DM-statistikken skal snu fortegn når argumentene byttes om."""
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
    print("Alle røyktester passerte.")
