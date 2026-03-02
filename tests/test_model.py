"""Unit tests for amoc.model.BoxModel."""
import numpy as np
import pytest
from amoc.config import ModelConfig
from amoc.model import BoxModel


@pytest.fixture
def default_model():
    cfg = ModelConfig()
    return BoxModel(cfg, T1r=5.0, T2r=25.0)


def test_tendency_shape(default_model):
    y = np.array([5.0, 25.0, 35.0])
    dydt = default_model.tendency(y, F=0.0)
    assert dydt.shape == (3,)


def test_tendency_returns_finite(default_model):
    y = np.array([5.0, 25.0, 35.0])
    dydt = default_model.tendency(y, F=-0.01)
    assert np.all(np.isfinite(dydt))


def test_amoc_flow_positive_at_equator_pole_contrast(default_model):
    """Positive temperature contrast (T2 > T1) should drive positive m."""
    y = np.array([5.0, 25.0, 35.0])
    m = default_model.amoc_flow(y)
    assert m > 0, f"Expected positive AMOC, got {m}"


def test_amoc_flow_clipping(default_model):
    """Extreme state values must be clipped to m_clip bounds."""
    y = np.array([0.0, 100.0, 0.0])
    m = default_model.amoc_flow(y)
    lo, hi = default_model.cfg.m_clip
    assert lo <= m <= hi, f"m={m} outside clip bounds {lo, hi}"


def test_freshwater_forcing_affects_salinity(default_model):
    """dS1/dt must differ by exactly ΔF between two runs."""
    y = np.array([5.0, 25.0, 35.0])
    dydt_pos = default_model.tendency(y, F=0.1)
    dydt_neg = default_model.tendency(y, F=-0.1)
    assert abs((dydt_pos[2] - dydt_neg[2]) - 0.2) < 1e-10


def test_scipy_rhs_signature(default_model):
    """scipy_rhs must accept (t, y, F) and return shape (3,)."""
    y = np.array([5.0, 25.0, 35.0])
    result = default_model.scipy_rhs(0.0, y, -0.01)
    assert result.shape == (3,)


def test_pure_restoring_when_no_flow(default_model):
    """When m=0 (T1=T2, S1=S2=Smoy), dT/dt equals the pure restoring term."""
    cfg = default_model.cfg
    T_common = 15.0   # T1 == T2 → no thermal gradient → m=0
    y = np.array([T_common, T_common, cfg.Smoy])
    dydt = default_model.tendency(y, F=0.0)
    # With m=0: dT1/dt = -(T1 - T1r)/tau
    expected_dT1 = -(T_common - default_model.T1r) / cfg.tau
    expected_dT2 = -(T_common - default_model.T2r) / cfg.tau
    assert abs(dydt[0] - expected_dT1) < 1e-12
    assert abs(dydt[1] - expected_dT2) < 1e-12
