"""Smoke tests for numerical integrators."""
import numpy as np
import pytest
from amoc.config import ModelConfig
from amoc.model import BoxModel
from amoc.integrators import deterministic_solve, euler_maruyama, Solution


@pytest.fixture
def fast_model():
    """Short-horizon model for fast tests."""
    cfg = ModelConfig(t_end=50.0, n_eval=100, em_T=200.0, em_dt=0.1)
    return BoxModel(cfg, T1r=5.0, T2r=25.0)


# ── deterministic_solve ────────────────────────────────────────────────────────

def test_deterministic_returns_solution(fast_model):
    sol = deterministic_solve(fast_model, F=-0.01)
    assert isinstance(sol, Solution)


def test_deterministic_shape(fast_model):
    sol = deterministic_solve(fast_model, F=-0.01)
    assert sol.T1.shape == sol.t.shape
    assert sol.m.shape  == sol.t.shape


def test_deterministic_finite(fast_model):
    sol = deterministic_solve(fast_model, F=-0.01)
    for arr in (sol.T1, sol.T2, sol.S1, sol.m):
        assert np.all(np.isfinite(arr)), f"Non-finite values in {arr}"


def test_deterministic_default_y0(fast_model):
    """Solver should use [15, 20, 35] when y0 is not provided."""
    sol = deterministic_solve(fast_model, F=0.0)
    assert len(sol.t) == fast_model.cfg.n_eval


def test_deterministic_custom_y0(fast_model):
    y0 = np.array([3.0, 28.0, 34.5])
    sol = deterministic_solve(fast_model, F=-0.01, y0=y0)
    assert np.all(np.isfinite(sol.T1))


# ── euler_maruyama ─────────────────────────────────────────────────────────────

def test_euler_maruyama_shape(fast_model):
    y0  = np.array([5.0, 25.0, 35.0])
    rng = np.random.default_rng(42)
    sol = euler_maruyama(fast_model, F=-0.01, y0=y0, rng=rng)
    expected_len = int(fast_model.cfg.em_T / fast_model.cfg.em_dt) + 1
    assert len(sol.T1) == expected_len


def test_euler_maruyama_finite(fast_model):
    y0  = np.array([5.0, 25.0, 35.0])
    rng = np.random.default_rng(0)
    sol = euler_maruyama(fast_model, F=-0.01, y0=y0, rng=rng)
    for arr in (sol.T1, sol.T2, sol.S1, sol.m):
        assert np.all(np.isfinite(arr)), "Non-finite values in stochastic solution"


def test_euler_maruyama_reproducible(fast_model):
    """Same seed must produce identical trajectories."""
    y0 = np.array([5.0, 25.0, 35.0])
    sol1 = euler_maruyama(fast_model, F=-0.01, y0=y0, rng=np.random.default_rng(7))
    sol2 = euler_maruyama(fast_model, F=-0.01, y0=y0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(sol1.T1, sol2.T1)


# ── Solution properties ────────────────────────────────────────────────────────

def test_solution_s2_property(fast_model):
    sol = deterministic_solve(fast_model, F=-0.01)
    expected_S2 = 2.0 * 35.0 - sol.S1
    np.testing.assert_allclose(sol.S2, expected_S2)
