"""
amoc — Atlantic Meridional Overturning Circulation box model.

A Stommel-type 3-box thermohaline circulation model developed at
CentraleSupélec Modelling Week, January 2025.

Quick start
-----------
>>> from amoc import BoxModel, ModelConfig
>>> from amoc.integrators import deterministic_solve
>>> cfg   = ModelConfig()
>>> model = BoxModel(cfg, T1r=5.2, T2r=26.1)
>>> sol   = deterministic_solve(model, F=-0.01)
>>> print(f"Steady-state AMOC: {sol.m[-1]:.2f} Sv")
"""
from .model import BoxModel
from .config import ModelConfig

__all__ = ["BoxModel", "ModelConfig"]
__version__ = "0.1.0"
