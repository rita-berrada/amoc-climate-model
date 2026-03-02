# AMOC Box Model

[![CI](https://github.com/rita-berrada/amoc-box-model/actions/workflows/ci.yml/badge.svg)](https://github.com/rita-berrada/amoc-box-model/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Stommel-type **3-box thermohaline circulation model** of the Atlantic Meridional Overturning Circulation (AMOC), developed during CentraleSupélec Modelling Week (January 2025).

The model captures the key non-linearity responsible for AMOC tipping-point behaviour: the competition between a stabilising thermal gradient and a destabilising haline (salinity) gradient under freshwater forcing. Both deterministic (LSODA) and stochastic (Euler–Maruyama) integration schemes are implemented, with validation against real CMEMS ocean reanalysis data.

---

## Scientific Background

The AMOC transports approximately **17 Sv** (1 Sv = 10⁶ m³ s⁻¹) of warm, salty water northward at the surface and cold, dense water southward at depth, playing a critical role in regulating European and global climate. Palaeoclimate records (Dansgaard–Oeschger events) and CMIP6 projections indicate this circulation could weaken significantly or collapse under anthropogenic freshwater input from Greenland ice melt — a **climate tipping point** of global consequence.

This model explores that tipping-point behaviour using a minimal but physically grounded reduced-order framework.

---

## Model Equations

The state vector is **y = [T₁, T₂, S₁]** where:

| Variable | Meaning |
|----------|---------|
| T₁, S₁  | Temperature and salinity of box 1 (high latitudes, 60–80°N) |
| T₂, S₂  | Temperature and salinity of box 2 (equatorial, −10 to +10°) |

**Salinity closure** (global salt conservation):
```
S₂ = 2·S̄ − S₁,    S̄ = 35 psu
```

**AMOC flow** (density-driven, in Sverdrups):
```
m = μ · [(T₂ − T₁) − δ · (S₂ − S₁)]
```

**Prognostic equations**:
```
dT₁/dt = −(T₁ − T₁ʳ)/τ  +  |m|/V · (T₂ − T₁)
dT₂/dt = −(T₂ − T₂ʳ)/τ  +  |m|/V · (T₁ − T₂)
dS₁/dt =                     |m|/V · (S₂ − S₁)  +  F
```

| Symbol | Meaning | Default |
|--------|---------|---------|
| μ      | Flow sensitivity to density gradient | 1.5 Sv °C⁻¹ |
| δ      | Haline-to-thermal contraction ratio | 8 |
| τ      | SST restoring timescale | 10 yr |
| V      | Box volume scaling | 300 × 10³ km³ |
| F      | Freshwater forcing (negative = freshening) | swept in analysis |
| T₁ʳ, T₂ʳ | Restoring temperatures (obs-initialised) | from ARMOR3D |

**Stochastic forcing** (Euler–Maruyama scheme):
```
yₙ₊₁ = yₙ + f(yₙ, F)·Δt + σ·√Δt·ξₙ,   ξₙ ~ N(0, I)
```

---

## Repository Structure

```
amoc-box-model/
├── amoc/                    # Importable Python package
│   ├── __init__.py
│   ├── config.py            # ModelConfig dataclass — all parameters
│   ├── model.py             # BoxModel: ODE tendencies, AMOC flow
│   ├── integrators.py       # deterministic_solve, euler_maruyama
│   ├── analysis.py          # bifurcation_sweep, lorenz_attractor
│   ├── data_loader.py       # load_obs: CMEMS data loading
│   └── plotting.py          # All Matplotlib / Cartopy figures
│
├── notebooks/
│   └── AMOC_Modelling.ipynb # Full analysis pipeline
│
├── tests/                   # pytest unit tests
│   ├── test_model.py
│   ├── test_integrators.py
│   └── test_data_loader.py
│
├── scripts/
│   └── download_data.py     # CMEMS ARMOR3D data downloader
│
├── data/
│   └── AMOC_index.nc        # Observational AMOC index (13 KB, committed)
│                            # ocean_obs.nc (80 MB) — download via script
│
├── figures/                 # Generated output (git-ignored)
├── .github/workflows/ci.yml # GitHub Actions CI (Python 3.10–3.12)
├── pyproject.toml
├── requirements.txt
└── LICENSE                  # MIT
```

---

## Installation

```bash
git clone https://github.com/rita-berrada/amoc-box-model.git
cd amoc-box-model
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Obtaining the ocean reanalysis data

The 80 MB ARMOR3D dataset (`ocean_obs.nc`) is not committed to this repository. Download it with:

```bash
pip install copernicusmarine
python scripts/download_data.py
# You will be prompted for your free CMEMS account credentials
# Register at: https://marine.copernicus.eu/
```

The 13 KB AMOC observational index (`data/AMOC_index.nc`) is already included in the repository.

---

## Quick Start

```python
from amoc import BoxModel, ModelConfig
from amoc.integrators import deterministic_solve
from amoc.plotting import plot_time_series

# Configure and run the model
cfg   = ModelConfig(mu=1.5, delta=8.0, tau=10.0)
model = BoxModel(cfg, T1r=5.2, T2r=26.1)   # obs-initialised restoring temps

sol   = deterministic_solve(model, F=-0.01)
fig   = plot_time_series(sol, title_suffix="F = -0.01 psu yr⁻¹")
fig.savefig("figures/timeseries.png", dpi=150, bbox_inches="tight")
```

**Bifurcation analysis:**
```python
import numpy as np
from amoc.analysis import bifurcation_sweep
from amoc.plotting import plot_bifurcation

F_values = np.linspace(0.0, -0.08, 80)
F_vals, m_steady = bifurcation_sweep(model, F_values)
fig = plot_bifurcation(F_vals, m_steady)
```

**Stochastic run:**
```python
from amoc.integrators import euler_maruyama

y0  = np.array([5.2, 26.1, 35.0])
rng = np.random.default_rng(42)
sol = euler_maruyama(model, F=-0.02, y0=y0, rng=rng)
```

---

## Running the Tests

```bash
# All unit tests (no data files required)
pytest tests/ -m "not requires_data"

# Including obs-validation test (requires ocean_obs.nc)
pytest tests/
```

---

## Results

### Time Series
Evolution of box temperatures T₁, T₂, salinity S₁, and AMOC transport m under constant freshwater forcing.

### Bifurcation Diagram
Steady-state AMOC transport as a function of freshwater forcing F. A tipping point is observed near F ≈ −0.04 psu yr⁻¹ where the circulation abruptly collapses to near zero.

### Phase Space vs. Lorenz Attractor
The stochastic AMOC trajectory in (T₁, T₂, S₁) space compared to the canonical Lorenz attractor — illustrating deterministic chaos versus noise-driven dynamics in low-dimensional climate models.

---

## Data Sources

| Dataset | Product | Variables | Period |
|---------|---------|-----------|--------|
| [ARMOR3D](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012) | CMEMS MULTIOBS_GLO_PHY | Temperature, salinity | 1993–2022 |
| [AMOC index](https://data.marine.copernicus.eu/) | CMEMS reanalysis ensemble | amoc_mean [m³ s⁻¹] | 1993–2019 |

---

## Citation

If you use this model in your work, please cite:

```bibtex
@misc{berrada2025amoc,
  author = {Berrada, Rita},
  title  = {{AMOC Box Model}: A Stommel-type thermohaline circulation model},
  year   = {2025},
  url    = {https://github.com/rita-berrada/amoc-box-model},
  note   = {Developed at CentraleSupélec Modelling Week, January 2025}
}
```

---

## References

1. Stommel, H. (1961). *Thermohaline convection with two stable regimes of flow.* Tellus, 13(2), 224–230.
2. Rahmstorf, S. (1996). *On the freshwater forcing and transport of the Atlantic thermohaline circulation.* Climate Dynamics, 12(11), 799–811.
3. Lorenz, E. N. (1963). *Deterministic nonperiodic flow.* Journal of Atmospheric Sciences, 20(2), 130–141.
4. Lenton, T. M., et al. (2008). *Tipping elements in the Earth's climate system.* PNAS, 105(6), 1786–1793.

---

## License

MIT © 2025 Rita Berrada. See [LICENSE](LICENSE) for details.
