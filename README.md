# bi-poly-particle-mixing-dem-sub-model

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-RNN-FF6F00?logo=tensorflow&logoColor=white)
![YADE](https://img.shields.io/badge/YADE-DEM-2C5282)

Repository dedicated to the development of the project titled **Polyhedral-Based DEM Surrogate Modeling for Time Series Prediction in Bidisperse Particle Mixing and Segregation** as terminal project, conformant of the UNISON MCD academic program.

## Contents

- **0_simulation** — YADE helpers: chord-box ingress (`ingress_func_v1.py`), mill setup, particle I/O, and balance / rotation utilities (`s1_sim_functions.py`).
- **1_data_processing** — CSV to Parquet conversion and particle-size consistency checks across Parquet frames.
- **2_model / 1_RNNSR** — Sequence models and frame-wise prediction scripts (TensorFlow where used).
- **3_metrics / 0_lacey_mixing_index** — Lacey mixing index over time from ground-truth or predicted frames.
- **4_visualization** — Parquet frame animation and gridded particle views.
- **tests** — Pytest suite for pure Python paths; YADE-backed tests are skipped when YADE is not installed.

## Development

- Install dev dependencies: `pip install -r requirements-dev.txt`
- Run tests from the repo root: `python -m pytest tests -q`

## Relevant links

- [COST Action CA22132 — Working Groups and Membership](https://www.cost.eu/actions/CA22132/#tabs+Name:Working%20Groups%20and%20Membership)
- [On-DEM Confluence — Index overview](https://on-dem.atlassian.net/wiki/spaces/Index/overview?mode=global)
