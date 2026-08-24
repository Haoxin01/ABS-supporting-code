# Supporting code for the ABS study

This repository contains the Python source code used for the heat-transfer simulations, machine-learning analyses, and techno-economic analysis (TEA) reported in the associated paper.

## Contents

| File | Description |
| --- | --- |
| `supplementary_data_1_heat_transfer.py` | Two-dimensional transient heat-conduction simulations for single- and double-layer carbon-paper configurations, including probe-temperature histories and temperature maps. |
| `supplementary_data_2_recovery_purity_gpr.py` | Gaussian-process regression of Sb recovery and purity, leave-one-out cross-validation, interpolation-based data augmentation, and feasibility mapping. |
| `supplementary_data_3_voltage_time_feasibility.py` | Gaussian-process models linking voltage and treatment time to temperature and heating rate, with a monotonic-temperature constraint and binary feasibility mapping. |
| `supplementary_data_4_tea_monte_carlo.py` | Monte Carlo uncertainty analysis of process costs for solvent-based treatment, pyrolysis, and IC-FJH, with summary tables and distribution plots. |

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run each analysis from the repository directory:

```bash
python supplementary_data_1_heat_transfer.py
python supplementary_data_2_recovery_purity_gpr.py
python supplementary_data_3_voltage_time_feasibility.py
python supplementary_data_4_tea_monte_carlo.py
```

The heat-transfer script additionally requires `carbon_temp_curve_ABS.xlsx`, supplied with the paper's source data, in the same directory. The other scripts contain the numerical inputs used by their respective analyses. Output CSV, NumPy, PNG, and SVG files are written to the current working directory.

## Reproducibility notes

- Random seeds are fixed in the machine-learning and Monte Carlo scripts.
- Model assumptions, numerical settings, thresholds, and uncertainty parameters are defined near the beginning of each script.
- The source-code contents are preserved from the versions supplied with the paper; only the filenames were standardized for this repository.

## License

This project is licensed under the MIT License. See `LICENSE`.

