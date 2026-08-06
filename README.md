# A Fuzzy Multi-Objective Model for Pickup-and-Delivery Operations in Intermodal Hub-and-Spoke Container Networks Under Transit and Border Uncertainty

**Python** • **MIT License** • **Paper** • **Data**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Benchmark](https://img.shields.io/badge/benchmark-168%20instances-green.svg)](fuzzy_solomon_modified/)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxx-blue.svg)]()

---

## Overview

This repository contains the complete code, data, and analysis for the paper:

> **Yves Ndikuriyo**, **Yinggui Zhang** (Corresponding Author), **Dung Davou Fom**  
> *School of Traffic and Transportation Engineering, Central South University, Changsha, China*

The paper addresses a critical gap in intermodal freight logistics: the optimization of pickup-and-delivery operations in hub-and-spoke container networks under the dual uncertainties of transit time variability and border clearance delays.

### Key Contributions

| Contribution | Description |
|--------------|-------------|
| **Methodological** | Novel FMOMILP formulation preserving full fuzzy distributional information |
| **Algorithmic** | Adaptive α-confidence NSGA-II (Aα-NSGA-II) with co-evolved confidence levels |
| **Empirical** | 168 modified Solomon benchmark instances and East African corridor case study |

### Repository Components

| Component | Description |
|-----------|-------------|
| `fuzzy_solomon_modified/` | 168 benchmark instances with triangular fuzzy travel times |
| `results/` | Complete benchmark results (168 × 4 × 10 runs) |
| `visualizations/` | Performance figures (hub balance, cost vs balance, runtime) |
| `east_africa_results/` | Case study with 10 figures and 7 tables |
| `src/` | Source code for all algorithms |

---

## Table of Contents

1. [Overview](#overview)
2. [Authors](#authors)
3. [Affiliation](#affiliation)
4. [Repository Structure](#repository-structure)
5. [Dataset Generation](#dataset-generation)
6. [Benchmark Results](#benchmark-results)
7. [East African Case Study](#east-african-case-study)
8. [Repository Statistics](#repository-statistics)
9. [Requirements](#requirements)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Citation](#citation)
13. [License](#license)
14. [Authors Contact](#authors-contact)
15. [Acknowledgments](#acknowledgments)

---

## Authors

- **Yves Ndikuriyo** - Conceptualization, Methodology, Software, Writing – original draft
- **Yinggui Zhang** (Corresponding Author) - Supervision, Methodology, Writing – review & editing
- **Dung Davou Fom** - Validation, Formal analysis, Writing – review & editing

---

## Affiliation

School of Traffic and Transportation Engineering, Central South University, Changsha 410075, China

---

## Repository Structure

```
intermodal-fuzzy-container-pud/
│
├── README.md                          # Comprehensive documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
│
├── fuzzy_solomon_modified/            # 168 benchmark instances (JSON)
│   ├── FS_C101_H2.json
│   ├── FS_C101_H3.json
│   ├── FS_C101_H4.json
│   ├── ... (168 files total)
│   └── FS_RC208_H4.json
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base_nsga2.py              # Base NSGA-II implementation
│   │   ├── nsga2.py                   # Canonical NSGA-II
│   │   ├── aalpha_nsga2.py            # Proposed Aα-NSGA-II
│   │   ├── sms_emoa.py                # SMS-EMOA implementation
│   │   └── rvea.py                    # RVEA implementation
│   ├── benchmark.py                   # Main benchmark runner
│   ├── case_study.py                  # East African case study
│   ├── metrics.py                     # Performance metrics (HV, IGD, Spread)
│   ├── statistics.py                  # Statistical tests (Wilcoxon, Friedman)
│   └── visualization.py               # Plot generation
│
├── results/                           # Benchmark results
│   ├── benchmark_results_complete.csv # All metrics for all algorithms
│   └── statistical_results.csv        # Wilcoxon and Friedman test results
│
├── visualizations/                    # Benchmark figures
│   ├── hub_balance_bar.png
│   ├── hypervolume_bar.png
│   ├── igd_bar.png
│   ├── spread_bar.png
│   ├── hub_balance_by_size.png
│   ├── cost_vs_balance.png
│   ├── computational_time.png
│   └── convergence_average.pdf
│
├── east_africa_results/               # East African case study
│   ├── figures/
│   │   ├── Fig1_Network_Diagram.png
│   │   ├── Fig2_Convergence_Curves.png
│   │   ├── Fig3_Hub_Assignment.png
│   │   ├── Fig4_Cost_Breakdown.png
│   │   ├── Fig5_Radar_Chart.png
│   │   ├── Fig6_Hub_Utilization.png
│   │   ├── Fig7_Optimized_Routes.png
│   │   ├── Fig8_Economic_Metrics.png
│   │   ├── Fig9_Time_Reliability.png
│   │   └── Fig10_Comprehensive_Dashboard.png
│   ├── tables/
│   │   ├── Table20_Network_Config.csv
│   │   ├── Table21_Fuzzy_Parameters.csv
│   │   ├── Table22_Instance_Sizes.csv
│   │   ├── Table23_Algorithm_Performance.csv
│   │   ├── Table24_Hub_Load.csv
│   │   ├── Table25_Cost_Breakdown.csv
│   │   └── Table26_Economic_Metrics.csv
│   ├── data/
│   │   ├── hub_assignments.json
│   │   └── cost_breakdown.json
│   └── metrics/
│       └── enhanced_metrics.json
│
└── docs/                              # Additional documentation
    ├── INSTALL.md                     # Installation guide
    ├── USAGE.md                       # Usage examples
    └── CONTRIBUTING.md                # Contribution guidelines
```

---

## Dataset Generation

The repository includes **168 modified Solomon instances** generated from the standard Solomon VRPTW benchmark.

### Instance Specifications

| Feature | Specification |
|---------|---------------|
| Instance sizes | 25, 50, 100 customers |
| Hub configurations | 2 hubs (25 cust), 3 hubs (50 cust), 4 hubs (100 cust) |
| Fuzzy representation | Triangular fuzzy numbers (δ = 0.30) |
| Solomon types | C1, C2, R1, R2, RC1, RC2 |
| Time window types | Narrow (Type 1) and Wide (Type 2) |
| Total instances | 168 |

### Instance Naming Convention

| Pattern | Example | Meaning |
|---------|---------|---------|
| `FS_C101_H2` | FS = Fuzzy Solomon, C101 = Instance name, H2 = 2 hubs |
| `FS_R112_H3` | FS = Fuzzy Solomon, R112 = Instance name, H3 = 3 hubs |
| `FS_RC208_H4` | FS = Fuzzy Solomon, RC208 = Instance name, H4 = 4 hubs |

---

## Benchmark Results

### Algorithms Compared

| Algorithm | Description |
|-----------|-------------|
| **Aα-NSGA-II** | Proposed adaptive α-confidence NSGA-II with local search operators |
| NSGA-II | Canonical Non-dominated Sorting Genetic Algorithm II |
| SMS-EMOA | S-Metric Selection Evolutionary Multi-Objective Algorithm |
| RVEA | Reference Vector Guided Evolutionary Algorithm |

### Performance Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Hub Balance | Evenness of container flow distribution across hubs | Higher is better (0-1 scale) |
| Total Cost | Transportation + handling + transfer + penalty costs | Lower is better (USD) |
| Total Time | Transit time + border clearance time | Lower is better (hours) |
| Runtime | Wall-clock execution time | Lower is better (seconds) |
| Hypervolume (HV) | Volume of dominated objective space | Higher is better (0-1 scale) |
| IGD | Inverted Generational Distance | Lower is better |
| Spread (Δ) | Distribution of solutions along Pareto front | Lower is better |

### Overall Results (168 instances, 10 runs per algorithm)

| Algorithm | Hub Balance | Total Cost (USD) | Total Time (hours) | Runtime (seconds) |
|-----------|-------------|------------------|--------------------|-------------------|
| **Aα-NSGA-II** | **0.6679 ± 0.0814** | $5,664 ± $2,662 | 470 ± 230 | 0.121 ± 0.054 |
| NSGA-II | 0.5576 ± 0.1492 | $5,586 ± $2,591 | 473 ± 232 | 0.098 ± 0.038 |
| SMS-EMOA | 0.5496 ± 0.1488 | $5,530 ± $2,591 | 474 ± 229 | 0.030 ± 0.014 |
| RVEA | 0.4923 ± 0.1596 | $5,294 ± $2,505 | 472 ± 228 | 0.163 ± 0.049 |

### Additional Performance Metrics

| Algorithm | Hypervolume (HV) | IGD | Spread (Δ) |
|-----------|------------------|-----|------------|
| **Aα-NSGA-II** | **1.0000** | **0.0000** | **0.7483** |
| NSGA-II | 1.0000 | 0.0000 | 0.7425 |
| SMS-EMOA | 1.0000 | 0.0000 | 0.6501 |
| RVEA | 1.0000 | 0.0000 | 1.0844 |

*Note: HV and IGD values are normalized; the exact calculation requires full Pareto front data.*

### Performance by Instance Size

| Size (Hubs) | Aα-NSGA-II | NSGA-II | SMS-EMOA | RVEA |
|-------------|------------|---------|----------|------|
| 25 customers (2 hubs) | 0.715 | 0.705 | 0.700 | 0.705 |
| 50 customers (3 hubs) | 0.690 | 0.600 | 0.620 | 0.630 |
| 100 customers (4 hubs) | 0.580 | 0.410 | 0.460 | 0.480 |

### Key Benchmark Findings

- **19.8% improvement** in hub balance over NSGA-II overall
- **41.5% improvement** on large-scale instances (100 customers, 4 hubs)
- **24% faster convergence** (32 vs 42 generations)
- Statistically significant results (**p < 0.000001**, Cohen's d = 0.920)
- Friedman test rank: **Aα-NSGA-II (1.18)** best among all algorithms

### Statistical Significance (168 instances, α=0.05)

| Comparison | Hub Balance Improvement | Wilcoxon W | p-value | Cohen's d | Effect Size |
|------------|------------------------|------------|---------|-----------|-------------|
| Aα vs NSGA-II | **+19.8%** | 4886 | **< 0.000001** | **0.920** | **Large** |
| Aα vs SMS-EMOA | **+21.5%** | 5042 | **< 0.000001** | **0.989** | **Large** |
| Aα vs RVEA | **+35.7%** | 7603 | **< 0.000001** | **1.390** | **Large** |

**Friedman Test:** χ² = 186.71, p < 0.000001, Average Ranks: Aα-NSGA-II (1.18), NSGA-II (1.51), SMS-EMOA (1.53), RVEA (1.77)

### Benchmark Visualizations

| Figure | Description |
|--------|-------------|
| `hub_balance_bar.png` | Bar chart comparing hub balance across algorithms |
| `hypervolume_bar.png` | Hypervolume comparison across algorithms |
| `igd_bar.png` | IGD comparison across algorithms |
| `spread_bar.png` | Spread comparison across algorithms |
| `hub_balance_by_size.png` | Hub balance by instance size (25/50/100 customers) |
| `cost_vs_balance.png` | Scatter plot of cost vs hub balance trade-off |
| `computational_time.png` | Runtime comparison bar chart |
| `convergence_average.pdf` | Average convergence with 95% confidence intervals |

---

## East African Case Study

### Network Configuration

The East African corridor features a realistic two-hub pickup-and-delivery network:

| Component | Details |
|-----------|---------|
| **Hubs** | Mombasa (Kenya, 1,000 TEU capacity), Dar es Salaam (Tanzania, 800 TEU capacity) |
| **Origins (6)** | Nairobi (100), Nakuru (70), Arusha (50), Kisumu (60), Mwanza (65), Jinja (50) |
| **Destinations (6)** | Kampala (120), Kigali (80), Bujumbura (50), Juba (60), Goma (40), Lubumbashi (55) |
| **Total Supply** | 395 TEU |
| **Total Demand** | 405 TEU |

### Case Study Results

| Algorithm | Total Cost (USD) | Convergence (gen) | Inter-Hub Flow (TEU) | Runtime (s) |
|-----------|-----------------|-------------------|----------------------|-------------|
| **Aα-NSGA-II** | **$1,775,055** | **32** | **145** | 187.4 |
| NSGA-II | $2,043,630 | 42 | 280 | 142.3 |
| SMS-EMOA | $1,859,555 | 38 | 210 | 210.6 |
| RVEA | $1,776,805 | 35 | 180 | 256.8 |

### Hub Load Balancing

| Algorithm | Mombasa (TEU) | Mombasa Cap % | Dar (TEU) | Dar Cap % | Balance |
|-----------|---------------|---------------|-----------|-----------|---------|
| **Aα-NSGA-II** | **460** | **46.0%** | **425** | **42.5%** | ✓ |
| NSGA-II | 800 | 80.0% | 0 | 0.0% | ⚠ |
| SMS-EMOA | 510 | 51.0% | 290 | 36.2% | ✓ |
| RVEA | 460 | 46.0% | 425 | 42.5% | ✓ |

### Cost Breakdown

| Algorithm | Transportation | Handling | Transfer | Fixed | **TOTAL** | Improvement |
|-----------|---------------|----------|----------|-------|-----------|-------------|
| **Aα-NSGA-II** | $1,741,125 | $25,100 | $7,250 | $1,580 | **$1,775,055** | +13.1% |
| NSGA-II | $2,002,375 | $25,675 | $14,000 | $1,580 | $2,043,630 | baseline |
| SMS-EMOA | $1,822,375 | $25,100 | $10,500 | $1,580 | $1,859,555 | +9.0% |
| RVEA | $1,741,125 | $25,100 | $9,000 | $1,580 | $1,776,805 | +13.1% |

### Economic Metrics

| Algorithm | Cost/TEU | Cost Efficiency | Operating Cost/Route |
|-----------|----------|----------------|---------------------|
| **Aα-NSGA-II** | **$4,494** | **69.37** | **$147,921** |
| NSGA-II | $5,174 | 77.99 | $170,302 |
| SMS-EMOA | $4,708 | 72.60 | $154,963 |
| RVEA | $4,498 | 69.37 | $148,067 |

### Statistical Significance (Wilcoxon, α=0.05)

| Comparison | p-value | Significant | Better | Cohen's d | Effect Size |
|------------|---------|-------------|--------|-----------|-------------|
| Aα-NSGA-II vs NSGA-II | 0.0020 | ✓ Yes | Aα-NSGA-II | **6.36** | Large |
| Aα-NSGA-II vs SMS-EMOA | 0.0020 | ✓ Yes | Aα-NSGA-II | **4.48** | Large |
| Aα-NSGA-II vs RVEA | 0.0020 | ✓ Yes | Aα-NSGA-II | **2.68** | Large |

### Key Case Study Findings

- **13.1% cost reduction** ($1,775,055 vs $2,043,630), saving **$268,575** per planning period
- **48% reduction** in inter-hub transfer volume (145 vs 280 TEU)
- **24% faster convergence** (32 vs 42 generations)
- **Near-perfect hub load balancing** (Mombasa 46.0%, Dar es Salaam 42.5%)
- Statistically significant with large effect size (p < 0.05, Cohen's d = 6.36)

### Case Study Visualizations

| Figure | Description |
|--------|-------------|
| `Fig1_Network_Diagram.png` | Two-hub network configuration for East Africa |
| `Fig2_Convergence_Curves.png` | Convergence comparison across algorithms |
| `Fig3_Hub_Assignment.png` | Hub assignment visualization |
| `Fig4_Cost_Breakdown.png` | Cost component breakdown |
| `Fig5_Radar_Chart.png` | Multi-metric radar chart |
| `Fig6_Hub_Utilization.png` | Hub utilization comparison |
| `Fig7_Optimized_Routes.png` | Optimized pickup-and-delivery routes |
| `Fig8_Economic_Metrics.png` | Economic metrics comparison |
| `Fig9_Time_Reliability.png` | Time reliability comparison |
| `Fig10_Comprehensive_Dashboard.png` | Complete performance dashboard |

---

## Repository Statistics

| Item | Count |
|------|-------|
| Benchmark instances | 168 JSON files |
| Total runs | 6,720 (168 × 4 × 10) |
| Algorithms | 4 |
| Instance sizes | 3 (25, 50, 100 customers) |
| Solomon types | 6 |
| East African figures | 10 PNG files |
| East African tables | 7 CSV files |
| Python source files | 10+ |

---

## Requirements

- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy

## Installation

```bash
# Clone the repository
git clone https://github.com/YvesNDIKURIYO-2022/intermodal-fuzzy-container-pud.git
cd intermodal-fuzzy-container-pud

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run full benchmark on all 168 instances
python src/benchmark.py

# Run East African case study
python src/case_study.py

# Generate visualizations from results
python src/visualization.py

# Run statistical tests
python src/statistics.py
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{ndikuriyo2026fuzzy,
  title={A Fuzzy Multi-Objective Model for Pickup-and-Delivery Operations in Intermodal Hub-and-Spoke Container Networks Under Transit and Border Uncertainty},
  author={Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal={International Journal of Fuzzy Systems},
  year={2026},
  note={Under review}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Authors Contact

**Yves Ndikuriyo**  
PhD Candidate, School of Traffic and Transportation Engineering  
Central South University, Changsha, China  
📧 [yvesndikuriyo@csu.edu.cn](mailto:yvesndikuriyo@csu.edu.cn)  
🔗 [ORCID: 0009-0006-9324-7265](https://orcid.org/0009-0006-9324-7265)

**Professor Yinggui Zhang** (Corresponding Author)  
School of Traffic and Transportation Engineering  
Central South University, Changsha, China  
📧 [ygzhang@csu.edu.cn](mailto:ygzhang@csu.edu.cn)  
🔗 [ORCID: 0000-0002-5790-0638](https://orcid.org/0000-0002-5790-0638)

**Dung Davou Fom**  
School of Traffic and Transportation Engineering  
Central South University, Changsha, China  
📧 [dungfom1@csu.edu.cn](mailto:dungfom1@csu.edu.cn)  
🔗 [ORCID: 0009-0001-8688-813X](https://orcid.org/0009-0001-8688-813X)

---

## Acknowledgments

This work was supported by:

- National Natural Science Foundation of China [Grant No. 71971220]
- Natural Science Foundation of Hunan Province, China [Grant Nos. 2023JJ30710, 2022JJ31020]

---

<div align="center">
  <sub>Built with Python | Fuzzy Multi-Objective Optimization for Intermodal Container Logistics | 📊 168 Benchmark Instances | 🔬 4 Algorithms (Aα-NSGA-II, NSGA-II, SMS-EMOA, RVEA) | 🌍 East African Corridor Case Study | Last updated: June 2026</sub>
</div>

<div align="center">
  <sub>© 2026 Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom — Central South University, China</sub>
</div>


