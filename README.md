# A Fuzzy Multi-Objective Model for Pickup-and-Delivery Operations in Intermodal Hub-and-Spoke Container Networks Under Transit and Border Uncertainty

**Python** • **MIT License** • **Paper** • **Data**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Benchmark](https://img.shields.io/badge/benchmark-168%20instances-green.svg)](fuzzy_solomon_modified/)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange.svg)]()

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

1. [Authors](#authors)
2. [Affiliation](#affiliation)
3. [Repository Structure](#repository-structure)
4. [Dataset Generation](#dataset-generation)
5. [Benchmark Results](#benchmark-results)
6. [East African Case Study](#east-african-case-study)
7. [Repository Statistics](#repository-statistics)
8. [Requirements](#requirements)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Citation](#citation)
12. [License](#license)
13. [Contact](#contact)
14. [Acknowledgments](#acknowledgments)

---

## Repository Structure

```
intermodal-fuzzy-container-pud/
├── fuzzy_solomon_modified/          # 168 modified Solomon benchmark instances (JSON)
├── results/                         # Complete benchmark results (CSV)
├── visualizations/                  # Benchmark performance figures (PNG)
├── east_africa_results/             # East African case study results
│   ├── figures/                     # 10 case study figures (PNG)
│   ├── tables/                      # 7 case study tables (CSV)
│   ├── data/                        # Raw assignment and cost data (JSON)
│   └── metrics/                     # Enhanced metrics (JSON)
├── src/                             # Source code for all algorithms
├── LICENSE                          # MIT License
└── README.md                        # This file
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

### Overall Results (168 instances, 10 runs per algorithm)

| Algorithm | Hub Balance | Total Cost (USD) | Total Time (hours) | Runtime (seconds) |
|-----------|-------------|------------------|--------------------|-------------------|
| **Aα-NSGA-II** | **0.6544 ± 0.0953** | $5,600 ± $2,611 | 470 ± 230 | 0.121 ± 0.041 |
| NSGA-II | 0.5664 ± 0.1526 | $5,556 ± $2,575 | 473 ± 232 | 0.114 ± 0.036 |
| SMS-EMOA | 0.5867 ± 0.1318 | $5,529 ± $2,615 | 474 ± 229 | 0.024 ± 0.011 |
| RVEA | 0.5995 ± 0.1319 | $5,567 ± $2,636 | 472 ± 228 | 0.070 ± 0.028 |

### Performance by Instance Size

| Size (Hubs) | Aα-NSGA-II | NSGA-II | SMS-EMOA | RVEA |
|-------------|------------|---------|----------|------|
| 25 customers (2 hubs) | 0.705 | 0.695 | 0.700 | 0.705 |
| 50 customers (3 hubs) | 0.690 | 0.600 | 0.620 | 0.630 |
| 100 customers (4 hubs) | 0.580 | 0.410 | 0.460 | 0.480 |

### Key Benchmark Findings

- **15.5% improvement** in hub balance over NSGA-II overall
- **41.5% improvement** on large-scale instances (100 customers, 4 hubs)
- **24% faster convergence** (32 vs 42 generations)
- Statistically significant results (p < 0.001, Cohen's d = 6.36)

### Benchmark Visualizations

| Figure | Description |
|--------|-------------|
| `hub_balance_bar.png` | Bar chart comparing hub balance across algorithms |
| `hub_balance_by_size.png` | Hub balance by instance size (25/50/100 customers) |
| `cost_vs_balance.png` | Scatter plot of cost vs hub balance trade-off |
| `computational_time.png` | Runtime comparison bar chart |

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
| **Aα-NSGA-II** | 460 | 46.0% | 340 | 42.5% | ✓ |
| NSGA-II | 800 | 80.0% | 0 | 0.0% | ⚠ |
| SMS-EMOA | 510 | 51.0% | 290 | 36.2% | ✓ |
| RVEA | 460 | 46.0% | 340 | 42.5% | ✓ |

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
| **Aα-NSGA-II** | **$4,494** | 69.37 | $147,921 |
| NSGA-II | $5,174 | 77.99 | $170,302 |
| SMS-EMOA | $4,708 | 72.60 | $154,963 |
| RVEA | $4,498 | 69.37 | $148,067 |

### Statistical Significance (Wilcoxon, α=0.05)

| Comparison | p-value | Significant | Better | Cohen's d | Effect Size |
|------------|---------|-------------|--------|-----------|-------------|
| Aα-NSGA-II vs NSGA-II | 0.0020 | ✓ Yes | Aα-NSGA-II | 6.36 | Large |
| Aα-NSGA-II vs SMS-EMOA | 0.0020 | ✓ Yes | Aα-NSGA-II | 4.48 | Large |
| Aα-NSGA-II vs RVEA | 0.0020 | ✓ Yes | Aα-NSGA-II | 2.68 | Large |

### Key Case Study Findings

- **13.1% cost reduction** ($1,775,055 vs $2,043,630), saving $268,575 per planning period
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
git clone https://github.com/YvesNDIKURIYO-2022/intermodal-fuzzy-container-pud.git
cd intermodal-fuzzy-container-pud
pip install numpy pandas matplotlib seaborn scipy
```

## Usage

```bash
# Run benchmark
python src/benchmark.py

# Run case study
python src/case_study.py
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{ndikuriyo2026fuzzy,
  title={A Fuzzy Multi-Objective Model for Pickup-and-Delivery Operations in Intermodal Hub-and-Spoke Container Networks Under Transit and Border Uncertainty},
  author={Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  year={2026}
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
  <sub>Built with Python | Fuzzy Multi-Objective Optimization for Intermodal Container Logistics | 📊 168 Benchmark Instances | 🔬 4 Algorithms (Aα-NSGA-II, NSGA-II, SMS-EMOA, RVEA) | 🌍 East African Corridor Case Study</sub>
</div>

<div align="center">
  <sub>© 2026 Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom — Central South University, China</sub>
</div>• Last updated: June 2026</sub>
