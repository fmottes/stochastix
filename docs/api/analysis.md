# Analysis

The analysis module provides functions for analyzing simulation results, including correlation analysis, kernel density estimation (KDE), entropy estimation, and mutual information computation.

Helpers that interpolate trajectories default to `time_kernel_bw='adaptive'`.
They require full saved event histories and propensities; pass `time_kernel_bw=0`
for ordinary forward-fill reads, ODE results, or trajectories without these fields.
`state_kde` additionally requires an explicit `t` when its time kernel is enabled.
The KDE `bw_multiplier` remains `1.0`; it is separate from the time kernel.

## Correlation Functions

---
::: stochastix.analysis.autocorrelation

---
::: stochastix.analysis.cross_correlation

## Kernel Density Estimation

### 1D KDE

---
::: stochastix.analysis.kde

---
::: stochastix.analysis.kde_triangular

---
::: stochastix.analysis.kde_exponential

---
::: stochastix.analysis.kde_gaussian

---
::: stochastix.analysis.kde_wendland_c2

### 2D KDE

---
::: stochastix.analysis.kde_2d

---
::: stochastix.analysis.kde_triangular_2d

---
::: stochastix.analysis.kde_exponential_2d

---
::: stochastix.analysis.kde_gaussian_2d

---
::: stochastix.analysis.kde_wendland_c2_2d

### Simulation results

---
::: stochastix.analysis.state_kde

## Entropy

---
::: stochastix.analysis.entropy

---
::: stochastix.analysis.state_entropy

## Mutual Information

---
::: stochastix.analysis.mutual_information

---
::: stochastix.analysis.state_mutual_info
