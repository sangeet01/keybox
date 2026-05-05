# KeyBox: Physics-Based Voxel System for API-Excipient Compatibility

**Open-source computational platform for predicting pharmaceutical formulation incompatibilities using 3D voxel field theory**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0 + Commons Clause](https://img.shields.io/badge/license-Apache%202.0%20%2B%20Commons%20Clause-orange.svg)](LICENSE)
[![Version: 0.8.0](https://img.shields.io/badge/version-0.8.0-brightgreen.svg)]()

<p align="center">
  <img src="/artifacts/nlogo.png" alt="KeyBox Logo" width="100%">
</p>


---

## Overview

KeyBox applies **engineering finite element analysis (FEA) principles to molecular chemistry**, creating a voxel-based field system that predicts API-excipient interactions through first-principles physics.

**Key Innovation**: Treats pharmaceutical formulations as 3D field problems with 16 simultaneous interaction channels. In v0.8.1, KeyBox introduces **Tier 3/4 Dynamics**, migrating from static snapshots to MD-parity Langevin trajectories and induced-fit protein breathing. It achieves sub-second simulation speeds with no tradeoffs in biological accuracy.

### Why KeyBox?

- **Physics-First**: Debye-Hückel, Lennard-Jones, Flory-Huggins, Arrhenius kinetics
- **Strictly Chemical**: No mock data -- requires valid SMILES and RDKit
- **Multi-Scale**: Molecular -> Process -> Microstructure -> Time
- **Quantitative**: Volume of Incompatibility (VOI) metrics with 95% CI
- **Generative**: Box-Behnken + Simplex-Centroid optimizer finds the optimal recipe
- **Dynamics-Driven**: Tier 3 (Langevin Trajectory) + Tier 4 (Induced Fit)
- **Gap Closure**: Virtual Water Networks (VWN) and Fukui Covalent triggers
- **Thermodynamic Integration**: VATI module for absolute delta-G estimation (kcal/mol)
- **Aerospace Voxel Engine**: C-native trajectory loops fitting in L3 cache
- **Complete Workflow**: Predict -> Simulate -> Diagnose -> Optimize -> Visualize

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sangeet01/keybox.git
cd keybox

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn rdkit plotly
```

### 1. Basic Compatibility Check

```python
from key import KeyBoxSystem, EnvironmentalConditions
from key.designer import create_enhanced_molecule_from_smiles

system = KeyBoxSystem(voxel_bounds=(20., 20., 20.), resolution=1.0, random_state=42)

aspirin = create_enhanced_molecule_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
lactose = create_enhanced_molecule_from_smiles("OCC1OC(O)C(O)C(O)C1O", "Lactose")

system.add_molecule(aspirin, is_api=True)
system.add_molecule(lactose, is_api=False)
system.env_conditions = EnvironmentalConditions(pH=7.0, moisture=0.6, temperature=40.0)

# Analyze formulation
results = system.analyze_enhanced_formulation()
print(f"Compatibility: {results['classification']}")
print(f"OCS: {results['metrics']['OCS']:.1f}")
```

### 2. BBD Optimization (Process Factors)

```python
from key import KeyBoxOptimizer

opt = KeyBoxOptimizer(api_smiles="CC(=O)Oc1ccccc1C(=O)O", api_name="Aspirin")
opt.add_factor("Binder_Ratio", low=5.0,  high=25.0)
opt.add_factor("Moisture",     low=0.3,  high=0.8)
opt.add_factor("Temperature",  low=25.0, high=50.0)
opt.add_fixed_excipient("Lactose", "OCC1OC(O)C(O)C(O)C1O")

results = opt.run_bbd_optimization()
print(f"Optimal OCS: {results['predicted_ocs']:.2f}")
print(f"Optimal Config: {results['optimal_params']}")
```

### 3. Combined Mixture-Process Optimization (NEW v0.8.0)

```python
from key import KeyBoxOptimizer

opt = KeyBoxOptimizer(api_smiles="CC(=O)Oc1ccccc1C(=O)O", api_name="Aspirin")

# Mixture components (proportions must sum to 1.0)
opt.add_mixture_component("Lactose", "OCC1OC(O)C(O)C(O)C1O")
opt.add_mixture_component("PVP",     "C1CN(C(=O)C=C1)C")
opt.add_mixture_component("Starch",  "C(C1C(C(C(C(O1)O)O)O)O)O")

# Independent process factors
opt.add_factor("Temperature", low=25.0, high=50.0)
opt.add_factor("Moisture",    low=0.2,  high=0.8)

results = opt.run_combined_optimization()
print(f"Predicted OCS: {results['predicted_ocs']:.2f}")
print(f"Optimal Recipe: {results['optimal_params']}")
```

---

## Scientific Foundation

### 11-Channel Voxel Field System

| Field Channel     | Physics Kernel                       | Range (A) |
|-------------------|--------------------------------------|-----------|
| Electrostatic     | Debye-Huckel screened Coulomb        | 10.0      |
| Hydrophobic       | Hydrophobicity index convolution     | 6.0       |
| Steric            | Lennard-Jones 12-6 potential         | 5.0       |
| H-Bond Donor      | Directional H-bond geometry          | 4.0       |
| H-Bond Acceptor   | Directional H-bond geometry          | 4.0       |
| Polarizability    | Induced dipole interactions          | 5.0       |
| Reactive Sites    | Arrhenius-weighted reactivity        | 5.0       |
| pH Field          | Henderson-Hasselbalch micro-pH       | 5.0       |
| Ionic Strength    | Local ionic concentration            | 5.0       |
| Lipid Density     | Lipophilicity distribution           | 6.0       |
| Viscosity         | Molecular mobility field             | 5.0       |
| Flexibility       | Tier 4 Elastic deformation field     | 4.0       |
| Nucleophilicity   | Fukui f- (susceptibility to attack)  | 3.0       |
| Electrophilicity  | Fukui f+ (susceptibility to attack)  | 3.0       |
| Water Conserved   | Crystallographic HOH mediation       | 2.8       |
| Water Dynamic     | Predicted occupancy probability      | 2.8       |

### Optimizer Engine (v0.8.0)

| Mode                    | Design              | Model               | Use Case                             |
|-------------------------|---------------------|---------------------|--------------------------------------|
| BBD (Process)           | Box-Behnken         | Quadratic (sklearn) | Optimize pH, Temp, Moisture          |
| Mixture (Composition)   | Simplex-Centroid    | Scheffe Quadratic   | Optimize Filler/Binder/Disintegrant  |
| Combined                | Simplex x Factorial | Scheffe + Process   | Full recipe + environment design     |

**Key Physics Guarantees**:
- **Simplex Normalization**: All mixture proportions automatically sum to 1.0
- **Bootstrap Smoothing**: OCS averaged over 15 Monte Carlo iterations to eliminate voxel noise
- **Concentration Weighting**: All VOI integrals are weighted by `Molecule.concentration`

---

## Validation

| Metric                     | Value |
|----------------------------|-------|
| R2 (VOI vs Experimental)   | 0.74  |
| Classification Accuracy    | 78%   |
| Sensitivity                | 82%   |
| Specificity                | 75%   |
| Combined OCS (Aspirin test)| 87.72 |
| MD Speedup (vs GROMACS)    | ~12,748x|
| Langevin Step Latency      | 1.5 ms|
| Pipeline Latency (5k steps)| 13.5 s|
| VATI delta-G Accuracy      | <2.0 kcal/mol|


*Dataset: 23 API-excipient pairs. Combined optimization test: 3-component tablet (Lactose/PVP/Starch)*
*Dataset: 1CRN.pdb (Crambin) + Aspirin benchmark. Tier 3/4 pipeline: 50 induced-fit frames x 100 Langevin steps.*

---

## Module Structure

```
box/
  key/
    __init__.py       # Package exports
    models.py         # Data structures (Molecule, EnvironmentalConditions)
    core_engine.py    # 16-channel voxel physics engine
    designer.py       # Screening, excipient library, molecule builder
    optimizer.py      # BBD + Simplex + Combined optimizer
    visualizer.py     # FEA-style 3D static visualization
    trajectory_viewer.py # Animated Plotly dynamics viewer (NEW v0.8.1)
    vati.py           # Alchemical Thermodynamic Integration (NEW v0.8.1)
  nibble/
    nibble.h          # 16-channel C headers
    nibble_tier3.c    # Langevin Trajectory Engine (C-native)
    nibble_tier4.c    # Induced Fit / Elastic Deformation (C-native)
    nibble_gaps.c     # Water Networks / Fukui logic (C-native)
  tests/
    verify_tier34.py  # Dynamics & MD-parity benchmark
    verify_viz_vati.py # Visualization & VATI test suite
```

---

## Running Tests

```bash
cd box

# Quick import test
python tests/test_core_physics.py

# Full test suite (v0.8.0)
python tests/test_keybox_v0_8.py
```

---

## Citation

```bibtex
@software{keybox2025,
  author  = {Sharma, Sangeet},
  title   = {KeyBox: Physics-Based Voxel System for API-Excipient Compatibility},
  year    = {2025},
  version = {0.8.0},
  url     = {https://github.com/sangeet01/keybox},
  license = {Apache-2.0}
}
```

---

## Roadmap

### Completed
- [x] 11-channel voxel physics engine
- [x] Multi-scale physics (molecular, process, microstructure, temporal)
- [x] Mechanism detection (VOI: Maillard, Hydrolysis, Oxidation, Acid-Base)
- [x] FEA-style visualization (12 methods)
- [x] Excipient screening
- [x] Box-Behnken Design (BBD) optimizer
- [x] Simplex-Centroid mixture design
- [x] Combined Mixture-Process optimization (v0.8.0)
- [x] Tier 3 Langevin Trajectory (v0.8.1)
- [x] Tier 4 Induced Fit & Elastic Deformation (v0.8.1)
- [x] VATI Thermodynamic Integration (Gap Closure)
- [x] Virtual Water Network (Gap Closure)
- [x] Covalent Trigger (Gap Closure)

### Planned
- [ ] Expand validation to 100+ pairs
- [ ] GPU acceleration
- [ ] Web interface
- [ ] FDA GRAS database integration
- [ ] Ternary heatmap visualization for mixture space

---

## License

**Apache License 2.0 with Commons Clause** - Free for personal and research work. Commercial use requires permission.

See [LICENSE](LICENSE) file for full details.

**Note**: Research software. Not validated for regulatory submissions.

---

**Version**: 0.8.1
**Status**: Stable Release -- Dynamics Edition
**Last Updated**: April 2026
