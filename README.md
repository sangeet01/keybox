# KeyBox: Physics-Based Voxel System for API-Excipient Compatibility

**Open-source computational platform for predicting pharmaceutical formulation incompatibilities using 3D voxel field theory**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status: Stable](https://img.shields.io/badge/status-stable-brightgreen.svg)]()

---

## 🎯 Overview

KeyBox applies **engineering finite element analysis (FEA) principles to molecular chemistry**, creating a voxel-based field system that predicts API-excipient interactions through first-principles physics.

**Key Innovation**: Treats pharmaceutical formulations as 3D field problems with 11 simultaneous interaction channels (electrostatic, hydrophobic, steric, H-bond, etc.), enabling mechanistic prediction of incompatibilities before wet lab testing.

### Why KeyBox?

- **Physics-First**: Debye-Hückel, Lennard-Jones, Flory-Huggins, Griffith fracture mechanics
- **Strictly Chemical**: No mock data—requires valid SMILES and RDKit
- **Multi-Scale**: Molecular → Process → Microstructure → Time
- **Quantitative**: Volume of Incompatibility (VOI) metrics with 95% CI
- **Complete Workflow**: Predict → Diagnose → Visualize

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

### Basic Usage

```python
from box import KeyBoxSystem, EnvironmentalConditions

# Initialize system
system = KeyBoxSystem(
    voxel_bounds=(20.0, 20.0, 20.0),
    resolution=1.0,
    random_state=42
)

# Add molecules (SMILES input)
system.add_molecule_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", is_api=True)
system.add_molecule_from_smiles("C(CO)O", "Ethylene_Glycol", is_api=False)

# Set environmental conditions
system.env_conditions = EnvironmentalConditions(
    pH=7.0,
    moisture=0.6,
    temperature=40.0,
    storage_time=6.0
)

# Analyze formulation
results = system.analyze_enhanced_formulation(include_voi_uncertainty=True)

print(f"Compatibility: {results['classification']}")
print(f"Mechanisms: {results['mechanisms']}")
print(f"VOI Total: {results['voi_total']:.2f} ± {results['voi_uncertainty']['total']['std']:.2f}")
```

### Visualization

```python
from box.visualizer import VoxelVisualizer

viz = VoxelVisualizer(system)
viz.plot_field_slice('electrostatic', slice_axis='z', slice_index=10)
viz.plot_interaction_heatmap()
viz.plot_vector_field('steric')
```

---

## 🔬 Scientific Foundation

### 11-Channel Voxel Field System

| Field Channel | Physics Kernel | Range (Å) |
|--------------|----------------|-----------|
| **Electrostatic** | Debye-Hückel screened Coulomb | 10.0 |
| **Hydrophobic** | Hydrophobicity index convolution | 6.0 |
| **Steric** | Lennard-Jones 12-6 potential | 5.0 |
| **H-Bond Donor** | Directional H-bond geometry | 4.0 |
| **H-Bond Acceptor** | Directional H-bond geometry | 4.0 |
| **Polarizability** | Induced dipole interactions | 5.0 |
| **Reactive Sites** | Arrhenius-weighted reactivity | 5.0 |
| **pH Field** | Henderson-Hasselbalch micro-pH | 5.0 |
| **Ionic Strength** | Local ionic concentration | 5.0 |
| **Lipid Density** | Lipophilicity distribution | 6.0 |
| **Viscosity** | Molecular mobility field | 5.0 |

### Advanced Physics

- **Multi-Component**: Flory-Huggins χ-parameter, ternary interactions, competitive adsorption
- **Process Physics**: Griffith fracture mechanics, percolation theory, Heckel compaction
- **Microstructure**: Cahn-Hilliard phase-field, interface energy, particle packing
- **Temporal**: Fick's diffusion, Arrhenius kinetics, nucleation theory, WLF time-temperature

---

## 📊 Validation

| Metric | Value |
|--------|-------|
| **R² (VOI vs. Experimental)** | 0.74 |
| **Classification Accuracy** | 78% |
| **Sensitivity** | 82% |
| **Specificity** | 75% |

*Dataset: 23 API-excipient pairs*

---

## 📁 Module Structure

```
box/
├── __init__.py          # Package exports
├── models.py            # Data structures (Molecule, EnvironmentalConditions)
├── core_engine.py       # 11-channel voxel physics engine
├── visualizer.py        # FEA-style 3D visualization
└── designer.py          # Formulation screening
```

### Core Modules

**models.py** - Data Structures
- `Molecule`: 15 field channels per molecule
- `EnvironmentalConditions`: pH, moisture, temperature, storage time
- `DosageForm`: 7 dosage form types

**core_engine.py** - Physics Engine
- `EnhancedVoxelGrid`: 11-channel 3D field system
- `MultiComponentSystem`: Flory-Huggins, reaction networks
- `ProcessPhysics`: Griffith fracture, percolation
- `MicrostructurePhysics`: Cahn-Hilliard phase-field
- `TemporalPhysics`: Nucleation theory, diffusion
- `KeyBoxSystem`: Main orchestrator

**visualizer.py** - Visualization
- Field slices, 3D isosurfaces, interaction heatmaps
- Vector fields, mechanism overlays
- Cross-sections, phase separation animations
- 12 visualization methods total

**designer.py** - High-Level Screening
- `create_enhanced_molecule_from_smiles`: RDKit-based builder
- `ENHANCED_EXCIPIENT_LIBRARY`: 11 validated excipients
- `screen_enhanced_excipients_for_api`: Automated screening

---

## 🎓 Citation

```bibtex
@software{keybox2025,
  author = {Sharma, Sangeet},
  title = {KeyBox: Physics-Based Voxel System for API-Excipient Compatibility},
  year = {2025},
  version = {0.6.0},
  url = {https://github.com/sangeet01/keybox},
  license = {Apache-2.0}
}
```



---

## 🚧 Roadmap

### Completed
- [x] 11-channel voxel physics engine
- [x] Multi-scale physics (molecular, process, microstructure, temporal)
- [x] Mechanism detection (10 types)
- [x] FEA-style visualization (12 methods)
- [x] Excipient screening

### Planned
- [ ] Expand validation to 100+ pairs
- [ ] GPU acceleration
- [ ] Web interface
- [ ] FDA GRAS database integration

---

## 👥 Contributing

Contributions welcome! Areas of interest:
- Experimental validation data
- New physics modules
- Performance optimization
- Dosage form extensions

```bash
# Development setup
git clone https://github.com/yourusername/keybox.git
cd keybox
pip install -e .
```

---

## 📄 License

**Apache License 2.0**

Free for academic and commercial use. See [LICENSE](LICENSE) for details.

**Note**: This is research software. Not validated for regulatory submissions.

---

## 🔗 Links

- **Documentation**: [docs.keybox.io](https://docs.keybox.io) (coming soon)
- **Issues**: [GitHub Issues](https://github.com/sangeet01/keybox/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sangeet01/keybox/discussions)

---

## 📧 Contact

**Sangeet Sharma**  
Pharmaceutical Sciences | Computational Chemistry  
Linkedin:   
GitHub: [@sangeet0](https://github.com/sangeet01)

---

## 🙏 Acknowledgments

- **RDKit Community**: Cheminformatics toolkit
- **NumPy/SciPy**: Scientific computing
- **Matplotlib/Seaborn**: Visualization
- **Pharmaceutical Sciences Community**: Domain expertise

---

**Version**: 0.6.0  
**Status**: Stable Release  
**Last Updated**: January 2025
