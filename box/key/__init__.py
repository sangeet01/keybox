"""
KeyBox: Physics-Based Voxel System for API-Excipient Compatibility Prediction

Open-source computational platform for predicting pharmaceutical formulation
incompatibilities using 3D voxel field theory.

Author: Sangeet Sharma
Version: 0.8.0
License: Apache 2.0
"""

__version__ = "0.8.0"
__author__ = "Sangeet Sharma"
__license__ = "Apache 2.0"

# Core data structures
from .models import (
    MoleculeType,
    DosageForm,
    Molecule,
    EnvironmentalConditions,
    DosageFormConfig
)

# Physics engine
from .core_engine import (
    EnhancedVoxelGrid,
    MultiComponentSystem,
    ProcessPhysics,
    MicrostructurePhysics,
    TemporalPhysics,
    KeyBoxSystem
)

# Visualization
from .visualizer import (
    VoxelVisualizer,
    quick_visualize
)

# High-level screening and design
from .designer import (
    create_enhanced_molecule_from_smiles,
    screen_enhanced_excipients_for_api,
    analyze_enhanced_formulation_with_visualization,
    ENHANCED_EXCIPIENT_LIBRARY,
    DesignerOptimizer
)

# Optimization engine (v0.8.0)
from .optimizer import (
    FormulationOptimizer,
    KeyBoxOptimizer,
    generate_box_behnken
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",

    # Data structures
    "MoleculeType",
    "DosageForm",
    "Molecule",
    "EnvironmentalConditions",
    "DosageFormConfig",

    # Physics engine
    "EnhancedVoxelGrid",
    "MultiComponentSystem",
    "ProcessPhysics",
    "MicrostructurePhysics",
    "TemporalPhysics",
    "KeyBoxSystem",

    # Visualization
    "VoxelVisualizer",
    "quick_visualize",

    # Designer
    "create_enhanced_molecule_from_smiles",
    "screen_enhanced_excipients_for_api",
    "analyze_enhanced_formulation_with_visualization",
    "ENHANCED_EXCIPIENT_LIBRARY",
    "DesignerOptimizer",

    # Optimizer (v0.8.0)
    "FormulationOptimizer",
    "KeyBoxOptimizer",
    "generate_box_behnken",
]
