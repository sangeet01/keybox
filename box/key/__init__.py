"""
KeyBox: Physics-Based Voxel System for API-Excipient Compatibility Prediction

Open-source computational platform for predicting pharmaceutical formulation
incompatibilities using 3D voxel field theory.

Author: Sangeet Sharma
Version: 0.9.3
License: Apache 2.0
"""

__version__ = "0.9.3"
__author__ = "Sangeet Sharma"
__license__ = "Apache 2.0"

# Core data structures
from .models import (
    MoleculeType,
    DosageForm,
    Molecule,
    EnvironmentalConditions,
    DosageFormConfig,
    DosageFormResult,
)

# Modular Physics Engine
from .voxel_grid import EnhancedVoxelGrid
from .physics import (
    MultiComponentSystem,
    ProcessPhysics,
    MicrostructurePhysics,
    TemporalPhysics,
    compute_hansen_chi,
    _estimate_hansen_params,
)
from .mechanisms import (
    compute_metal_ox_voi,
    compute_polymorph_voi,
    compute_photo_voi,
    compute_nucl_add_voi,
    classify_gfa,
    compute_transester_voi,
    compute_complexation_voi,
    compute_adsorption_voi,
    compute_eutectic_voi,
    classify_salt,
    _compute_disprop_voi
)
from .core_engine import KeyBoxSystem
from .nibble_bridge import NibbleEngine

# Legacy entry point (re-exports)
from . import core_engine

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

# Optimization engine (v0.8.0+)
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
    "DosageFormResult",

    # Physics engine
    "EnhancedVoxelGrid",
    "MultiComponentSystem",
    "ProcessPhysics",
    "MicrostructurePhysics",
    "TemporalPhysics",
    "KeyBoxSystem",
    "compute_hansen_chi",
    "_estimate_hansen_params",
    "compute_metal_ox_voi",
    "compute_polymorph_voi",
    "compute_photo_voi",
    "compute_nucl_add_voi",
    "classify_salt",
    "_compute_disprop_voi",
    "compute_transester_voi",
    "compute_complexation_voi",
    "compute_adsorption_voi",
    "compute_eutectic_voi",

    # Visualization
    "VoxelVisualizer",
    "quick_visualize",

    # Designer
    "create_enhanced_molecule_from_smiles",
    "screen_enhanced_excipients_for_api",
    "analyze_enhanced_formulation_with_visualization",
    "ENHANCED_EXCIPIENT_LIBRARY",
    "DesignerOptimizer",

    # Optimizer
    "FormulationOptimizer",
    "KeyBoxOptimizer",
    "generate_box_behnken",
    "NibbleEngine",
]
