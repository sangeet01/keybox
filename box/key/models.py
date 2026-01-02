"""
KeyBox Models: Data Structures & Enums
Separating state from physics logic for cleaner imports and debugging.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

class MoleculeType(Enum):
    SMALL_MOLECULE = "small_molecule"
    BIOLOGIC = "biologic"
    POLYMER = "polymer"
    LIPID = "lipid"
    SURFACTANT = "surfactant"

class DosageForm(Enum):
    ORAL_SOLID = "oral_solid"
    SOLUTION = "solution"
    SUSPENSION = "suspension"
    LIPID_BASED = "lipid_based"
    INJECTABLE = "injectable"
    CONTROLLED_RELEASE = "controlled_release"
    MULTI_PHASE = "multi_phase"

@dataclass
class Molecule:
    """Enhanced molecule representation for all API/excipient types"""
    name: str
    mol_type: MoleculeType
    coords: np.ndarray  # 3D coordinates (N_atoms, 3)
    charges: np.ndarray  # Partial charges
    hydrophobicity: np.ndarray  # Hydrophobicity indices
    vdw_radii: np.ndarray  # Van der Waals radii
    h_donors: np.ndarray  # H-bond donor flags
    h_acceptors: np.ndarray  # H-bond acceptor flags
    polarizability: np.ndarray  # Atomic polarizabilities
    reactive_groups: np.ndarray  # Reactive functional group flags
    acid_groups: np.ndarray = None  # Acidic group flags (e.g. COOH)
    base_groups: np.ndarray = None  # Basic group flags (e.g. NH2)
    amine_groups: np.ndarray = None  # Specifically for Maillard
    carbonyl_groups: np.ndarray = None  # Specifically for Maillard (Sugar carbonyls)
    concentration: float = 1.0  # Molar fraction or concentration
    conformations: Optional[List[np.ndarray]] = None  # Multiple conformations for flexibility
    molecular_weight: float = 0.0
    flexibility: float = 0.0  # Conformational flexibility score
    smiles: Optional[str] = None # Functional group tracking

@dataclass
class EnvironmentalConditions:
    """Enhanced environmental parameters"""
    pH: float = 7.0
    moisture: float = 0.0  # Relative humidity (0-1)
    temperature: float = 25.0  # Celsius
    ionic_strength: float = 0.0  # M
    dielectric_constant: float = 78.5  # Water at 25°C
    storage_time: float = 0.0  # Months
    pressure: float = 1.0  # atm
    oxygen_level: float = 0.21  # Fraction

@dataclass
class DosageFormConfig:
    """Configuration for different dosage forms"""
    form_type: DosageForm
    voxel_resolution: float = 1.0  # Adaptive resolution
    phase_boundaries: List[Tuple[float, float, float]] = field(default_factory=list)
    release_kinetics: bool = False
    multi_layer: bool = False
    solvent_accessible: bool = True
