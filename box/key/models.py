"""
KeyBox Models: Data Structures & Enums

v0.9.1 — Dosage-form agnostic (Layer 1+2):
  DosageFormConfig extended with form-specific parameters:
    - coating_thickness_um  : CONTROLLED_RELEASE membrane thickness
    - coating_diffusivity   : CONTROLLED_RELEASE membrane D (cm²/s)
    - lipid_fraction        : LIPID_BASED oil phase volume fraction
    - dispersed_phase_volume: MULTI_PHASE dispersed phase fraction
    - droplet_diameter_um   : MULTI_PHASE initial droplet size
    - target_osmolarity     : INJECTABLE target mOsm/kg
    - particle_diameter_um  : SUSPENSION mean particle diameter
  EnvironmentalConditions already had ionic_strength and dielectric_constant —
  these are now actively used by SOLUTION and INJECTABLE physics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class MoleculeType(Enum):
    SMALL_MOLECULE = "small_molecule"
    BIOLOGIC       = "biologic"
    POLYMER        = "polymer"
    LIPID          = "lipid"
    SURFACTANT     = "surfactant"


class DosageForm(Enum):
    ORAL_SOLID        = "oral_solid"
    SOLUTION          = "solution"
    SUSPENSION        = "suspension"
    LIPID_BASED       = "lipid_based"
    INJECTABLE        = "injectable"
    CONTROLLED_RELEASE= "controlled_release"
    MULTI_PHASE       = "multi_phase"


@dataclass
class Molecule:
    """Enhanced molecule representation for all API/excipient types"""
    name:              str
    mol_type:          MoleculeType
    coords:            np.ndarray   # 3D coordinates (N_atoms, 3)
    charges:           np.ndarray   # Partial charges
    hydrophobicity:    np.ndarray   # Hydrophobicity indices
    vdw_radii:         np.ndarray   # Van der Waals radii
    h_donors:          np.ndarray   # H-bond donor flags
    h_acceptors:       np.ndarray   # H-bond acceptor flags
    polarizability:    np.ndarray   # Atomic polarizabilities
    reactive_groups:   np.ndarray   # Reactive functional group flags
    acid_groups:       np.ndarray   = None
    base_groups:       np.ndarray   = None
    amine_groups:      np.ndarray   = None
    carbonyl_groups:   np.ndarray   = None
    oxidizable_groups: np.ndarray   = None
    oxidizing_groups:  np.ndarray   = None
    concentration:     float        = 1.0
    # Tautomer fields (v0.9.0)
    conformations:         Optional[List[np.ndarray]] = None
    taut_weights:          Optional[List[float]]      = None
    taut_charges:          Optional[List[np.ndarray]] = None
    taut_h_donors:         Optional[List[np.ndarray]] = None
    taut_h_acceptors:      Optional[List[np.ndarray]] = None
    taut_amine_groups:     Optional[List[np.ndarray]] = None
    taut_carbonyl_groups:  Optional[List[np.ndarray]] = None
    taut_acid_groups:      Optional[List[np.ndarray]] = None
    taut_base_groups:      Optional[List[np.ndarray]] = None
    molecular_weight:  float        = 0.0
    flexibility:       float        = 0.0
    smiles:            Optional[str]= None


@dataclass
class EnvironmentalConditions:
    """Enhanced environmental parameters"""
    pH:                float = 7.0
    moisture:          float = 0.0    # Relative humidity (0–1)
    temperature:       float = 25.0   # Celsius
    ionic_strength:    float = 0.0    # mol/L (actively used by SOLUTION/INJECTABLE)
    dielectric_constant: float = 78.5 # Relative permittivity (water=78.5, lipid~3)
    storage_time:      float = 0.0    # Months
    pressure:          float = 1.0    # atm
    oxygen_level:      float = 0.21   # Fraction


@dataclass
class DosageFormConfig:
    """
    Configuration for different dosage forms.

    v0.9.1 — form-specific parameters added:
      CONTROLLED_RELEASE : coating_thickness_um, coating_diffusivity
      LIPID_BASED        : lipid_fraction
      MULTI_PHASE        : dispersed_phase_volume, droplet_diameter_um
      INJECTABLE         : target_osmolarity
      SUSPENSION         : particle_diameter_um
    All parameters are optional with sensible defaults so existing code
    that instantiates DosageFormConfig(DosageForm.ORAL_SOLID) is unchanged.
    """
    form_type:             DosageForm
    voxel_resolution:      float = 1.0
    phase_boundaries:      List[Tuple[float, float, float]] = field(default_factory=list)
    release_kinetics:      bool  = False
    multi_layer:           bool  = False
    solvent_accessible:    bool  = True

    # CONTROLLED_RELEASE parameters
    # Ref: Siepmann & Göpferich, Adv. Drug Deliv. Rev. 2001
    coating_thickness_um:  float = 50.0    # µm membrane thickness
    coating_diffusivity:   float = 1e-10   # cm²/s diffusivity through coating

    # LIPID_BASED parameters
    # Ref: Pouton, Eur. J. Pharm. Sci. 2006
    lipid_fraction:        float = 0.6     # oil phase volume fraction (0–1)

    # MULTI_PHASE parameters
    # Ref: McClements, Food Emulsions 2015
    dispersed_phase_volume: float = 0.3    # dispersed phase volume fraction
    droplet_diameter_um:   float = 1.0     # initial droplet diameter µm

    # INJECTABLE parameters
    # Ref: USP <1> Injections
    target_osmolarity:     float = 300.0   # mOsm/kg (isotonic = 285–310)

    # SUSPENSION parameters
    # Ref: Müller & Böhm, Pharm. Ind. 1997
    particle_diameter_um:  float = 10.0    # mean particle diameter µm



@dataclass
class DosageFormResult:
    """
    Structured result from a dosage-form-specific physics analysis.
    Returned by KeyBoxSystem._run_form_physics() and embedded in
    analyze_enhanced_formulation() output under key 'form_physics'.
    """
    form_type:          str
    ocs_modifier:       float        # Additive OCS adjustment (can be negative)
    modifier_label:     str          # Human-readable reason
    extra_metrics:      dict = field(default_factory=dict)
    warnings:           List[str]    = field(default_factory=list)
    is_stub:            bool         = False  # True for SUSPENSION/INJECTABLE
