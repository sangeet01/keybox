import sys
import os
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from key.models import Molecule, MoleculeType, DosageForm, EnvironmentalConditions, DosageFormConfig
from key.engine import KeyBoxSystem

def test_dosage_form_physics():
    print("=== TESTING DOSAGE-FORM SPECIFIC PHYSICS (Tier 3) ===")
    
    # 1. SUSPENSION TEST
    # Mean particle size 10um, Density diff 0.3 g/cm3
    cfg_susp = DosageFormConfig(
        form_type=DosageForm.SUSPENSION,
        particle_diameter_um=10.0,
        particle_density_g_cm3=1.5,
        viscosity_cP=100.0,
        suspending_agent_conc=1.0
    )
    
    system_susp = KeyBoxSystem(voxel_bounds=(20, 20, 20), resolution=1.0, dosage_config=cfg_susp)
    # Add a mock molecule to satisfy tc > 0
    mol = Molecule(
        name="Mock", mol_type=MoleculeType.SMALL_MOLECULE,
        coords=np.array([[10, 10, 10]]), charges=np.array([0.0]), hydrophobicity=np.array([0.0]),
        vdw_radii=np.array([1.5]), h_donors=np.array([0]), h_acceptors=np.array([0]),
        polarizability=np.array([1.0]), reactive_groups=np.array([0]), smiles="C"
    )
    system_susp.add_molecule(mol, is_api=True)
    
    res_susp = system_susp.analyze_enhanced_formulation()
    print(f"Suspension OCS Modifier: {res_susp['metrics'].get('form_ocs_modifier', 0):.2f}")
    print(f"Label: {res_susp['form_physics']['modifier_label']}")
    
    # 2. INJECTABLE TEST
    cfg_inj = DosageFormConfig(
        form_type=DosageForm.INJECTABLE,
        target_osmolarity=300.0
    )
    system_inj = KeyBoxSystem(voxel_bounds=(20, 20, 20), resolution=1.0, dosage_config=cfg_inj)
    # Set concentration to something that gives non-zero osmolarity
    mol_inj = Molecule(
        name="API", mol_type=MoleculeType.SMALL_MOLECULE,
        coords=np.array([[10, 10, 10]]), charges=np.array([0.0]), hydrophobicity=np.array([1.0]), # hydrophobic
        vdw_radii=np.array([1.5]), h_donors=np.array([0]), h_acceptors=np.array([0]),
        polarizability=np.array([1.0]), reactive_groups=np.array([0]), smiles="C",
        concentration=0.1 # 0.1 mol/L -> 100 mOsm/L if i=1
    )
    system_inj.add_molecule(mol_inj, is_api=True)
    
    res_inj = system_inj.analyze_enhanced_formulation()
    print(f"Injectable OCS Modifier: {res_inj['metrics'].get('form_ocs_modifier', 0):.2f}")
    print(f"Label: {res_inj['form_physics']['modifier_label']}")
    print(f"Extra Metrics: {res_inj['form_physics']['extra_metrics']}")

    print("\nDosage-Form Physics: SUCCESS")

if __name__ == "__main__":
    test_dosage_form_physics()
