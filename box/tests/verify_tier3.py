import sys
import os
import numpy as np
from rdkit import Chem

# Add parent directory to path to import key
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from key.models import (
    MoleculeType, DosageForm, Molecule, 
    EnvironmentalConditions, DosageFormConfig
)
from key.engine import KeyBoxSystem

def test_tier3_physics():
    print("=== TESTING TIER 3 PHYSICS (Suspension & Injectable) ===\n")
    
    # 1. Test Suspension Physics
    print("--- Testing Suspension Physics ---")
    # API: Indomethacin (small molecule)
    smi_indo = "CC1=C(C=C2C(=C1)N(C(=O)C3=CC=C(C=C3)Cl)C(=C2CC(=O)O)C)OC"
    mol_indo = Chem.MolFromSmiles(smi_indo)
    
    api_indo = Molecule(
        name="Indomethacin",
        mol_type=MoleculeType.SMALL_MOLECULE,
        coords=np.random.randn(mol_indo.GetNumAtoms(), 3),
        charges=np.zeros(mol_indo.GetNumAtoms()),
        hydrophobicity=np.random.randn(mol_indo.GetNumAtoms()),
        vdw_radii=np.ones(mol_indo.GetNumAtoms()) * 1.5,
        h_donors=np.zeros(mol_indo.GetNumAtoms()),
        h_acceptors=np.zeros(mol_indo.GetNumAtoms()),
        polarizability=np.ones(mol_indo.GetNumAtoms()),
        reactive_groups=np.zeros(mol_indo.GetNumAtoms()),
        smiles=smi_indo
    )
    
    # Suspension Config
    cfg_susp = DosageFormConfig(
        form_type=DosageForm.SUSPENSION,
        particle_diameter_um=10.0
    )
    
    system = KeyBoxSystem(dosage_config=cfg_susp)
    system.add_molecule(api_indo, is_api=True)
    
    result = system.analyze_enhanced_formulation()
    form_res = result['form_physics']
    
    print(f"Form: {form_res['form_type']}")
    print(f"Modifier: {form_res['modifier_label']}")
    print(f"Sedimentation Velocity: {form_res['extra_metrics']['sedimentation_velocity_um_s']:.2f} um/s")
    
    assert form_res['form_type'] == "suspension"
    assert 'sedimentation_velocity_um_s' in form_res['extra_metrics']
    print("Suspension Physics: SUCCESS\n")
    
    # 2. Test Injectable Physics
    print("--- Testing Injectable Physics ---")
    # API: Insulin-like peptide (proxy)
    smi_pep = "CC(C)CC(C(=O)NC(CC1=CC=CC=C1)C(=O)O)N"
    mol_pep = Chem.MolFromSmiles(smi_pep)
    
    api_pep = Molecule(
        name="Peptide",
        mol_type=MoleculeType.BIOLOGIC,
        coords=np.random.randn(mol_pep.GetNumAtoms(), 3),
        charges=np.zeros(mol_pep.GetNumAtoms()),
        hydrophobicity=np.random.randn(mol_pep.GetNumAtoms()),
        vdw_radii=np.ones(mol_pep.GetNumAtoms()) * 1.5,
        h_donors=np.zeros(mol_pep.GetNumAtoms()),
        h_acceptors=np.zeros(mol_pep.GetNumAtoms()),
        polarizability=np.ones(mol_pep.GetNumAtoms()),
        reactive_groups=np.zeros(mol_pep.GetNumAtoms()),
        smiles=smi_pep,
        concentration=0.3 # 300 mM approx
    )
    
    # Injectable Config
    cfg_inj = DosageFormConfig(
        form_type=DosageForm.INJECTABLE,
        target_osmolarity=300.0
    )
    
    system_inj = KeyBoxSystem(dosage_config=cfg_inj)
    system_inj.add_molecule(api_pep, is_api=True)
    
    result_inj = system_inj.analyze_enhanced_formulation()
    form_res_inj = result_inj['form_physics']
    
    print(f"Form: {form_res_inj['form_type']}")
    print(f"Modifier: {form_res_inj['modifier_label']}")
    print(f"Calculated Osmolarity: {form_res_inj['extra_metrics']['calculated_osmolarity']:.1f} mOsm/L")
    print(f"Aggregation Propensity: {form_res_inj['extra_metrics']['aggregation_propensity']:.2f}")
    
    assert form_res_inj['form_type'] == "injectable"
    assert 'calculated_osmolarity' in form_res_inj['extra_metrics']
    print("Injectable Physics: SUCCESS\n")

if __name__ == "__main__":
    try:
        test_tier3_physics()
        print("Tier 3 Physics Verification: SUCCESS")
    except Exception as e:
        print(f"Tier 3 Physics Verification: FAILED - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
