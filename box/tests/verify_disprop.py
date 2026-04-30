import sys
import os
import numpy as np

# Ensure box directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key.models import Molecule, MoleculeType, EnvironmentalConditions, DosageForm, DosageFormConfig
from key.core_engine import KeyBoxSystem, classify_salt
from key.designer import create_enhanced_molecule_from_smiles

def test_disproportionation():
    print("=== TESTING DISPROPORTIONATION PHYSICS (Tier 1) ===")
    
    # Ciprofloxacin HCl
    cipro_hcl_smi = "C1CC1N2C=C(C(=O)C3=CC(F)=C(C=C32)N4CCNCC4)C(=O)O.Cl"
    salt_info = classify_salt(cipro_hcl_smi)
    print(f"\n[TEST] Salt Classification: {salt_info['salt_type']}")
    assert salt_info['is_salt'] == True
    # Cipro is a basic API salt because of the HCl
    assert salt_info['salt_type'] == 'basic_salt'
    print("Salt classification: VALID")

    # 2. High Risk Pair: Cipro HCl + Magnesium Stearate
    # MgSt is alkaline, should trigger disproportionation
    mgst_smi = "[Mg+2].[O-]C(=O)CCCCCCCCCCCCCCCCC.[O-]C(=O)CCCCCCCCCCCCCCCCC"
    cipro_mol = create_enhanced_molecule_from_smiles(cipro_hcl_smi, "Cipro_HCl", MoleculeType.SMALL_MOLECULE)
    mgst_mol = create_enhanced_molecule_from_smiles(mgst_smi, "MgSt", MoleculeType.SMALL_MOLECULE)
    
    env = EnvironmentalConditions(temperature=40.0, moisture=0.75, pH=7.0)
    cfg = DosageFormConfig(form_type=DosageForm.ORAL_SOLID)
    
    system = KeyBoxSystem(env_conditions=env, dosage_config=cfg)
    system.add_molecule(cipro_mol, is_api=True)
    system.add_molecule(mgst_mol, is_api=False)
    
    analysis = system.analyze_enhanced_formulation()
    metrics = analysis['metrics']
    mechs = analysis['mechanisms']
    recs = system.get_stability_recommendations()
    
    print(f"\n--- Cipro HCl + MgSt (40C/75%RH) ---")
    print(f"Disprop VOI: {metrics.get('Disproportionation_VOI', 0):.3f}")
    print(f"OCS: {metrics['OCS']:.1f}")
    print(f"Mechanisms: {mechs}")
    
    disprop_found = any("Disproportionation" in m for m in mechs)
    assert disprop_found, "FAILED: Disproportionation risk not detected for Cipro+MgSt"
    assert metrics['OCS'] < 90, f"FAILED: OCS should be penalized, got {metrics['OCS']}"
    
    critical_recs = [r for r in recs if "[CRITICAL] Disproportionation" in r or "[WARNING] Disproportionation" in r]
    assert len(critical_recs) > 0, "FAILED: No stability recommendation for disproportionation"
    print(f"Recommendation: {critical_recs[0][:100]}...")
    print("High-risk detection: VALID")

    # 3. Low Risk Pair: Cipro HCl + MCC (neutral)
    mcc_smi = "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@@H]([C@@H]([C@H]2O)O)O)CO)O)O)O)O" # Cellulose fragment
    mcc_mol = create_enhanced_molecule_from_smiles(mcc_smi, "MCC", MoleculeType.SMALL_MOLECULE)
    
    system_mcc = KeyBoxSystem(env_conditions=env, dosage_config=cfg)
    system_mcc.add_molecule(cipro_mol, is_api=True)
    system_mcc.add_molecule(mcc_mol, is_api=False)
    
    analysis_mcc = system_mcc.analyze_enhanced_formulation()
    metrics_mcc = analysis_mcc['metrics']
    
    print(f"\n--- Cipro HCl + MCC (40C/75%RH) ---")
    print(f"Disprop VOI: {metrics_mcc.get('Disproportionation_VOI', 0):.3f}")
    print(f"OCS: {metrics_mcc['OCS']:.1f}")
    
    assert metrics_mcc.get('Disproportionation_VOI', 0) < metrics.get('Disproportionation_VOI', 0), "FAILED: MCC should have lower disprop risk than MgSt"
    print("Low-risk distinction: VALID")

    print("\nDisproportionation Verification: SUCCESS")

if __name__ == "__main__":
    test_disproportionation()
