import sys
import os
import numpy as np

# Ensure box directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key.models import Molecule, MoleculeType, EnvironmentalConditions, DosageForm, DosageFormConfig
from key.core_engine import KeyBoxSystem
from key.designer import create_enhanced_molecule_from_smiles

def test_photodegradation():
    print("=== TESTING PHOTODEGRADATION PHYSICS (Tier 1) ===")
    
    # 1. High Risk API: Nifedipine (Nitroaromatic)
    nifedipine_smi = "CC1=C(NC(=C(C1C2=CC=CC=C2[N+](=O)[O-])C(=O)OC)C)C(=O)OC"
    nif_mol = create_enhanced_molecule_from_smiles(nifedipine_smi, "Nifedipine", MoleculeType.SMALL_MOLECULE)
    
    # Neutral excipient (MCC)
    mcc_smi = "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@@H]([C@@H]([C@H]2O)O)O)CO)O)O)O)O"
    mcc_mol = create_enhanced_molecule_from_smiles(mcc_smi, "MCC", MoleculeType.SMALL_MOLECULE)
    
    env = EnvironmentalConditions(oxygen_level=0.21) # Ambient oxygen
    cfg = DosageFormConfig(form_type=DosageForm.ORAL_SOLID)
    
    system = KeyBoxSystem(env_conditions=env, dosage_config=cfg)
    system.add_molecule(nif_mol, is_api=True)
    system.add_molecule(mcc_mol, is_api=False)
    
    analysis = system.analyze_enhanced_formulation()
    metrics = analysis['metrics']
    mechs = analysis['mechanisms']
    recs = system.get_stability_recommendations()
    
    print(f"\n--- Nifedipine + MCC ---")
    print(f"Photo VOI: {metrics.get('Photo_VOI', 0):.3f}")
    print(f"Mechanisms: {mechs}")
    
    photo_found = any("Photodegradation" in m for m in mechs)
    assert photo_found, "FAILED: Photodegradation risk not detected for Nifedipine"
    
    # Check for specific chromophore in recommendation
    photo_recs = [r for r in recs if "Photodegradation" in r]
    assert len(photo_recs) > 0, "FAILED: No stability recommendation for photodegradation"
    print(f"Recommendation: {photo_recs[0][:120]}...")
    assert "nitroaromatic" in photo_recs[0].lower(), "FAILED: Chromophore detail missing in recommendation"

    # 2. Photosensitizer Effect: TiO2
    tio2_smi = "O=[Ti]=O"
    tio2_mol = create_enhanced_molecule_from_smiles(tio2_smi, "TiO2", MoleculeType.SMALL_MOLECULE)
    
    system_tio2 = KeyBoxSystem(env_conditions=env, dosage_config=cfg)
    system_tio2.add_molecule(nif_mol, is_api=True)
    system_tio2.add_molecule(tio2_mol, is_api=False)
    
    analysis_tio2 = system_tio2.analyze_enhanced_formulation()
    metrics_tio2 = analysis_tio2['metrics']
    
    print(f"\n--- Nifedipine + TiO2 ---")
    print(f"Photo VOI with TiO2: {metrics_tio2.get('Photo_VOI', 0):.3f}")
    
    assert metrics_tio2.get('Photo_VOI', 0) > metrics.get('Photo_VOI', 0), "FAILED: TiO2 should increase photodegradation risk"
    print("Photosensitizer bonus: VALID")

    print("\nPhotodegradation Verification: SUCCESS")

if __name__ == "__main__":
    test_photodegradation()
