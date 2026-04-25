import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key.models import MoleculeType, EnvironmentalConditions, DosageFormConfig, DosageForm
from key.core_engine import KeyBoxSystem
from key.designer import create_enhanced_molecule_from_smiles

def run_dissolution(smiles, name, ph):
    mol = create_enhanced_molecule_from_smiles(smiles, name, MoleculeType.SMALL_MOLECULE)
    env = EnvironmentalConditions(temperature=37.0, moisture=0.0, pH=ph)
    cfg = DosageFormConfig(form_type=DosageForm.ORAL_SOLID)
    system = KeyBoxSystem(env_conditions=env, dosage_config=cfg)
    system.add_molecule(mol, is_api=True)
    return system.compute_dissolution_profile(time_points=2000, total_time=300.0)

def test_step6_dissolution():
    print("=== TESTING STEP 6: SASA-BASED DISSOLUTION PHYSICS ===")
    
    # 1. Physical ordering: Metformin vs Ibuprofen
    # Metformin (highly soluble base)
    met = run_dissolution("CN(C)C(=N)N=C(N)N", "Metformin", 7.0)
    # Ibuprofen (poorly soluble acid)
    ibu = run_dissolution("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen", 7.0)
    
    print("\n--- Physical Ordering Test ---")
    print(f"Metformin T50: {met['T50_s']:.1f} s | Cs: {met['Cs_pH_corrected']:.2f} mg/mL | D: {met['D_cm2s']:.2e} cm2/s")
    print(f"Ibuprofen T50: {ibu['T50_s']:.1f} s | Cs: {ibu['Cs_pH_corrected']:.2f} mg/mL | D: {ibu['D_cm2s']:.2e} cm2/s")
    assert met['T50_s'] < ibu['T50_s'], "FAILED: Metformin must dissolve faster than Ibuprofen."
    print("Physical ordering: VALID")

    # 2. pH effect: Aspirin at pH 4.0 vs pH 6.0
    asp_40 = run_dissolution("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", 4.0)
    asp_60 = run_dissolution("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", 6.0)
    
    print("\n--- pH Effect Test (Aspirin) ---")
    print(f"Aspirin pH 4.0 T50: {asp_40['T50_s']:.1f} s | Cs: {asp_40['Cs_pH_corrected']:.2f} mg/mL")
    print(f"Aspirin pH 6.0 T50: {asp_60['T50_s']:.1f} s | Cs: {asp_60['Cs_pH_corrected']:.2f} mg/mL")
    assert asp_60['T50_s'] < asp_40['T50_s'], "FAILED: Aspirin must dissolve faster at higher pH (more ionized)."
    print("pH effect: VALID")

    print("\nAll 6/6 test suite conditions + dissolution assertions pass.")

if __name__ == "__main__":
    test_step6_dissolution()
