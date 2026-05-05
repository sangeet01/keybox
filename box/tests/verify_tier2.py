import sys
import os
import numpy as np

# Ensure box directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key.models import Molecule, MoleculeType, EnvironmentalConditions, DosageForm, DosageFormConfig
from key.core_engine import KeyBoxSystem, classify_gfa
from key.designer import create_enhanced_molecule_from_smiles

def test_tier2_physics():
    print("=== TESTING TIER 2 PHYSICS (Polymorph & Metal-Ox) ===")
    
    # 1. Polymorph Conversion: Indomethacin (Class I, Low Tg)
    # SMILES: CC(=O)Oc1ccccc1C(=O)O is Aspirin.
    # Indomethacin: CC1=C(C=C2C(=C1)N(C(=C2CC(=O)O)C)C(=O)C3=CC=C(C=C3)Cl)OC
    indomethacin_smi = "CC1=C(C=C2C(=C1)N(C(=C2CC(=O)O)C)C(=O)C3=CC=C(C=C3)Cl)OC"
    ind_mol = create_enhanced_molecule_from_smiles(indomethacin_smi, "Indomethacin", MoleculeType.SMALL_MOLECULE)
    
    gfa_s, gfa_cl, tg, tg_eff = classify_gfa(indomethacin_smi)
    print(f"\n[TEST] Indomethacin GFA: Class {gfa_cl}, Tg: {tg:.1f}C")
    assert gfa_cl == 'I' or gfa_cl == 'II' # Literature often calls it Class I/II borderline
    
    # Excipient: MgSt (Lubricant, should accelerate)
    mgst_smi = "[Mg+2].[O-]C(=O)CCCCCCCCCCCCCCCCC.[O-]C(=O)CCCCCCCCCCCCCCCCC"
    mgst_mol = create_enhanced_molecule_from_smiles(mgst_smi, "MgSt", MoleculeType.SMALL_MOLECULE)
    
    env = EnvironmentalConditions(temperature=60.0, storage_time=6.0) # 6 months at 60C
    cfg = DosageFormConfig(form_type=DosageForm.ORAL_SOLID)
    
    system = KeyBoxSystem(env_conditions=env, dosage_config=cfg)
    system.add_molecule(ind_mol, is_api=True)
    system.add_molecule(mgst_mol, is_api=False)
    
    analysis = system.analyze_enhanced_formulation()
    metrics = analysis['metrics']
    mechs = analysis['mechanisms']
    
    print(f"Polymorph VOI: {metrics.get('Polymorph_VOI', 0):.3f}")
    poly_found = any("Polymorph" in m for m in mechs)
    assert poly_found, "FAILED: Polymorph conversion risk not detected for Indomethacin"
    print("Polymorph detection: VALID")

    # 2. Metal-Catalysed Oxidation: Ciprofloxacin (Amine) + Trace Metal (Talc proxy)
    # Cipro SMILES (fixed)
    cipro_smi = "C1CC1N2C=C(C(=O)C3=CC(F)=C(C=C32)N4CCNCC4)C(=O)O"
    cipro_mol = create_enhanced_molecule_from_smiles(cipro_smi, "Cipro", MoleculeType.SMALL_MOLECULE)
    
    # Talc proxy (contains [Si] which triggers metal exposure in our model)
    talc_smi = "O=[Si]=O" 
    talc_mol = create_enhanced_molecule_from_smiles(talc_smi, "Talc_Proxy", MoleculeType.SMALL_MOLECULE)
    
    system_ox = KeyBoxSystem(env_conditions=env, dosage_config=cfg)
    system_ox.add_molecule(cipro_mol, is_api=True)
    system_ox.add_molecule(talc_mol, is_api=False)
    
    analysis_ox = system_ox.analyze_enhanced_formulation()
    metrics_ox = analysis_ox['metrics']
    
    print(f"\n--- Cipro + Talc (Metal-Ox) ---")
    print(f"Metal-Ox VOI: {metrics_ox.get('MetalOx_VOI', 0):.3f}")
    ox_found = any("Metal-Catalysed Oxidation" in m for m in analysis_ox['mechanisms'])
    assert ox_found, "FAILED: Metal-Ox risk not detected for Cipro+Talc"
    print("Metal-Ox detection: VALID")

    print("\nTier 2 Physics Verification: SUCCESS")

if __name__ == "__main__":
    test_tier2_physics()
