import sys
import os
import numpy as np

# Ensure box directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key.models import Molecule, MoleculeType, EnvironmentalConditions, DosageForm, DosageFormConfig
from key.core_engine import KeyBoxSystem
from key.designer import create_enhanced_molecule_from_smiles

def test_step5_physics():
    print("\n=== TESTING STEP 5: MORRIS GSA, RECOMMENDATIONS, TOPOLOGY ===")
    
    # Aspirin
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    api = create_enhanced_molecule_from_smiles(aspirin_smiles, "Aspirin", MoleculeType.SMALL_MOLECULE)
    
    # PVP (Polyvinylpyrrolidone) - simplified repeating unit
    pvp_smiles = "CC(C1=CC=CC=C1)N2CCCC2=O"
    exc = create_enhanced_molecule_from_smiles(pvp_smiles, "PVP", MoleculeType.POLYMER)
    
    env = EnvironmentalConditions(temperature=40.0, moisture=0.75, pH=4.5)
    
    # CONTROLLED_RELEASE Form
    config = DosageFormConfig(
        form_type=DosageForm.CONTROLLED_RELEASE,
        coating_thickness_um=50.0,
        coating_diffusivity=1e-8
    )
    system = KeyBoxSystem(env_conditions=env, dosage_config=config)
    
    # Ensure properties are populated
    system.add_molecule(api)
    system.add_molecule(exc)
    
    res = system.analyze_enhanced_formulation()
    
    print(f"\n[BASELINE] OCS: {res['metrics']['OCS']:.2f}")
    
    # 1. Global Sensitivity Analysis (Morris)
    print("\n--- Running Global Sensitivity Analysis (Morris) ---")
    gsa = system.perform_global_sensitivity_analysis(samples=10)
    print(f"Primary Risk: {gsa['primary_risk']}")
    print(f"Ranking: {gsa['ranking']}")
    for lever in gsa['ranking']:
        print(f"  {lever:15}: mu*={gsa['mu_star'][lever]:.3f}, sigma={gsa['sigma'][lever]:.3f}, flags={gsa['interaction_flags'][lever]}")

    # 2. Stability Recommendations
    print("\n--- Stability Recommendations ---")
    recs = system.get_stability_recommendations()
    for r in recs:
        print(f" - {r}")

    # 3. Topology Optimization
    print("\n--- Running Tablet Topology Optimization ---")
    topo = system.optimize_tablet_topology()
    print(f"Best Architecture: {topo['best_architecture']}")
    print(f"Expected Improvement: {topo['improvement']:.3f} points")
    print("\nAll Architectures:")
    for name, r in topo['all_architectures'].items():
        print(f"  {name:15}: Score={r['score']:.3f}, Retained={r['mean_retained_24h']:.3f}, Released={r['mean_released_1h']:.3f}")

if __name__ == "__main__":
    test_step5_physics()
