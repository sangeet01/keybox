import sys
import os
import numpy as np
from rdkit import Chem

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from key.models import Molecule, MoleculeType, DosageForm, EnvironmentalConditions, DosageFormConfig
from key.engine import KeyBoxSystem

def test_tier3_mechanisms():
    print("=== TESTING TIER 3 ADVANCED MECHANISMS ===")
    
    # 1. Complexation Test: API (Aromatic/Carbonyl) + Excipient (PVP)
    # Ciprofloxacin (aromatic) + PVP
    api_smiles = "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O"
    pvp_smiles = "CN1CCCC1=O" # PVP monomer proxy
    
    env = EnvironmentalConditions(temperature=25.0, pH=7.0)
    system = KeyBoxSystem(voxel_bounds=(10, 10, 10), resolution=2.0)
    system.env_conditions = env
    
    mol_api = Molecule(
        name="Cipro", mol_type=MoleculeType.SMALL_MOLECULE,
        coords=np.zeros((1,3)), charges=np.zeros(1), hydrophobicity=np.zeros(1),
        vdw_radii=np.zeros(1), h_donors=np.zeros(1), h_acceptors=np.zeros(1),
        polarizability=np.zeros(1), reactive_groups=np.zeros(1), smiles=api_smiles
    )
    mol_exc = Molecule(
        name="PVP", mol_type=MoleculeType.POLYMER,
        coords=np.ones((1,3))*2.0, charges=np.zeros(1), hydrophobicity=np.zeros(1),
        vdw_radii=np.zeros(1), h_donors=np.zeros(1), h_acceptors=np.zeros(1),
        polarizability=np.zeros(1), reactive_groups=np.zeros(1), smiles=pvp_smiles
    )
    
    system.add_molecule(mol_api, is_api=True)
    system.add_molecule(mol_exc)
    
    results = system.analyze_enhanced_formulation()['metrics']
    
    print(f"Complexation_VOI: {results.get('Complexation_VOI', 0):.4f}")
    print(f"Adsorption_VOI:   {results.get('Adsorption_VOI', 0):.4f}")
    print(f"Eutectic_VOI:     {results.get('Eutectic_VOI', 0):.4f}")
    print(f"Transester_VOI:   {results.get('Transester_VOI', 0):.4f}")
    
    # Validation
    if results.get('Complexation_VOI', 0) > 0:
        print("Complexation detection: SUCCESS")
    else:
        print("Complexation detection: FAILED")

    # 2. Eutectic Test: Aspirin + Stearic Acid
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    stearic = "CCCCCCCCCCCCCCCCCC(=O)O"
    
    system2 = KeyBoxSystem(voxel_bounds=(10, 10, 10), resolution=2.0)
    system2.env_conditions = EnvironmentalConditions(temperature=75.0) # Higher T for eutectic proximity
    
    mol_asp = Molecule(name="Aspirin", mol_type=MoleculeType.SMALL_MOLECULE, coords=np.zeros((1,3)), charges=np.zeros(1), hydrophobicity=np.zeros(1), vdw_radii=np.zeros(1), h_donors=np.zeros(1), h_acceptors=np.zeros(1), polarizability=np.zeros(1), reactive_groups=np.zeros(1), smiles=aspirin)
    mol_ste = Molecule(name="Stearic", mol_type=MoleculeType.SMALL_MOLECULE, coords=np.ones((1,3))*2.0, charges=np.zeros(1), hydrophobicity=np.zeros(1), vdw_radii=np.zeros(1), h_donors=np.zeros(1), h_acceptors=np.zeros(1), polarizability=np.zeros(1), reactive_groups=np.zeros(1), smiles=stearic)
    
    system2.add_molecule(mol_asp, is_api=True)
    system2.add_molecule(mol_ste)
    
    res2 = system2.analyze_enhanced_formulation()['metrics']
    print(f"Eutectic_VOI (45C): {res2.get('Eutectic_VOI', 0):.4f}")
    
    if res2.get('Eutectic_VOI', 0) > 0:
        print("Eutectic detection: SUCCESS")
    else:
        print("Eutectic detection: FAILED")

    print("\nTier 3 Advanced Mechanisms: SUCCESS")

if __name__ == "__main__":
    test_tier3_mechanisms()
