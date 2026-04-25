import sys
import os
import time
import numpy as np
from rdkit import Chem

# Set path to 'box' directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key.core_engine import KeyBoxSystem, Molecule, MoleculeType, DosageForm, DosageFormConfig
from key.designer import create_enhanced_molecule_from_smiles, _extract_atom_properties

def test_pvp_smarts_fix():
    print("--- Testing PVP SMARTS Fix ---")
    # PVP monomer: cyclic lactam
    pvp_smiles = "CC(C(=O)N1CCCC1)C"
    mol = Chem.MolFromSmiles(pvp_smiles)
    mol_h = Chem.AddHs(mol)
    
    props = _extract_atom_properties(mol_h)
    base_count = np.sum(props['base_groups'])
    
    print(f"PVP base group count: {base_count}")
    # Lactams are non-basic.
    assert base_count == 0, f"PVP should have 0 base groups, found {base_count}"
    print("PVP SMARTS fix: VALID")

def test_performance_scaling():
    print("\n--- Testing Performance Scaling ---")
    # Use a medium-sized molecule
    api_smi = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    api = create_enhanced_molecule_from_smiles(api_smi, "Aspirin")
    
    # 1. 20^3 grid (8000 voxels) - Threshold is exactly 8000.
    # In my core_engine.py: USE_SPARSE = N >= 8000
    # So 20^3 should use SPARSE path? 
    # Wait, 20*20*20 = 8000. 
    # Let's check 15^3 vs 25^3.
    
    def measure(side):
        kbs = KeyBoxSystem(voxel_bounds=(float(side), float(side), float(side)), resolution=1.0)
        kbs.add_molecule(api, is_api=True)
        start = time.perf_counter()
        fields = kbs.project_enhanced_molecule_fields(api)
        end = time.perf_counter()
        return (end - start) * 1000, fields

    t15, f15 = measure(15)
    print(f"15^3 Grid (3375 voxels, Dense path) time: {t15:.2f} ms")
    
    t25, f25 = measure(25)
    print(f"25^3 Grid (15625 voxels, Sparse path) time: {t25:.2f} ms")
    
    vox_ratio = (25/15)**3
    time_ratio = t25 / t15
    print(f"Voxel ratio: {vox_ratio:.2f}x | Time ratio: {time_ratio:.2f}x")
    
    # In dense, time_ratio should be >= vox_ratio.
    # In sparse, time_ratio should be < vox_ratio because cost is O(atoms) not O(voxels).
    print(f"Efficiency gain vs dense scaling: {vox_ratio / time_ratio:.2f}x")
    
    # Verify field consistency
    v1 = np.max(np.abs(f15['electrostatic']))
    v2 = np.max(np.abs(f25['electrostatic']))
    print(f"Max Electrostatic consistency: {v1:.4e} vs {v2:.4e}")
    assert abs(v1 - v2) / max(v1, 1e-10) < 0.1

if __name__ == "__main__":
    test_pvp_smarts_fix()
    test_performance_scaling()
    print("\nStep 7 Verification: SUCCESS")
