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

def test_metal_ox_physics():
    print("=== TESTING METAL-CATALYSED OXIDATION (Tier 2) ===\n")
    
    # 1. API: Thiol-containing (e.g., Captopril)
    smi_cap = "C[C@H](CS)C(=O)N1[C@H](CCC1)C(=O)O"
    mol_cap = Chem.MolFromSmiles(smi_cap)
    
    api_cap = Molecule(
        name="Captopril",
        mol_type=MoleculeType.SMALL_MOLECULE,
        coords=np.random.randn(mol_cap.GetNumAtoms(), 3),
        charges=np.zeros(mol_cap.GetNumAtoms()),
        hydrophobicity=np.random.randn(mol_cap.GetNumAtoms()),
        vdw_radii=np.ones(mol_cap.GetNumAtoms()) * 1.5,
        h_donors=np.zeros(mol_cap.GetNumAtoms()),
        h_acceptors=np.zeros(mol_cap.GetNumAtoms()),
        polarizability=np.ones(mol_cap.GetNumAtoms()),
        reactive_groups=np.zeros(mol_cap.GetNumAtoms()),
        smiles=smi_cap,
        concentration=1.0
    )
    
    # 2. Excipient: Iron-contaminated (proxy: iron ion in SMILES)
    smi_iron = "[Fe+2].CC(C)O" # Isopropanol with trace Iron
    mol_iron = Chem.MolFromSmiles(smi_iron)
    
    exc_iron = Molecule(
        name="Contaminated_Excipient",
        mol_type=MoleculeType.SMALL_MOLECULE,
        coords=np.random.randn(mol_iron.GetNumAtoms(), 3) + 2.0,
        charges=np.zeros(mol_iron.GetNumAtoms()),
        hydrophobicity=np.random.randn(mol_iron.GetNumAtoms()),
        vdw_radii=np.ones(mol_iron.GetNumAtoms()) * 1.5,
        h_donors=np.zeros(mol_iron.GetNumAtoms()),
        h_acceptors=np.zeros(mol_iron.GetNumAtoms()),
        polarizability=np.ones(mol_iron.GetNumAtoms()),
        reactive_groups=np.zeros(mol_iron.GetNumAtoms()),
        smiles=smi_iron,
        concentration=1.0
    )
    
    # 3. Excipient: Peroxide-containing (e.g., PEG with peroxide)
    smi_peg = "COCCOCCOCCO.OO" # PEG with Hydrogen Peroxide
    mol_peg = Chem.MolFromSmiles(smi_peg)
    
    exc_peg = Molecule(
        name="PEG_Peroxide",
        mol_type=MoleculeType.POLYMER,
        coords=np.random.randn(mol_peg.GetNumAtoms(), 3) - 2.0,
        charges=np.zeros(mol_peg.GetNumAtoms()),
        hydrophobicity=np.random.randn(mol_peg.GetNumAtoms()),
        vdw_radii=np.ones(mol_peg.GetNumAtoms()) * 1.5,
        h_donors=np.zeros(mol_peg.GetNumAtoms()),
        h_acceptors=np.zeros(mol_peg.GetNumAtoms()),
        polarizability=np.ones(mol_peg.GetNumAtoms()),
        reactive_groups=np.zeros(mol_peg.GetNumAtoms()),
        smiles=smi_peg,
        concentration=1.0
    )
    
    system = KeyBoxSystem()
    system.add_molecule(api_cap, is_api=True)
    system.add_molecule(exc_iron, is_api=False)
    system.add_molecule(exc_peg, is_api=False)
    
    result = system.analyze_enhanced_formulation()
    metrics = result['metrics']
    recs = system.get_stability_recommendations()
    
    print(f"MetalOx_VOI: {metrics['MetalOx_VOI']:.4f}")
    print(f"OCS: {metrics['OCS']:.2f}")
    print("\nRecommendations:")
    for r in recs:
        if "Metal-catalysed" in r or "Haber-Weiss" in r:
            print(f"- {r}")
            
    assert metrics['MetalOx_VOI'] > 0.1
    print("\nMetal-Catalysed Oxidation Physics (Tier 2): SUCCESS\n")

if __name__ == "__main__":
    try:
        test_metal_ox_physics()
        print("MetalOx Verification: SUCCESS")
    except Exception as e:
        print(f"MetalOx Verification: FAILED - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
