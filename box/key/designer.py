"""
KeyBox Designer: High-Level Formulation Logic & Screening
Inspired by Leap71 Buildup logic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from scipy.stats import norm, qmc

from .models import (
    Molecule, MoleculeType, EnvironmentalConditions, 
    DosageForm, DosageFormConfig
)
from .core_engine import KeyBoxSystem

def create_enhanced_molecule_from_smiles(smiles: str, name: str, mol_type: MoleculeType = MoleculeType.SMALL_MOLECULE,
                                       concentration: float = 1.0) -> Molecule:
    """Create enhanced molecule from SMILES with all properties"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array([float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0 
                       for atom in mol.GetAtoms()])
    
    # Metal ion adjustment
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() in [12, 20]: # Mg, Ca
            charges[i] = 2.0
        elif atom.GetAtomicNum() in [11, 19]: # Na, K
            charges[i] = 1.0

    hydrophobicity = np.array([Crippen._GetAtomContribs(mol)[i][0] for i in range(mol.GetNumAtoms())])
    vdw_radii = np.array([Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()])
    
    h_donors = np.zeros(mol.GetNumAtoms())
    h_acceptors = np.zeros(mol.GetNumAtoms())
    donor_pattern = Chem.MolFromSmarts('[OH,NH,NH2,SH]')
    if donor_pattern:
        for match in mol.GetSubstructMatches(donor_pattern): h_donors[match[0]] = 1.0
    
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalDegree() < 4: h_acceptors[i] = 1
    
    polarizability = np.array([atom.GetAtomicNum() * 0.1 for atom in mol.GetAtoms()])
    acid_groups, base_groups, amine_groups, carbonyl_groups = [np.zeros(mol.GetNumAtoms()) for _ in range(4)]
    
    for smarts in ['[CX3](=O)[OX2H1]', '[SX4](=O)(=O)[OX2H1]']:
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            for m in mol.GetSubstructMatches(pat):
                for idx in m: acid_groups[idx] = 1.0
                
    for smarts in ['[NX3;H2,H1,H0;!$(NC=O)]', '[$(N[C,H](=[N,H])[N,H])]']:
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            for m in mol.GetSubstructMatches(pat):
                for idx in m: base_groups[idx] = 1.0
    
    # Metal cations as bases (Mg2+, Ca2+, Na+, K+)
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() in [12, 20, 11, 19]:  # Mg, Ca, Na, K
            base_groups[i] = 1.0
    
    # Maillard specifics
    amine_pat = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
    if amine_pat:
        for m in mol.GetSubstructMatches(amine_pat):
            for idx in m: amine_groups[idx] = 1.0
            
    sugar_smarts = ['[OX2H][CX4H]1[OX2][CX4][CX4][CX4][CX4]1', '[CX3H1](=O)[#6]'] # Hemiacetal and Aldehyde
    for smarts in sugar_smarts:
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            for m in mol.GetSubstructMatches(pat):
                for idx in m: carbonyl_groups[idx] = 1.0

    reactive_groups = np.zeros(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() in [7, 8] and mol.GetAtomWithIdx(i).GetTotalDegree() <= 3:
            reactive_groups[i] = 1
            
    return Molecule(
        name=name, mol_type=mol_type, coords=coords, charges=charges,
        hydrophobicity=hydrophobicity, vdw_radii=vdw_radii, h_donors=h_donors,
        h_acceptors=h_acceptors, polarizability=polarizability,
        reactive_groups=reactive_groups, acid_groups=acid_groups,
        base_groups=base_groups, amine_groups=amine_groups,
        carbonyl_groups=carbonyl_groups, concentration=concentration,
        molecular_weight=Descriptors.MolWt(mol), smiles=smiles
    )

ENHANCED_EXCIPIENT_LIBRARY = {
    'Lactose': ('C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O)O', MoleculeType.SMALL_MOLECULE),
    'Mannitol': ('C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O', MoleculeType.SMALL_MOLECULE),
    'PEG_400': ('COCCOCCOCCOCCOCCO', MoleculeType.POLYMER),
    'PVP': ('CC(C(=O)N1CCCC1)C', MoleculeType.POLYMER),
    'HPMC': ('COC1C(C(C(C(O1)CO)O)O)OC', MoleculeType.POLYMER),
    'MCC': ('C1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O', MoleculeType.POLYMER),
    'Starch': ('C1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O', MoleculeType.POLYMER),
    'Stearic_Acid': ('CCCCCCCCCCCCCCCCCC(=O)O', MoleculeType.LIPID),
    'Magnesium_Stearate': ('[Mg+2].[O-]C(=O)CCCCCCCCCCCCCCCCC.[O-]C(=O)CCCCCCCCCCCCCCCCC', MoleculeType.LIPID),
    'Citric_Acid': ('C(C(=O)O)C(CC(=O)O)(C(=O)O)O', MoleculeType.SMALL_MOLECULE),
    'SLS': ('CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]', MoleculeType.SURFACTANT)
}

def screen_enhanced_excipients_for_api(api_smiles: str, api_name: str, api_type: MoleculeType,
                                     excipient_library: Dict[str, Tuple[str, MoleculeType]],
                                     dosage_config: Optional[DosageFormConfig] = None,
                                     env_conditions: Optional[EnvironmentalConditions] = None) -> pd.DataFrame:
    print(f"=== KeyBox Designer: Screening for {api_name} ===")
    api = create_enhanced_molecule_from_smiles(api_smiles, api_name, api_type)
    results = []
    for name, (smiles, etype) in excipient_library.items():
        try:
            exc = create_enhanced_molecule_from_smiles(smiles, name, etype)
            sys = KeyBoxSystem(dosage_config=dosage_config, env_conditions=env_conditions)
            sys.add_molecule(api, is_api=True)
            sys.add_molecule(exc, is_api=False)
            res = sys.analyze_enhanced_formulation(include_uncertainty=True)
            res['excipient'] = name
            results.append(res)
            un = res['confidence_intervals'].get('std', 0.0)
            print(f"✓ {name}: OCS={res['metrics']['OCS']:.1f} ± {un:.1f}")
        except Exception as e:
            print(f"✗ {name}: {e}")
            
    df = pd.DataFrame([{
        'Excipient': r['excipient'], 
        'OCS': f"{r['metrics']['OCS']:.1f} ± {r['confidence_intervals'].get('std', 0.0):.1f}", 
        'Classification': r['classification'], 
        'Risks': ", ".join(r['mechanisms'][:2])
    } for r in sorted(results, key=lambda x: x['metrics']['OCS'], reverse=True)])
    print(df.to_string(index=False))
    return df

class DesignerOptimizer:
    """Buildup logic for optimizing formulations"""
    def __init__(self, system_bounds=(30,30,30)):
        self.bounds = system_bounds
        
    def find_best_excipient(self, api_smiles, api_name):
        return screen_enhanced_excipients_for_api(api_smiles, api_name, MoleculeType.SMALL_MOLECULE, ENHANCED_EXCIPIENT_LIBRARY)

def analyze_enhanced_formulation_with_visualization(api_smiles, api_name, exc_smiles, exc_name):
    api = create_enhanced_molecule_from_smiles(api_smiles, api_name)
    exc = create_enhanced_molecule_from_smiles(exc_smiles, exc_name)
    sys = KeyBoxSystem()
    sys.add_molecule(api, is_api=True)
    sys.add_molecule(exc, is_api=False)
    res = sys.analyze_enhanced_formulation(include_uncertainty=True)
    sys.visualize_comprehensive_enhanced_analysis()
    
    # Generate OCS distribution plot
    if 'confidence_intervals' in res and 'distribution' in res['confidence_intervals']:
        plt.figure(figsize=(8, 5))
        dist = res['confidence_intervals']['distribution']
        plt.hist(dist, bins=15, alpha=0.7, color='teal', edgecolor='black')
        plt.axvline(res['metrics']['OCS'], color='red', linestyle='dashed', linewidth=2, label=f"Mean={res['metrics']['OCS']:.1f}")
        plt.title(f"OCS Uncertainty Distribution: {api_name} + {exc_name}")
        plt.xlabel("Overall Compatibility Score (OCS)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("viz_output/ocs_distribution.jpg")
        plt.close()
        print(f"[VISUALIZATION] Distribution plot saved to viz_output/ocs_distribution.jpg")
        
    return res
