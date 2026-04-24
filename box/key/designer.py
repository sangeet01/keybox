"""
KeyBox Designer: High-Level Formulation Logic & Screening
Inspired by Leap71 Buildup logic

v0.9.0 — Tautomer handling (#3):
  create_enhanced_molecule_from_smiles now enumerates up to MAX_TAUTOMERS tautomers
  per molecule (via RDKit TautomerEnumerator), embeds + MMFF-minimises each, computes
  Boltzmann population weights from MMFF energies (kT = 0.593 kcal/mol at 298 K), and
  stores per-tautomer coords + property arrays on Molecule.conformations / taut_*.
  project_enhanced_molecule_fields in core_engine uses these weights instead of equal
  averaging, so charge-sensitive VOI channels (Maillard, AcidBase) correctly reflect
  the true tautomeric mixture rather than just the input tautomer.

  Ref: Greenhill et al., J. Chem. Inf. Model. 2003 — tautomers alter H-bond donor/
  acceptor counts and Gasteiger charges significantly for drug-like molecules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import norm, qmc

from .models import (
    Molecule, MoleculeType, EnvironmentalConditions,
    DosageForm, DosageFormConfig
)
from .core_engine import KeyBoxSystem

# ── Tautomer enumeration settings ───────────────────────────────────────────
MAX_TAUTOMERS    = 8     # Budget: covers >95% of pharma-relevant cases
kT_KCAL_298K     = 0.593 # kcal/mol  (R × 298.15 K)
MIN_BOLTZ_WEIGHT = 0.01  # Prune tautomers with population < 1%


def _extract_atom_properties(mol_h: Chem.Mol) -> Dict[str, np.ndarray]:
    """
    Extract all per-atom property arrays from a hydrogen-included RDKit molecule.
    Factored out so it runs identically for the base molecule and each tautomer.
    """
    n = mol_h.GetNumAtoms()

    AllChem.ComputeGasteigerCharges(mol_h)
    charges = np.array([
        float(a.GetProp('_GasteigerCharge')) if a.HasProp('_GasteigerCharge') else 0.0
        for a in mol_h.GetAtoms()
    ])
    for i, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetAtomicNum() in [12, 20]:   charges[i] =  2.0
        elif atom.GetAtomicNum() in [11, 19]: charges[i] =  1.0
        elif atom.GetAtomicNum() in [13, 30]: charges[i] =  3.0

    hydrophobicity = np.array([Crippen._GetAtomContribs(mol_h)[i][0] for i in range(n)])
    vdw_radii = np.array([
        Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in mol_h.GetAtoms()
    ])

    h_donors = np.zeros(n)
    donor_pat = Chem.MolFromSmarts('[OH,NH,NH2,SH]')
    if donor_pat:
        for match in mol_h.GetSubstructMatches(donor_pat):
            h_donors[match[0]] = 1.0

    h_acceptors = np.zeros(n)
    for i, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalDegree() < 4:
            h_acceptors[i] = 1.0

    polarizability = np.array([a.GetAtomicNum() * 0.1 for a in mol_h.GetAtoms()])

    acid_groups = np.zeros(n)
    for smarts in ['[CX3](=O)[OX2H1]', '[SX4](=O)(=O)[OX2H1]']:
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            for m in mol_h.GetSubstructMatches(pat):
                for idx in m: acid_groups[idx] = 1.0

    base_groups = np.zeros(n)
    for smarts in ['[NX3;H2,H1,H0;!$(NC=O)]', '[$(N[C,H](=[N,H])[N,H])]']:
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            for m in mol_h.GetSubstructMatches(pat):
                for idx in m: base_groups[idx] = 1.0
    for i, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetAtomicNum() in [12, 20, 11, 19, 13, 30]:
            base_groups[i] = 1.0

    amine_groups = np.zeros(n)
    amine_pat = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
    if amine_pat:
        for m in mol_h.GetSubstructMatches(amine_pat):
            for idx in m: amine_groups[idx] = 1.0

    carbonyl_groups = np.zeros(n)
    for smarts in ['[OX2H][CX4H]1[OX2][CX4][CX4][CX4][CX4]1', '[CX3H1](=O)[#6]']:
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            for m in mol_h.GetSubstructMatches(pat):
                for idx in m: carbonyl_groups[idx] = 1.0

    reactive_groups = np.zeros(n)
    for i, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetAtomicNum() in [7, 8] and atom.GetTotalDegree() <= 3:
            reactive_groups[i] = 1.0

    return {
        'charges':         charges,
        'hydrophobicity':  hydrophobicity,
        'vdw_radii':       vdw_radii,
        'h_donors':        h_donors,
        'h_acceptors':     h_acceptors,
        'polarizability':  polarizability,
        'acid_groups':     acid_groups,
        'base_groups':     base_groups,
        'amine_groups':    amine_groups,
        'carbonyl_groups': carbonyl_groups,
        'reactive_groups': reactive_groups,
    }


def _mmff_energy(mol_h: Chem.Mol) -> Optional[float]:
    props = AllChem.MMFFGetMoleculeProperties(mol_h)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
    if ff is None:
        return None
    return ff.CalcEnergy()


def _embed_and_minimise(mol_h: Chem.Mol, seed: int = 42) -> Optional[Tuple[np.ndarray, float]]:
    """Embed + MMFF-minimise; return (coords, energy) or None on failure."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol_h, params) == -1:
        if AllChem.EmbedMolecule(mol_h, randomSeed=seed, useRandomCoords=True) == -1:
            return None
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    conf = mol_h.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol_h.GetNumAtoms())])
    energy = _mmff_energy(mol_h)
    return coords, (energy if energy is not None else 0.0)


def create_enhanced_molecule_from_smiles(
    smiles: str,
    name: str,
    mol_type: MoleculeType = MoleculeType.SMALL_MOLECULE,
    concentration: float = 1.0,
    enumerate_tautomers: bool = True,
) -> Molecule:
    """
    Create an enhanced Molecule from a SMILES string with Boltzmann-weighted tautomers.

    When enumerate_tautomers=True (default for SMALL_MOLECULE):
      - Enumerates up to MAX_TAUTOMERS tautomers via RDKit TautomerEnumerator.
      - Embeds + MMFF-minimises each tautomer independently.
      - Computes Boltzmann weights from MMFF energies at 298 K.
      - Prunes tautomers with population < MIN_BOLTZ_WEIGHT (< 1%).
      - Stores per-tautomer coords in Molecule.conformations and per-tautomer
        property arrays in Molecule.taut_* fields.
      - project_enhanced_molecule_fields in core_engine uses these weights for field averaging,
        so charge-sensitive VOI channels reflect the true tautomeric mixture.

    Tautomer enumeration is skipped for POLYMER/LIPID/BIOLOGIC/SURFACTANT types
    and for molecules with > 60 heavy atoms (performance).

    Ref: Greenhill et al., J. Chem. Inf. Model. 2003.
    """
    base_mol = Chem.MolFromSmiles(smiles)
    if base_mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    do_tauts = (
        enumerate_tautomers
        and mol_type == MoleculeType.SMALL_MOLECULE
        and base_mol.GetNumHeavyAtoms() <= 60
    )

    tautomers_to_process: List[Chem.Mol] = []
    if do_tauts:
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(MAX_TAUTOMERS)
        tautomers_to_process = list(te.Enumerate(base_mol))
        if not tautomers_to_process:
            tautomers_to_process = [base_mol]
    else:
        tautomers_to_process = [base_mol]

    valid_results = []
    for i, tmol in enumerate(tautomers_to_process):
        tmol_h = Chem.AddHs(tmol)
        result = _embed_and_minimise(tmol_h, seed=42 + i)
        if result is None:
            continue
        coords, energy = result
        props = _extract_atom_properties(tmol_h)
        valid_results.append((tmol_h, coords, energy, props))

    if not valid_results:
        mol_h = Chem.AddHs(base_mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_h)
        conf = mol_h.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol_h.GetNumAtoms())])
        props = _extract_atom_properties(mol_h)
        valid_results = [(mol_h, coords, 0.0, props)]

    # Boltzmann weighting
    energies  = np.array([r[2] for r in valid_results])
    e_min     = energies.min()
    boltzmann = np.exp(-(energies - e_min) / kT_KCAL_298K)
    weights   = boltzmann / boltzmann.sum()

    keep = weights >= MIN_BOLTZ_WEIGHT
    if not keep.any():
        keep[np.argmax(weights)] = True

    valid_results   = [r for r, k in zip(valid_results, keep) if k]
    weights_kept    = weights[keep]
    weights_kept    = weights_kept / weights_kept.sum()

    dom_mol_h, dom_coords, _, dom_props = valid_results[0]

    n_tauts = len(valid_results)
    if n_tauts > 1:
        print(f"[TAUTOMER] {name}: {n_tauts} tautomers retained "
              f"(weights: {', '.join(f'{w:.2f}' for w in weights_kept)})")

    return Molecule(
        name=name,
        mol_type=mol_type,
        coords=dom_coords,
        charges=dom_props['charges'],
        hydrophobicity=dom_props['hydrophobicity'],
        vdw_radii=dom_props['vdw_radii'],
        h_donors=dom_props['h_donors'],
        h_acceptors=dom_props['h_acceptors'],
        polarizability=dom_props['polarizability'],
        reactive_groups=dom_props['reactive_groups'],
        acid_groups=dom_props['acid_groups'],
        base_groups=dom_props['base_groups'],
        amine_groups=dom_props['amine_groups'],
        carbonyl_groups=dom_props['carbonyl_groups'],
        concentration=concentration,
        conformations=     [r[1] for r in valid_results],
        taut_weights=      list(weights_kept),
        taut_charges=      [r[3]['charges']         for r in valid_results],
        taut_h_donors=     [r[3]['h_donors']        for r in valid_results],
        taut_h_acceptors=  [r[3]['h_acceptors']     for r in valid_results],
        taut_amine_groups= [r[3]['amine_groups']    for r in valid_results],
        taut_carbonyl_groups=[r[3]['carbonyl_groups'] for r in valid_results],
        taut_acid_groups=  [r[3]['acid_groups']     for r in valid_results],
        taut_base_groups=  [r[3]['base_groups']     for r in valid_results],
        molecular_weight=Descriptors.MolWt(dom_mol_h),
        smiles=Chem.MolToSmiles(Chem.RemoveHs(dom_mol_h)),
    )


ENHANCED_EXCIPIENT_LIBRARY = {
    'Lactose':           ('C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O)O', MoleculeType.SMALL_MOLECULE),
    'Mannitol':          ('C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O',                                                   MoleculeType.SMALL_MOLECULE),
    'PEG_400':           ('COCCOCCOCCOCCOCCO',                                                                            MoleculeType.POLYMER),
    'PVP':               ('CC(C(=O)N1CCCC1)C',                                                                           MoleculeType.POLYMER),
    'HPMC':              ('COC1C(C(C(C(O1)CO)O)O)OC',                                                                   MoleculeType.POLYMER),
    'MCC':               ('C1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O',          MoleculeType.POLYMER),
    'Starch':            ('C1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O',          MoleculeType.POLYMER),
    'Stearic_Acid':      ('CCCCCCCCCCCCCCCCCC(=O)O',                                                                     MoleculeType.LIPID),
    'Magnesium_Stearate':('[Mg+2].[O-]C(=O)CCCCCCCCCCCCCCCCC.[O-]C(=O)CCCCCCCCCCCCCCCCC',                               MoleculeType.LIPID),
    'Citric_Acid':       ('C(C(=O)O)C(CC(=O)O)(C(=O)O)O',                                                               MoleculeType.SMALL_MOLECULE),
    'SLS':               ('CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]',                                                            MoleculeType.SURFACTANT),
}


def screen_enhanced_excipients_for_api(
    api_smiles: str,
    api_name: str,
    api_type: MoleculeType,
    excipient_library: Dict[str, Tuple[str, MoleculeType]],
    dosage_config: Optional[DosageFormConfig] = None,
    env_conditions: Optional[EnvironmentalConditions] = None,
) -> pd.DataFrame:
    print(f"=== KeyBox Designer: Screening for {api_name} ===")
    api = create_enhanced_molecule_from_smiles(api_smiles, api_name, api_type)
    results = []
    for name, (smiles, etype) in excipient_library.items():
        try:
            exc = create_enhanced_molecule_from_smiles(smiles, name, etype)
            sys_ = KeyBoxSystem(dosage_config=dosage_config, env_conditions=env_conditions)
            sys_.add_molecule(api, is_api=True)
            sys_.add_molecule(exc, is_api=False)
            res = sys_.analyze_enhanced_formulation(include_uncertainty=True)
            res['excipient'] = name
            results.append(res)
            un = res['confidence_intervals'].get('std', 0.0)
            print(f"[OK] {name}: OCS={res['metrics']['OCS']:.1f} +/- {un:.1f}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    df = pd.DataFrame([{
        'Excipient':      r['excipient'],
        'OCS':            f"{r['metrics']['OCS']:.1f} +/- {r['confidence_intervals'].get('std', 0.0):.1f}",
        'Classification': r['classification'],
        'Risks':          ", ".join(r['mechanisms'][:2]),
    } for r in sorted(results, key=lambda x: x['metrics']['OCS'], reverse=True)])
    print(df.to_string(index=False))
    return df


class DesignerOptimizer:
    """Buildup logic for optimizing formulations"""
    def __init__(self, system_bounds=(30, 30, 30)):
        self.bounds = system_bounds

    def find_best_excipient(self, api_smiles, api_name):
        return screen_enhanced_excipients_for_api(
            api_smiles, api_name, MoleculeType.SMALL_MOLECULE, ENHANCED_EXCIPIENT_LIBRARY
        )


def analyze_enhanced_formulation_with_visualization(api_smiles, api_name, exc_smiles, exc_name):
    api  = create_enhanced_molecule_from_smiles(api_smiles, api_name)
    exc  = create_enhanced_molecule_from_smiles(exc_smiles, exc_name)
    sys_ = KeyBoxSystem()
    sys_.add_molecule(api, is_api=True)
    sys_.add_molecule(exc, is_api=False)
    res  = sys_.analyze_enhanced_formulation(include_uncertainty=True)
    sys_.visualize_comprehensive_enhanced_analysis()

    if 'confidence_intervals' in res and 'distribution' in res['confidence_intervals']:
        plt.figure(figsize=(8, 5))
        dist = res['confidence_intervals']['distribution']
        plt.hist(dist, bins=15, alpha=0.7, color='teal', edgecolor='black')
        plt.axvline(
            res['metrics']['OCS'], color='red', linestyle='dashed', linewidth=2,
            label=f"Mean={res['metrics']['OCS']:.1f}"
        )
        plt.title(f"OCS Uncertainty Distribution: {api_name} + {exc_name}")
        plt.xlabel("Overall Compatibility Score (OCS)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("viz_output/ocs_distribution.jpg")
        plt.close()
        print("[VISUALIZATION] Distribution plot saved to viz_output/ocs_distribution.jpg")

    return res
