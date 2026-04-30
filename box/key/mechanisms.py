import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from rdkit import Chem
from rdkit.Chem import Descriptors
from .models import EnvironmentalConditions, Molecule, MoleculeType, DosageForm

# Metal-Catalysed Oxidation constants
_METAL_OX_SMARTS: Dict[str, Tuple[str, float]] = {
    'thiol':           ('[SX2H]',                                  1.0),
    'sulfide':         ('[SX2;!H;!$(S=O)]',                       0.9),
    'catechol':        ('Oc1ccccc1O',                             0.9),
    'aldehyde':        ('[CX3H1](=O)',                            0.8),
    'dihydropyridine': ('C1=CNC(=O)C=C1',                        0.9),
    'phenol_ortho':    ('[OX2H]c1ccccc1[OX2H]',                  0.7),
    'simple_phenol':   ('[OX2H]c1ccccc1',                        0.5),
    'tert_amine':      ('[NX3;H0;!$(NC=O);!$(N[C]=[N,n]);!$(n)]',0.5),
    'olefin_conj':     ('[CX3]=[CX3][CX3](=O)',                  0.4),
    'benzyl_ch':       ('[cX3][CH2]',                            0.3),
    'ether_alpha':     ('[OX2][CX4H]',                           0.2),
}
_METAL_OX_COMPILED: Dict[str, Tuple[Optional[Chem.Mol], float]] = {
    k: (Chem.MolFromSmarts(v[0]), v[1])
    for k, v in _METAL_OX_SMARTS.items()
}
_METAL_ION_SMARTS: Dict[str, Optional[Chem.Mol]] = {
    'iron_ion':      Chem.MolFromSmarts('[Fe+2,Fe+3]'),
    'copper_ion':    Chem.MolFromSmarts('[Cu+2,Cu+]'),
    'manganese_ion': Chem.MolFromSmarts('[Mn+2,Mn+3]'),
    'titanium':      Chem.MolFromSmarts('[Ti]'),
    'zinc_ion':      Chem.MolFromSmarts('[Zn+2]'),
}
_METAL_PRONE_EXCIPIENT_SUBSTRINGS: List[str] = ['C@@H', '[Si]', '[Ti]']
_PEROXIDE_SMARTS: Dict[str, Optional[Chem.Mol]] = {
    'peroxide': Chem.MolFromSmarts('[OX2][OX2]'),
    'hydroperoxide': Chem.MolFromSmarts('[OX2H][OX2]'),
}

def compute_metal_ox_voi(api_smiles: Optional[str], exc_smiles: Optional[str],
                         T_celsius: float = 25.0, oxygen_level: float = 0.21,
                         moisture: float = 0.3, metal_density: float = 0.1,
                         peroxide_density: float = 0.05) -> Tuple[float, List[str], float]:
    api_mol = Chem.MolFromSmiles(api_smiles) if api_smiles else None
    exc_mol = Chem.MolFromSmiles(exc_smiles) if exc_smiles else None
    if api_mol is None: return 0.0, [], 0.0
    found: Dict[str, float] = {}
    for gname, (patt, weight) in _METAL_OX_COMPILED.items():
        if patt is not None and api_mol.HasSubstructMatch(patt):
            found[gname] = weight
    ox_score = float(min(sum(found.values()), 3.0))
    if ox_score < 0.01: return 0.0, [], 0.0
    
    # Metal and Peroxide exposure from excipient
    metal_exposure = metal_density
    if exc_mol:
        for patt in _METAL_ION_SMARTS.values():
            if patt is not None and exc_mol.HasSubstructMatch(patt):
                metal_exposure = max(metal_exposure, 0.8); break
    
    per_exposure = peroxide_density
    if exc_mol:
        for patt in _PEROXIDE_SMARTS.values():
            if patt is not None and exc_mol.HasSubstructMatch(patt):
                per_exposure = max(per_exposure, 0.5); break
    
    oxy_factor = float(np.clip(oxygen_level / 0.21, 0.0, 1.0))
    moist_factor = 1.0 + 0.5 * float(moisture)
    Ea, T_K = 50000.0, T_celsius + 273.15  # Fenton Ea is lower than general ox
    arr = float(np.exp(Ea / 8.314 * (1.0 / 298.15 - 1.0 / T_K)))
    
    # Haber-Weiss/Fenton synergy: Metal * Peroxide * Oxygen
    voi = ox_score * metal_exposure * per_exposure * oxy_factor * moist_factor * arr * 10.0
    return float(np.log1p(voi)), list(found.keys()), ox_score

# Polymorph Conversion constants
_TG_REGRESSION_COEFFS = np.array([-81.6866, 0.2875, -10.8036, -25.5313, 0.7351, 5.7564, 11.5409])
_LUBRICANT_PATTERNS: Dict[str, Optional[Chem.Mol]] = {
    'fatty_acid_anion': Chem.MolFromSmarts('[CX3](=O)[O-]'),
    'long_chain_acid':  Chem.MolFromSmarts('CCCCCCCC(=O)[OH,O-]'),
}

def classify_gfa(smiles: str) -> Tuple[float, str, float, float]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None: return 0.3, 'II', 30.0, 27.9
    mw, nrot = float(Descriptors.MolWt(mol)), int(Descriptors.NumRotatableBonds(mol))
    hbd, tpsa = int(Descriptors.NumHDonors(mol)), float(Descriptors.TPSA(mol))
    logP = float(Descriptors.MolLogP(mol))
    n_ar = sum(1 for r in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    feat = np.array([1.0, mw, nrot, hbd, tpsa, logP, float(n_ar)])
    Tg_C = float(np.clip(_TG_REGRESSION_COEFFS @ feat, -80.0, 250.0))
    Tg_eff = Tg_C * 0.93
    is_I, is_III = (logP > 1.5 and hbd <= 1 and n_ar < 2), (mw > 500 or (hbd >= 4 and mw > 200) or tpsa > 150 or nrot >= 8)
    aryl_b, rot_b = 0.1 * min(n_ar, 3), 0.05 * min(nrot, 6)
    if is_III: gfa_cl, gfa_s = 'III', float(np.clip(0.70 + aryl_b + rot_b, 0.65, 1.0))
    elif is_I: gfa_cl, gfa_s = 'I', 0.20
    else: gfa_cl, gfa_s = 'II', float(np.clip(0.50 + aryl_b + rot_b, 0.35, 0.64))
    return gfa_s, gfa_cl, Tg_C, Tg_eff

def compute_polymorph_voi(api_smiles: Optional[str], exc_smiles: Optional[str],
                          T_celsius: float = 25.0, moisture: float = 0.3,
                          storage_months: float = 12.0) -> Tuple[float, float, str, float, float]:
    if not api_smiles: return 0.0, 0.0, 'II', 0.3, 30.0
    gfa_s, gfa_cl, Tg_C, Tg_eff = classify_gfa(api_smiles)
    t_s = storage_months * 2592000.0
    tg_floor = -40.0 if gfa_cl == 'I' else -20.0
    if Tg_eff < tg_floor: return 0.0, 0.0, gfa_cl, gfa_s, Tg_C
    if Tg_eff < 0.0:
        mol = Chem.MolFromSmiles(api_smiles)
        if mol is not None and Descriptors.MolLogP(mol) < 2.0: return 0.0, 0.0, gfa_cl, gfa_s, Tg_C
    C1, C2 = 17.44, 51.6
    if Tg_eff >= 0.0:
        dT = T_celsius - Tg_eff
        denom = max(abs(C2 + dT), 0.1)
        wlf = float(np.clip(10 ** (C1 * dT / denom), 1e-10, 1e6))
    else: wlf = float(np.exp(0.05 * (T_celsius - 25.0)))
    exc_mol = Chem.MolFromSmiles(exc_smiles) if exc_smiles else None
    has_lub = exc_mol is not None and any(p is not None and exc_mol.HasSubstructMatch(p) for p in _LUBRICANT_PATTERNS.values())
    nucl_factor, moist_factor = (1.5 if has_lub else 1.0), float(np.exp(moisture * 2.0))
    k_base = 1e-14
    k_eff = float(np.clip(k_base * (1.0 - gfa_s)**2 * wlf * nucl_factor * moist_factor, 0.0, 1e10))
    X_cr = float(np.clip(1.0 - np.exp(-k_eff * t_s**2), 0.0, 1.0))
    X_cr_s = X_cr * (0.5 if Tg_eff < 0.0 else 1.0)
    voi = float(np.log1p(X_cr_s * (1.0 - gfa_s) * 5.0))
    return voi, X_cr, gfa_cl, gfa_s, Tg_C

# Photodegradation constants
_CHROMOPHORE_SMARTS: Dict[str, str] = {
    'aromatic_ketone': '[cX3][CX3](=[OX1])[#6;!$([OX2])]', 'nitroaromatic': '[cX3][N+](=O)[O-]',
    'alpha_beta_nitro': '[CX3]=[CX3][N+](=O)[O-]', 'quinone': 'O=C1C=CC(=O)C=C1',
    'catechol': 'Oc1ccccc1O', 'benzophenone': 'O=C(c1ccccc1)c1ccccc1',
    'diaryl_sulfide': 'c1ccccc1Sc1ccccc1', 'diaryl_amine': 'c1ccccc1Nc1ccccc1',
    'quinolone': 'O=c1cc[nH,n]c2ccccc12', 'conjugated_diene': '[CX3]=[CX3][CX3]=[CX3]',
    'alpha_diketone': '[CX3](=[OX1])[CX3](=[OX1])', 'enone': '[CX3](=[OX1])[CX3]=[CX3]',
    'furan_ring': 'c1ccco1', 'phenol': '[OX2H]c1ccccc1',
}
_CHROMOPHORE_WEIGHTS: Dict[str, float] = {
    'aromatic_ketone':1.0, 'nitroaromatic':1.0, 'alpha_beta_nitro':0.9, 'quinone':1.0,
    'catechol':0.8, 'benzophenone':1.0, 'diaryl_sulfide':0.9, 'diaryl_amine':0.7,
    'quinolone':0.6, 'conjugated_diene':0.6, 'alpha_diketone':0.7, 'enone':0.7,
    'furan_ring':0.4, 'phenol':0.3,
}
_PHOTOSENSITISER_SMARTS: Dict[str, str] = {
    'titanium_dioxide': '[Ti]', 'azo_dye': '[nX2]=[nX2]',
    'flavin': 'c1cnc2[nH]c(=O)[nH]c(=O)c2n1',
}
_CHROMOPHORE_COMPILED: Dict[str, Optional[Chem.Mol]] = {k: Chem.MolFromSmarts(v) for k, v in _CHROMOPHORE_SMARTS.items()}
_PHOTOSENS_COMPILED: Dict[str, Optional[Chem.Mol]] = {k: Chem.MolFromSmarts(v) for k, v in _PHOTOSENSITISER_SMARTS.items()}

def compute_photo_voi(api_smiles: Optional[str], exc_smiles: Optional[str],
                      oxygen_level: float = 0.21) -> Tuple[float, List[str], float]:
    api_mol = Chem.MolFromSmiles(api_smiles) if api_smiles else None
    exc_mol = Chem.MolFromSmiles(exc_smiles) if exc_smiles else None
    if api_mol is None: return 0.0, [], 0.0
    found: Dict[str, float] = {}
    for k, patt in _CHROMOPHORE_COMPILED.items():
        if patt is not None and api_mol.HasSubstructMatch(patt): found[k] = _CHROMOPHORE_WEIGHTS[k]
    chrom_score = float(min(sum(found.values()), 3.0))
    if chrom_score < 0.01: return 0.0, [], 0.0
    photosens = 0.0
    if exc_mol:
        for k, patt in _PHOTOSENS_COMPILED.items():
            if patt is not None and exc_mol.HasSubstructMatch(patt): photosens += 0.5
    oxy_factor = 0.3 + 0.7 * float(np.clip(oxygen_level / 0.21, 0.0, 1.0))
    voi = chrom_score * (1.0 + photosens) * oxy_factor
    return float(np.log1p(voi)), list(found.keys()), chrom_score

# Nucleophilic Addition constants
_NUCL_ADD_SMARTS: Dict[str, str] = {
    'thiol': '[SX2H]', 'prim_amine_nuc': '[NX3;H2;!$(NC=O);!$(N/C=C/O)]',
    'hydroxyl_nuc': '[OX2H;!$(OC=O);!$(Oc1ccccc1)]', 'michael_acceptor': '[CX3;!R](=[CX3])[CX3](=O)',
    'aldehyde_elec': '[CX3;H1,H2](=O)', 'epoxide_elec': 'C1OC1',
    'vinyl_sulf_elec': '[CX3]=[CX3][SX4](=[OX1])(=[OX1])',
}
_NUCL_REACTION_PAIRS: List[Tuple] = [
    ('thiol', 'michael_acceptor', 50.0, 1.0, 'Thiol-Michael addition'),
    ('thiol', 'aldehyde_elec', 55.0, 0.7, 'Thiol-aldehyde addition'),
    ('thiol', 'epoxide_elec', 60.0, 0.8, 'Thiol-epoxide ring opening'),
    ('prim_amine_nuc', 'aldehyde_elec', 45.0, 0.6, 'Amine-aldehyde Schiff base'),
    ('prim_amine_nuc', 'epoxide_elec', 65.0, 0.5, 'Amine-epoxide ring opening'),
    ('hydroxyl_nuc', 'epoxide_elec', 70.0, 0.4, 'Hydroxyl-epoxide ring opening'),
    ('hydroxyl_nuc', 'michael_acceptor', 75.0, 0.3, 'Hydroxyl-Michael addition'),
]
_NUCL_COMPILED: Dict[str, Optional[Chem.Mol]] = {k: Chem.MolFromSmarts(v) for k, v in _NUCL_ADD_SMARTS.items()}

def compute_nucl_add_voi(api_smiles: Optional[str], exc_smiles: Optional[str],
                         T_celsius: float = 25.0, nucleophile_density: float = 1.0,
                         electrophile_density: float = 1.0) -> Tuple[float, List[str]]:
    api_mol = Chem.MolFromSmiles(api_smiles) if api_smiles else None
    exc_mol = Chem.MolFromSmiles(exc_smiles) if exc_smiles else None
    if api_mol is None and exc_mol is None: return 0.0, []
    T_K, R, tot, trig = T_celsius + 273.15, 8.314, 0.0, []
    for nuc_key, elec_key, Ea_kJ, weight, label in _NUCL_REACTION_PAIRS:
        nuc_p, elec_p = _NUCL_COMPILED.get(nuc_key), _NUCL_COMPILED.get(elec_key)
        if nuc_p is None or elec_p is None: continue
        has_nuc_api, has_elec_exc = (api_mol and api_mol.HasSubstructMatch(nuc_p)), (exc_mol and exc_mol.HasSubstructMatch(elec_p))
        has_nuc_exc, has_elec_api = (exc_mol and exc_mol.HasSubstructMatch(nuc_p)), (api_mol and api_mol.HasSubstructMatch(elec_p))
        if not ((has_nuc_api and has_elec_exc) or (has_nuc_exc and has_elec_api)): continue
        arr = float(np.exp(Ea_kJ * 1000.0 / R * (1.0 / 298.15 - 1.0 / T_K)))
        tot += weight * min(nucleophile_density, electrophile_density) * arr
        trig.append(label)
    return float(np.log1p(tot)), trig

def compute_transester_voi(api_smiles: Optional[str], exc_smiles: Optional[str],
                          T_celsius: float = 25.0, pH: float = 7.0,
                          moisture: float = 0.3, alcohol_density: float = 1.0) -> Tuple[float, List[str]]:
    api_mol = Chem.MolFromSmiles(api_smiles) if api_smiles else None
    exc_mol = Chem.MolFromSmiles(exc_smiles) if exc_smiles else None
    if api_mol is None or exc_mol is None: return 0.0, []
    
    # Using more specific SMARTS as suggested by Tier 3 design
    ester_p = Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]')
    nuc_p   = Chem.MolFromSmarts('[OX2H][CX4]') # alcohol
    
    pairs = []
    if api_mol.HasSubstructMatch(ester_p) and exc_mol.HasSubstructMatch(nuc_p):
        pairs.append(f"{api_smiles[:10]}-ester + {exc_smiles[:10]}-alcohol")
    
    if not pairs: return 0.0, []
    
    # Arrhenius scaling (Ea ~ 85 kJ/mol) + pH/moisture sensitivity
    Ea, R, T_K = 85000.0, 8.314, T_celsius + 273.15
    arr = float(np.exp(Ea / R * (1.0 / 298.15 - 1.0 / T_K)))
    
    # Transesterification is base-catalysed (pH > 7) and moisture-sensitive
    ph_factor = float(np.clip(1.0 + (pH - 7.0), 0.5, 5.0))
    moist_factor = 1.0 + 2.0 * moisture
    
    voi = alcohol_density * arr * ph_factor * moist_factor
    return float(np.log1p(voi)), pairs

# ── Complexation ──────────────────────────────────────────────────────────────
_COMPLEXATION_API_SMARTS: Dict[str, Tuple[str, float]] = {
    'aromatic_ring':  ('c1ccccc1',              1.0),
    'carbonyl':       ('[CX3](=[OX1])',          0.7),
    'h_bond_donor':   ('[OH,NH,NH2]',            0.8),
    'halogen_arom':   ('[F,Cl,Br]c',             0.5),
    'hydrophobic_lg': ('CCCC',                   0.4),
}
_COMPLEXATION_EXC_SMARTS: Dict[str, Tuple[str, float]] = {
    'cyclodextrin':   ('[C@@H]1O[C@@H]([C@H](O)[C@@H]1O)CO', 2.0),
    'pvp_lactam':     ('O=C1CCCN1',              1.5),
    'hpmc_ether':     ('COC1OC(CO)C(O)C1O',      1.2),
    'chitosan_amine': ('[NH2][C@@H]1OC(O)[C@H](O)C1', 1.3),
}
_CMPLX_API_COMP = {k:(Chem.MolFromSmarts(v[0]),v[1]) for k,v in _COMPLEXATION_API_SMARTS.items()}
_CMPLX_EXC_COMP = {k:(Chem.MolFromSmarts(v[0]),v[1]) for k,v in _COMPLEXATION_EXC_SMARTS.items()}

def compute_complexation_voi(api_smiles: Optional[str], exc_smiles: Optional[str], 
                             T_celsius: float = 25.0, pH: float = 7.0) -> Tuple[float, List[str]]:
    api_mol = Chem.MolFromSmiles(api_smiles) if api_smiles else None
    exc_mol = Chem.MolFromSmiles(exc_smiles) if exc_smiles else None
    if api_mol is None or exc_mol is None: return 0.0, []
    api_score = min(sum(w for k,(p,w) in _CMPLX_API_COMP.items() if p and api_mol.HasSubstructMatch(p)), 3.0)
    exc_score = min(sum(w for k,(p,w) in _CMPLX_EXC_COMP.items() if p and exc_mol.HasSubstructMatch(p)), 3.0)
    if api_score < 0.1 or exc_score < 0.1: return 0.0, []
    temp_factor = float(np.exp(-20000/8.314 * (1/298.15 - 1/(T_celsius+273.15))))
    voi = api_score * exc_score * temp_factor
    groups = [k for k,(p,w) in _CMPLX_API_COMP.items() if p and api_mol.HasSubstructMatch(p)]
    return float(np.log1p(voi)), groups

# ── Surface Adsorption ────────────────────────────────────────────────────────
_HIGH_SA_EXCIPIENT_SMARTS = {
    'silica':         ('[Si]([O])([O])[O]',  3.0),
    'talc':           ('[Mg][Si]',           2.0),
    'titanium_di':    ('[Ti]([O])[O]',       1.5),
    'cellulose_poly': ('OC1OC(CO)C(O)C1O',  1.0),
    'starch_poly':    ('C1OC(O)C(O)C(O)C1', 0.8),
}
_ADSORP_API_SMARTS = {
    'cationic_amine': ('[NX3;H2,H1,H0;!$(NC=O)]', 1.2),
    'anionic_acid':   ('[CX3](=O)[OX2H]',          1.0),
    'hydrophobic_lg': ('CCCCC',                     0.8),
    'aromatic_flat':  ('c1ccc2ccccc2c1',            1.0),
}
_ADSA_SA_COMP = {k:(Chem.MolFromSmarts(v[0]),v[1]) for k,v in _HIGH_SA_EXCIPIENT_SMARTS.items()}
_ADSA_AP_COMP = {k:(Chem.MolFromSmarts(v[0]),v[1]) for k,v in _ADSORP_API_SMARTS.items()}

def compute_adsorption_voi(api_smiles: Optional[str], exc_smiles: Optional[str], pH: float = 7.0) -> Tuple[float, List[str]]:
    api_mol = Chem.MolFromSmiles(api_smiles) if api_smiles else None
    exc_mol = Chem.MolFromSmiles(exc_smiles) if exc_smiles else None
    if api_mol is None or exc_mol is None: return 0.0, []
    api_score = min(sum(w for k,(p,w) in _ADSA_AP_COMP.items() if p and api_mol.HasSubstructMatch(p)), 3.0)
    exc_score = min(sum(w for k,(p,w) in _ADSA_SA_COMP.items() if p and exc_mol.HasSubstructMatch(p)), 3.0)
    if api_score < 0.1 or exc_score < 0.1: return 0.0, []
    pH_factor = 1.0 + 0.3 * abs(pH - 7.0) / 7.0
    voi = api_score * exc_score * pH_factor
    groups = [k for k,(p,w) in _ADSA_AP_COMP.items() if p and api_mol.HasSubstructMatch(p)]
    return float(np.log1p(voi)), groups

# ── Eutectic Formation ────────────────────────────────────────────────────────
_EUTECTIC_SMARTS = {
    'fatty_acid':   ('CCCCCCCCCCCC(=O)O', 1.5),
    'peg_chain':    ('CCOCCO',            1.2),
    'poloxamer':    ('CCOCCOCCO',         1.0),
    'glycerides':   ('OCC(O)CO',          0.8),
}
_EUTECTIC_COMP = {k:(Chem.MolFromSmarts(v[0]),v[1]) for k,v in _EUTECTIC_SMARTS.items()}

def compute_eutectic_voi(api_smiles: Optional[str], exc_smiles: Optional[str], T_celsius: float = 25.0) -> Tuple[float, List[str]]:
    api_mol = Chem.MolFromSmiles(api_smiles) if api_smiles else None
    exc_mol = Chem.MolFromSmiles(exc_smiles) if exc_smiles else None
    if api_mol is None or exc_mol is None: return 0.0, []
    exc_score = min(sum(w for k,(p,w) in _EUTECTIC_COMP.items() if p and exc_mol.HasSubstructMatch(p)), 3.0)
    if exc_score < 0.1: return 0.0, []
    _, _, Tg_C, _ = classify_gfa(api_smiles)
    Tm_api   = Tg_C * 1.5 + 273.15
    T_eut_C  = 0.85 * (Tm_api - 273.15)
    proximity = float(1.0 / (1.0 + np.exp(-2.0 * (T_celsius - T_eut_C))))
    voi = exc_score * proximity
    return float(np.log1p(voi)), [k for k,(p,w) in _EUTECTIC_COMP.items()
                                   if p and exc_mol.HasSubstructMatch(p)]

def classify_salt(smiles: str) -> bool:
    """Detection of salt forms."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return '.' in smiles if smiles else False

def _compute_disprop_voi(api_smiles, exc_smiles, pH=7.0):
    """Simplified salt disproportionation risk."""
    if not api_smiles or not exc_smiles: return 0.0, []
    is_salt = classify_salt(api_smiles)
    if not is_salt: return 0.0, []
    # pH sensitivity: Basic salt at low pH or Acidic salt at high pH
    risk = abs(pH - 7.0) / 7.0
    return float(np.log1p(risk)), ['salt_form']

def compute_voxel_overlap_integral(system: Any) -> Dict[str, float]:
    """
    Core VOI Integration: Cross-correlates all molecules in the voxel space 
    across 14 physical/chemical channels to identify localized instability.
    """
    env = system.env_conditions
    m_v, h_v, o_v, a_v, d_v = 0.0, 0.0, 0.0, 0.0, 0.0
    n_voi, p_voi, poly_voi, metal_voi = 0.0, 0.0, 0.0, 0.0
    te_voi, cmplx_voi, adsp_voi, eut_voi = 0.0, 0.0, 0.0, 0.0
    
    for api in system.apis:
        api_smi = getattr(api, 'smiles', None)
        for exc in system.excipients:
            exc_smi = getattr(exc, 'smiles', None)
            
            # Metal Ox
            v, _, _ = compute_metal_ox_voi(api_smi, exc_smi, T_celsius=env.temperature, oxygen_level=env.oxygen_level, moisture=env.moisture)
            metal_voi += v
            
            # Transesterification
            v, _ = compute_transester_voi(api_smi, exc_smi, T_celsius=env.temperature, pH=env.pH, moisture=env.moisture)
            te_voi += v
            
            # Complexation
            v, _ = compute_complexation_voi(api_smi, exc_smi, T_celsius=env.temperature, pH=env.pH)
            cmplx_voi += v
            
            # Adsorption
            v, _ = compute_adsorption_voi(api_smi, exc_smi, pH=env.pH)
            adsp_voi += v
            
            # Eutectic
            v, _ = compute_eutectic_voi(api_smi, exc_smi, T_celsius=env.temperature)
            eut_voi += v
            
            # Placeholder for others (Maillard, Hydrolysis, etc. normally use field integration)
            # For this refactor, we assume these are populated via field-based methods in KeyBoxSystem
            
    return {
        'MetalOx_VOI': float(metal_voi),
        'Transester_VOI': float(te_voi),
        'Complexation_VOI': float(cmplx_voi),
        'Adsorption_VOI': float(adsp_voi),
        'Eutectic_VOI': float(eut_voi),
        'Maillard_VOI': 0.0, # Field-based
        'Hydrolysis_VOI': 0.0, # Field-based
        'Oxidation_VOI': 0.0, # Field-based
        'AcidBase_VOI': 0.0, # Field-based
        'Disproportionation_VOI': 0.0, # Field-based
        'NuclAdd_VOI': 0.0,
        'Photo_VOI': 0.0,
        'Polymorph_VOI': 0.0
    }

def detect_enhanced_incompatibility_mechanisms(system: Any, voi: Dict[str, float], metrics: Dict[str, float]) -> List[str]:
    """Identifies active chemical degradation pathways from VOI signals."""
    mechs = []
    def get_conf(v, t): return 1.0 / (1.0 + np.exp(-4.0 * (v/t - 1.2)))
    
    thresholds = [
        ('Maillard_VOI', 0.1, 'Maillard Risk'),
        ('Hydrolysis_VOI', 0.05, 'Hydrolysis Risk'),
        ('Oxidation_VOI', 0.1, 'Oxidation Risk'),
        ('MetalOx_VOI', 0.1, 'Metal-Catalysed Oxidation Risk'),
        ('Transester_VOI', 0.1, 'Transesterification Risk'),
        ('Complexation_VOI', 0.1, 'Complexation Risk'),
        ('Adsorption_VOI', 0.1, 'Surface Adsorption Risk'),
        ('Eutectic_VOI', 0.1, 'Eutectic Formation Risk'),
    ]
    
    for key, thresh, label in thresholds:
        v = voi.get(key, 0.0)
        if v > thresh:
            conf = get_conf(v, thresh * 5.0)
            mechs.append(f"{label} (VOI={v:.3f}, Conf={conf:.1%})")
            
    if metrics.get('SCI', 0) > 0.02: 
        mechs.append(f"Steric Clash (SCI={metrics['SCI']:.3f}, Conf={get_conf(metrics['SCI'], 0.05):.1%})")
        
    return mechs if mechs else ['No specific mechanism detected']

def get_stability_recommendations(system: Any, voi: Dict[str, float], metrics: Dict[str, float]) -> List[str]:
    """Generates expert formulation advice based on detected risks."""
    recs, env = [], system.env_conditions
    ocs = float(metrics.get('OCS', 50.0))
    
    severity = 'INFO' if ocs>=80 else ('ADVISORY' if ocs>=60 else ('WARNING' if ocs>=40 else 'CRITICAL'))
    status = "compatible" if ocs>=80 else ("marginal" if ocs>=60 else ("concern" if ocs>=40 else "high risk"))
    recs.append(f"[{severity}] OCS={ocs:.1f} - {status}")
    
    if voi.get('MetalOx_VOI', 0) > 0.05:
        recs.append(f"[WARNING] Metal-catalysed oxidation risk (VOI={voi['MetalOx_VOI']:.3f}). TIP: use chelating agents (EDTA, Citrate).")
        
    if voi.get('Transester_VOI', 0) > 0.1:
        recs.append(f"[WARNING] Transesterification risk detected. Avoid PEG/polyol contact with ester APIs.")
        
    if voi.get('Complexation_VOI', 0) > 0.1:
        recs.append(f"[ADVISORY] Complexation risk (VOI={voi['Complexation_VOI']:.3f}): API-excipient complex may alter dissolution.")

    if voi.get('Adsorption_VOI', 0) > 0.1:
        recs.append(f"[ADVISORY] Surface Adsorption risk: Specify low-surface-area grades or add surfactant.")

    if voi.get('Eutectic_VOI', 0) > 0.1:
        recs.append(f"[WARNING] Eutectic Formation risk: Perform DSC phase screening.")

    return recs if recs else ["[INFO] No critical risks identified."]

# Disproportionation constants
def classify_salt(smiles: str) -> Dict:
    frags = smiles.split('.')
    if len(frags) < 2: return {'is_salt': False, 'salt_type': None, 'api_smiles': smiles, 'pKa_estimate': 7.0, 'counterion_atoms': [], 'logP': 0.0}
    frag_mols = [(f, Chem.MolFromSmiles(f)) for f in frags if Chem.MolFromSmiles(f) is not None]
    if not frag_mols: return {'is_salt': False, 'salt_type': None, 'api_smiles': smiles, 'pKa_estimate': 7.0, 'counterion_atoms': [], 'logP': 0.0}
    api_smi, api_mol = max(frag_mols, key=lambda x: x[1].GetNumHeavyAtoms())
    ci_atoms = set()
    for s, m in frag_mols:
        if s != api_smi:
            for a in m.GetAtoms(): ci_atoms.add(a.GetSymbol())
    basic_ci, acidic_ci = (ci_atoms & {'Cl', 'Br', 'I', 'S', 'P', 'F'}), (ci_atoms & {'Na', 'K', 'Ca', 'Mg', 'Li', 'Zn', 'Al'})
    salt_type = 'basic_salt' if basic_ci else ('acidic_salt' if acidic_ci else 'zwitterion_or_unknown')
    logP = float(Descriptors.MolLogP(api_mol))
    base_pat, acid_pat = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O);!$(N/C=C/O)]'), Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    has_base, has_acid = (base_pat and api_mol.HasSubstructMatch(base_pat)), (acid_pat and api_mol.HasSubstructMatch(acid_pat))
    pKa = (9.0 - 0.4 * max(logP, 0.0)) if (salt_type == 'basic_salt' and has_base) else ((4.0 + 0.3 * logP) if (salt_type == 'acidic_salt' and has_acid) else 7.0)
    return {'is_salt': True, 'salt_type': salt_type, 'counterion_atoms': list(ci_atoms), 'api_smiles': api_smi, 'pKa_estimate': pKa, 'logP': logP}

def _compute_disprop_voi(salt_info: Dict, pH_micro: float, exc_base_density: float, exc_acid_density: float,
                         T_celsius: float = 25.0) -> Tuple[float, float]:
    if not salt_info.get('is_salt'): return 0.0, 0.0
    pKa, T_K, Ea, R = salt_info['pKa_estimate'], T_celsius + 273.15, 60000.0, 8.314
    arr = float(np.exp(Ea / R * (1.0 / 298.15 - 1.0 / T_K)))
    st = salt_info.get('salt_type', '')
    if st == 'basic_salt':
        risk = float(1.0 / (1.0 + np.exp(-2.0 * (pH_micro - pKa))))
        voi = risk * exc_base_density * arr
    elif st == 'acidic_salt':
        risk = float(1.0 / (1.0 + np.exp(-2.0 * (pKa - pH_micro))))
        voi = risk * exc_acid_density * arr
    else: risk, voi = 0.0, 0.0
    return float(voi), float(risk)
