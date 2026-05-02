"""
KeyBox Engine (v0.9.3 - Modularized)
------------------------------------
This module serves as the primary orchestrator for the KeyBox voxel physics engine.
It handles multi-molecule superposition, physical field projection, and routes computations
to specialized modules (thermodynamics, degradation mechanisms, and the Nibble docking engine).
It evaluates both API-excipient compatibility and structural protein-ligand docking affinity.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy import ndimage
from scipy.spatial import cKDTree
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFreeSASA

from .models import (
    Molecule, MoleculeType, DosageForm, EnvironmentalConditions, 
    DosageFormConfig, DosageFormResult
)
from .nibble_bridge import NibbleEngine
from .voxel_grid import EnhancedVoxelGrid
from .thermodynamics import (
    compute_flory_huggins_phase_behavior,
    compute_dissolution_profile,
    compute_reaction_integrals,
    monte_carlo_uncertainty_analysis
)
from .dosage_forms import run_dosage_form_physics
from .physics import (
    compute_hansen_chi, ProcessPhysics, 
    MicrostructurePhysics, TemporalPhysics,
    DosageSpecificPhysics
)
from .mechanisms import (
    compute_metal_ox_voi, compute_polymorph_voi,
    compute_photo_voi, compute_nucl_add_voi,
    compute_transester_voi, compute_complexation_voi,
    compute_adsorption_voi, compute_eutectic_voi,
    classify_salt, _compute_disprop_voi,
    _LUBRICANT_PATTERNS, _METAL_ION_SMARTS,
    _METAL_PRONE_EXCIPIENT_SUBSTRINGS
)

class KeyBoxSystem:
    def __init__(self, voxel_bounds: Tuple[float, float, float] = (20.0, 20.0, 20.0),
                 resolution: float = 1.0,
                 dosage_config: Optional[DosageFormConfig] = None,
                 env_conditions: Optional[EnvironmentalConditions] = None,
                 random_state: Optional[int] = None):
        self.dosage_config = dosage_config or DosageFormConfig(DosageForm.ORAL_SOLID)
        self.random_state = random_state
        if random_state is not None: np.random.seed(random_state)
        self.voxel_grid = EnhancedVoxelGrid(voxel_bounds, resolution, self.dosage_config)
        self.molecules = []; self.apis = []; self.excipients = []
        self.interaction_metrics = {}
        self.env_conditions = env_conditions or EnvironmentalConditions()
        self.gaussian_width = 2.0; self.coulomb_constant = 332.0
        self.cutoff_distances = {
            'electrostatic': 10.0, 'steric': 5.0, 'h_bond': 4.0, 'hydrophobic': 6.0, 'default': 5.0
        }
        self.dielectric_scaling = self.env_conditions.dielectric_constant / 78.5
        self.phase_diffusion_rates = {0: 1.0, 1: 0.1, 2: 0.01}
        self.compatibility_thresholds = {'compatible': 60, 'partially_compatible': 40, 'incompatible': 0}
        self.mechanism_thresholds = {
            'maillard_reaction': {'sci': 0.12, 'temp': 35, 'moisture': 0.4},
            'hydrolysis': {'moisture': 0.55, 'hbp': 0.25, 'pH_range': (2, 12)},
            'oxidation': {'ec': 0.35, 'reactivity': 0.45, 'oxygen': 0.15},
            'polymorphic_change': {'sci': 0.18, 'temp': 25, 'pressure': 0.8},
            'physical_degradation': {'sci': 0.22, 'time': 6},
            'chemical_reaction': {'reactivity': 0.55, 'ec': 0.4, 'temp': 30},
            'phase_separation': {'ha': 0.15, 'multi_phase': True},
            'aggregation': {'sci': 0.3, 'protein': True},
            'precipitation': {'solvent_access': 0.2, 'ionic_strength': 0.1},
            'complexation': {'ec': 0.6, 'hbp': 0.4}
        }
        self.validation_data = {}; self.confidence_intervals = {}
        self.ocs_weights = {
            'ri':  0.55, 'sci': 0.08, 'ha':  0.12, 'ec':  0.05,
            'hbp': 5.0, 'voi_cap': 20.0, 'voi_scale': 10.0
        }
        self.calibration_factors = {'ec': 1.0, 'ha': 1.0, 'hbp': 1.0, 'sci': 1.0}
    
    def detect_functional_groups_rdkit(self, smiles: str) -> Dict[str, int]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return {}
        patterns = {
            'primary_amine': '[NX3;H2;!$(NC=O)]', 'secondary_amine': '[NX3;H1;!$(NC=O)]',
            'tertiary_amine': '[NX3;H0;!$(NC=O)]', 'carbonyl': '[CX3]=[OX1]',
            'carboxylic_acid': '[CX3](=O)[OX2H1]', 'ester': '[#6][CX3](=O)[OX2H0][#6]',
            'amide': '[NX3][CX3](=[OX1])[#6]', 'phenol': '[OX2H][c]',
            'alcohol': '[OX2H][CX4]', 'thiol': '[SX2H]', 'aldehyde': '[CX3H1](=O)[#6]',
            'ketone': '[#6][CX3](=O)[#6]', 'nitro': '[NX3](=O)=O', 'nitrile': '[NX1]#[CX2]',
            'epoxide': '[OX2r3]1[#6][#6]1', 'peroxide': '[OX2][OX2]', 'lactone': '[#6][CX3](=O)[OX2][#6]',
            'lactam': '[NX3][CX3](=O)[#6]', 'base_strong': '[$(N[C,H](=[N,H])[N,H]),$(n1[c,n][c,n][c,n][c,n]1)]',
            'alkali_metal': '[Li,Na,K,Rb,Cs].[#8-1]', 'reducing_sugar': '[C;R;H1]([OH])[O;R]'
        }
        group_counts = {}
        for name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern: group_counts[name] = len(mol.GetSubstructMatches(pattern))
        return group_counts
    
    def auto_detect_incompatibilities(self, mol1_smiles: str, mol2_smiles: str) -> List[Dict[str, any]]:
        groups1, groups2 = self.detect_functional_groups_rdkit(mol1_smiles), self.detect_functional_groups_rdkit(mol2_smiles)
        incompatibilities = []
        amines_1, amines_2 = groups1.get('primary_amine', 0) + groups1.get('secondary_amine', 0), groups2.get('primary_amine', 0) + groups2.get('secondary_amine', 0)
        carbonyls_1, carbonyls_2 = groups1.get('carbonyl', 0) + groups1.get('aldehyde', 0), groups2.get('carbonyl', 0) + groups2.get('aldehyde', 0)
        if (amines_1 > 0 and carbonyls_2 > 0) or (amines_2 > 0 and carbonyls_1 > 0):
            incompatibilities.append({'mechanism': 'Maillard Reaction', 'severity': 'CRITICAL', 'groups_involved': ['amine', 'carbonyl'], 'recommendation': 'Avoid moisture >40% RH, store <25C', 'auto_voi_flag': True, 'confidence': min(1.0, (amines_1 + amines_2 + carbonyls_1 + carbonyls_2) / 10.0)})
        esters_1, esters_2 = groups1.get('ester', 0) + groups1.get('lactone', 0), groups2.get('ester', 0) + groups2.get('lactone', 0)
        if esters_1 > 0 or esters_2 > 0:
            incompatibilities.append({'mechanism': 'Ester Hydrolysis', 'severity': 'HIGH', 'groups_involved': ['ester'], 'recommendation': 'Control moisture <30% RH, avoid pH extremes', 'auto_voi_flag': True, 'confidence': min(1.0, (esters_1 + esters_2) / 5.0)})
        return incompatibilities

    def add_molecule(self, molecule: Molecule, is_api: bool = False):
        if len(self.molecules) > 0: molecule = self.optimize_molecular_orientation(molecule, strategy='minimize_clash', steps=50)
        self.molecules.append(molecule)
        if is_api: self.apis.append(molecule)
        else: self.excipients.append(molecule)

    def compute_enhanced_field_kernel(self, distances: np.ndarray, field_type: str, property_value: float, phase_id: int = 0) -> np.ndarray:
        cutoff = self.cutoff_distances.get(field_type, self.cutoff_distances['default'])
        mask = distances <= cutoff; field = np.zeros_like(distances); phase_factor = self.phase_diffusion_rates.get(phase_id, 1.0)
        if field_type == 'electrostatic':
            kappa = np.sqrt(self.env_conditions.ionic_strength * 1000); epsilon_r = self.env_conditions.dielectric_constant; r_safe = distances[mask] + 0.1
            field[mask] = (self.coulomb_constant * property_value * self.dielectric_scaling / epsilon_r) * (np.exp(-kappa * r_safe) / r_safe)
        elif field_type == 'steric':
            sigma = property_value; epsilon_lj = 0.5; r_safe = np.maximum(distances[mask], 0.5) + 0.1; sigma_over_r = sigma / r_safe
            field[mask] = phase_factor * (4 * epsilon_lj * (sigma_over_r**12 - sigma_over_r**6)) / epsilon_lj
        elif field_type == 'polarizability': field[mask] = property_value * phase_factor / ((distances[mask] + 0.5)**6 + 1.0)
        elif field_type == 'reactive_sites': field[mask] = property_value * np.exp(-distances[mask]**2 / (2 * self.gaussian_width**2))
        elif field_type in ['h_donor', 'h_acceptor']: field[mask] = property_value * phase_factor * np.exp(-distances[mask])
        elif field_type in ['hydrophobic', 'lipid_density', 'viscosity']: field[mask] = property_value * phase_factor * np.exp(-distances[mask]**2 / (2 * self.gaussian_width**2))
        return field

    def project_enhanced_molecule_fields(self, molecule: Molecule) -> Dict[str, np.ndarray]:
        confs = molecule.conformations if molecule.conformations else [molecule.coords]
        weights = list(molecule.taut_weights) if molecule.taut_weights and len(molecule.taut_weights) == len(confs) else [1.0 / len(confs)] * len(confs)
        def _resolve(attr, fallback):
            val = getattr(molecule, attr, None)
            if val and len(val) == len(confs): return val
            if fallback is None: fallback = np.zeros(len(molecule.coords))
            return [fallback] * len(confs)
        t_ch, t_hd, t_ha, t_am, t_cb, t_ac, t_ba = _resolve('taut_charges', molecule.charges), _resolve('taut_h_donors', molecule.h_donors), _resolve('taut_h_acceptors', molecule.h_acceptors), _resolve('taut_amine_groups', molecule.amine_groups), _resolve('taut_carbonyl_groups', molecule.carbonyl_groups), _resolve('taut_acid_groups', molecule.acid_groups), _resolve('taut_base_groups', molecule.base_groups)
        is_l, is_p = molecule.mol_type in [MoleculeType.LIPID, MoleculeType.SURFACTANT], molecule.mol_type == MoleculeType.POLYMER
        visc_f, lipid_f = (0.8 if is_p else (0.4 if is_l else 0.1)), (1.0 if is_l else (0.5 if molecule.mol_type == MoleculeType.SMALL_MOLECULE and np.mean(molecule.hydrophobicity) > 1.0 else 0.0))
        pKa, name = 4.5, molecule.name.lower()
        if 'citric' in name: pKa = 3.1
        elif 'base' in name or 'amine' in name: pKa = 9.5
        ph_shift = float(np.clip((pKa - self.env_conditions.pH) * molecule.concentration / 1.1, -3.0, 3.0))
        sigma2_2, kappa, eps_r = 2.0 * self.gaussian_width ** 2, float(np.sqrt(max(self.env_conditions.ionic_strength * 1000, 0.0))), float(self.env_conditions.dielectric_constant)
        coulomb_k, epsilon_lj = self.coulomb_constant * self.dielectric_scaling / eps_r, 0.5
        VC, tree, N = self.voxel_grid._voxel_coords_flat, self.voxel_grid._voxel_tree, len(self.voxel_grid._voxel_coords_flat)
        GLOBAL_CUTOFF, USE_SPARSE = 10.0, N >= 8000
        f_es, f_hy, f_st, f_hd, f_ha, f_pol, f_rx, f_lip, f_vis, f_ph, f_acid, f_base, f_ami, f_car, f_met, f_per = [np.zeros(N) for _ in range(16)]
        
        # Detect metal ions and peroxides for field projection
        mol_rd = Chem.MolFromSmiles(molecule.smiles) if getattr(molecule, 'smiles', None) else None
        has_met = any(p is not None and mol_rd.HasSubstructMatch(p) for p in _METAL_ION_SMARTS.values()) if mol_rd else False
        has_per = mol_rd.HasSubstructMatch(Chem.MolFromSmarts('[OX2][OX2]')) if mol_rd else False
        is_met_prone = any(s in molecule.smiles for s in _METAL_PRONE_EXCIPIENT_SUBSTRINGS) if getattr(molecule, 'smiles', None) else False
        
        for t_idx, (conf, w) in enumerate(zip(confs, weights)):
            if USE_SPARSE:
                sdm = tree.sparse_distance_matrix(cKDTree(conf.astype(np.float64)), GLOBAL_CUTOFF, output_type='coo_matrix')
                vi, ai, d = sdm.row.astype(np.int32), sdm.col.astype(np.int32), sdm.data
            else:
                dist_mat = np.sqrt(((VC[:, None, :] - conf[None, :, :])**2).sum(axis=2))
                vi_2d, ai_2d = np.where(dist_mat <= GLOBAL_CUTOFF)
                vi, ai, d = vi_2d.astype(np.int32), ai_2d.astype(np.int32), dist_mat[vi_2d, ai_2d]
            d2 = d * d; m6, m5, m4 = d <= 6.0, d <= 5.0, d <= 4.0
            phase_factors = np.array([self.phase_diffusion_rates.get(self._get_phase_id(conf[k]), 1.0) for k in range(len(conf))])[ai]
            r_safe = d + 0.1; es_val = w * t_ch[t_idx][ai] * coulomb_k * np.exp(-kappa * r_safe) / r_safe
            np.add.at(f_es, vi, es_val); es_abs = np.abs(es_val)
            for mask_v, field in [(t_ac[t_idx][ai] > 0.5, f_acid), (t_ba[t_idx][ai] > 0.5, f_base), (t_am[t_idx][ai] > 0.5, f_ami), (t_cb[t_idx][ai] > 0.5, f_car)]:
                if mask_v.any(): np.add.at(field, vi[mask_v], es_abs[mask_v])
            if m4.any():
                exp_d4, pf4 = np.exp(-d[m4]), phase_factors[m4]
                np.add.at(f_hd, vi[m4], w * t_hd[t_idx][ai[m4]] * pf4 * exp_d4)
                np.add.at(f_ha, vi[m4], w * t_ha[t_idx][ai[m4]] * pf4 * exp_d4)
            if m5.any():
                rs5, pf5 = np.maximum(d[m5], 0.5) + 0.1, phase_factors[m5]
                np.add.at(f_st, vi[m5], w * pf5 * 4 * epsilon_lj * ((molecule.vdw_radii[ai[m5]] / rs5)**12 - (molecule.vdw_radii[ai[m5]] / rs5)**6) / epsilon_lj)
            if m6.any():
                gauss6, pf6 = np.exp(-d2[m6] / sigma2_2), phase_factors[m6]
                np.add.at(f_hy, vi[m6], w * molecule.hydrophobicity[ai[m6]] * pf6 * gauss6)
                np.add.at(f_lip, vi[m6], w * lipid_f * pf6 * gauss6)
                np.add.at(f_vis, vi[m6], w * visc_f * pf6 * gauss6)
                np.add.at(f_rx, vi[m6], w * molecule.reactive_groups[ai[m6]] * np.exp(-d2[m6] / sigma2_2))
            np.add.at(f_pol, vi, w * molecule.polarizability[ai] * phase_factors / ((d + 0.5)**6 + 1.0))
            if abs(ph_shift) > 0.1: np.add.at(f_ph, vi, w * ph_shift * coulomb_k * np.exp(-kappa * r_safe) / r_safe)
            if has_met or is_met_prone: 
                met_w = 0.8 if has_met else 0.3
                np.add.at(f_met, vi, w * met_w * np.exp(-d2 / sigma2_2))
            if has_per:
                np.add.at(f_per, vi, w * 1.0 * np.exp(-d2 / sigma2_2))
        sh, c = self.voxel_grid.shape, molecule.concentration
        return {'electrostatic': (f_es * c).reshape(sh), 'acid_density': (f_acid * c).reshape(sh), 'base_density': (f_base * c).reshape(sh), 'amine_density': (f_ami * c).reshape(sh), 'carbonyl_density': (f_car * c).reshape(sh), 'hydrophobic': (f_hy * c).reshape(sh), 'steric': (f_st * c).reshape(sh), 'h_donor': (f_hd * c).reshape(sh), 'h_acceptor': (f_ha * c).reshape(sh), 'polarizability': (f_pol * c).reshape(sh), 'reactive_sites': (f_rx * c).reshape(sh), 'lipid_density': (f_lip * c).reshape(sh), 'viscosity': (f_vis * c).reshape(sh), 'ph_field': (f_ph * c).reshape(sh), 'metal_ion': (f_met * c).reshape(sh), 'peroxide_density': (f_per * c).reshape(sh), 'ionic_field': np.zeros(sh)}

    def compute_flory_huggins_phase_behavior(self) -> Dict:
        R, T = 8.314, self.env_conditions.temperature + 273.15
        poly_d, solv_d = self.voxel_grid.viscosity, self.voxel_grid.solvent_accessible
        total = poly_d + solv_d + 1e-10; phi_p, phi_s = poly_d / total, solv_d / total
        hansen_results, N_p_def, N_s_def = [], 100.0, 1.0
        for api in self.apis:
            for exc in self.excipients:
                api_smi, exc_smi = getattr(api, 'smiles', None), getattr(exc, 'smiles', None)
                if api_smi and exc_smi:
                    hr = compute_hansen_chi(api_smi, exc_smi, T_celsius=self.env_conditions.temperature)
                    if hr: hansen_results.append(hr)
        if hansen_results:
            chi_scalar, chi_crit = np.mean([r['chi'] for r in hansen_results]), np.mean([r['chi_critical'] for r in hansen_results])
            N_p_def, N_s_def = np.mean([r['N_p'] for r in hansen_results]), np.mean([r['N_s'] for r in hansen_results])
        else: chi_scalar, chi_crit = (0.5 + 100.0 / T), 0.5 * (1 / np.sqrt(N_p_def) + 1 / np.sqrt(N_s_def))**2
        h_norm = self.voxel_grid.hydrophobic / (np.mean(np.abs(self.voxel_grid.hydrophobic)) + 1e-10)
        chi_field = chi_scalar * (1.0 + 0.1 * np.abs(h_norm - np.mean(h_norm)))
        dg = ((phi_p / N_p_def) * np.log(phi_p + 1e-10) + (phi_s / N_s_def) * np.log(phi_s + 1e-10) + chi_field * phi_p * phi_s) * R * T
        risk = (chi_field > chi_crit).astype(float); labels, n = ndimage.label(risk > 0.5)
        return {'chi': chi_field, 'chi_scalar': float(chi_scalar), 'chi_critical': float(chi_crit), 'dg': dg, 'risk': risk, 'labels': labels, 'n': n, 'hansen_pairs': hansen_results}

    def compute_dissolution_profile(self, time_points: int = 100, total_time: float = 7200.0, particle_d_um: float = 10.0, rho_g_cm3: float = 1.3, V_mL: float = 900.0, mass_mg: float = 500.0, h_cm: float = 0.003) -> Dict:
        api_smiles = next((getattr(api, 'smiles', None) for api in self.apis if getattr(api, 'smiles', None)), None)
        exc_smiles = next((getattr(exc, 'smiles', None) for exc in self.excipients if getattr(exc, 'smiles', None)), None)
        if not api_smiles:
            A_shell = np.sum((self.voxel_grid.steric > 0.1) & ~ndimage.binary_erosion(self.voxel_grid.steric > 0.1)) * (self.voxel_grid.resolution * 1e-8) ** 2
            k, t = (1e-5 * A_shell) / h_cm, np.linspace(0, total_time, time_points); dt = t[1] - t[0]
            M_d, C, M_t = np.zeros(time_points), np.zeros(time_points), np.sum(self.voxel_grid.steric) * 1e-15; M_r = M_t
            for i in range(1, time_points):
                if M_r <= 0: M_d[i], C[i] = M_t, M_t / V_mL; continue
                dm = min(k * max(1.0 - C[i-1], 0) * dt, M_r); M_d[i], M_r = M_d[i-1] + dm, M_r - dm; C[i] = M_d[i] / V_mL
            p = M_d / (M_t + 1e-10) * 100; t50 = float(t[np.argmax(p >= 50)]) if np.max(p) >= 50 else total_time
            return {'time': t, 'pct_dissolved': p, 'T50_s': t50, 'D_cm2s': 1e-5, 'Cs_mg_mL': 1.0, 'A_eff_cm2': float(A_shell), 'f_polar': 0.5, 'wetting_factor': 1.0, 'logS_ESOL': None, 'note': 'Legacy voxel-shell fallback'}
        mol_api = Chem.MolFromSmiles(api_smiles); mw = float(Descriptors.MolWt(mol_api)) if mol_api else 250.0
        T_K, kB, Na, eta = self.env_conditions.temperature + 273.15, 1.380649e-23, 6.022e23, 0.00089
        r_mol = (3 * mw / (4 * np.pi * Na * rho_g_cm3 * 1e6)) ** (1/3)
        D = float(kB * T_K / (6 * np.pi * eta * r_mol) * 1e4)
        logP, rings = (Descriptors.MolLogP(mol_api) if mol_api else 0.0), (mol_api.GetRingInfo().NumRings() if mol_api else 0)
        n_sp3 = sum(1 for a in mol_api.GetAtoms() if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.rdchem.HybridizationType.SP3) if mol_api else 0
        n_c = sum(1 for a in mol_api.GetAtoms() if a.GetAtomicNum() == 6) if mol_api else 1
        logS = 0.16 - 0.63*logP - 0.0062*mw + 0.066*rings - 0.74*(n_sp3/n_c); Cs_n = float(10**logS * mw); Cs, pH = Cs_n, self.env_conditions.pH
        acid_p, base_p = Chem.MolFromSmarts('[CX3](=O)[OX2H1]'), Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
        if mol_api and acid_p and mol_api.HasSubstructMatch(acid_p): Cs = min(Cs_n * (1.0 + 10**(pH - (4.0 + 0.3 * logP))), Cs_n * 100.0)
        elif mol_api and base_p and mol_api.HasSubstructMatch(base_p): Cs = min(Cs_n * (1.0 + 10**((9.0 - 0.4 * max(logP, 0.0)) - pH)), Cs_n * 100.0)
        A_geo, f_polar = (6.0 / (rho_g_cm3 * particle_d_um * 1e-4 * 1000.0)) * mass_mg, 0.5
        try:
            mol_h = Chem.AddHs(mol_api)
            if AllChem.EmbedMolecule(mol_h, randomSeed=42) != -1:
                AllChem.MMFFOptimizeMolecule(mol_h); radii = rdFreeSASA.classifyAtoms(mol_h)
                total_sasa = float(rdFreeSASA.CalcSASA(mol_h, radii))
                f_polar = float(min(Descriptors.TPSA(mol_api) * 1.5, total_sasa * 0.85) / max(total_sasa, 1.0))
        except: pass
        w_f = float(np.clip(1.0 + 0.3*(0.0 - Descriptors.MolLogP(Chem.MolFromSmiles(exc_smiles)))/3.0, 0.5, 2.0)) if exc_smiles and Chem.MolFromSmiles(exc_smiles) else 1.0
        A_eff, k, t = A_geo * f_polar * w_f, (D * A_geo * f_polar * w_f) / h_cm, np.linspace(0, total_time, time_points); dt = t[1] - t[0]
        M_d, C, M_r = np.zeros(time_points), np.zeros(time_points), float(mass_mg)
        for i in range(1, time_points):
            if M_r <= 0: M_d[i], C[i] = mass_mg, mass_mg / V_mL; continue
            dm = min(k * max(Cs - C[i-1], 0.0) * dt, M_r); M_d[i], M_r = M_d[i-1] + dm, M_r - dm; C[i] = M_d[i] / V_mL
        pct = M_d / mass_mg * 100.0; t50_idx = np.argmax(pct >= 50) if np.max(pct) >= 50 else -1
        t90_idx = np.argmax(pct >= 90) if np.max(pct) >= 90 else -1
        return {'time': t, 'pct_dissolved': pct, 'T50_s': float(t[t50_idx]) if t50_idx >= 0 else float(total_time), 'k_diss': float(-np.log(0.1) / (t[t90_idx] + 1.0)) if t90_idx > 0 else 0.0, 'D_cm2s': D, 'Cs_mg_mL': Cs_n, 'Cs_pH_corrected': Cs, 'A_eff_cm2': float(A_eff), 'f_polar': f_polar, 'wetting_factor': w_f, 'logS_ESOL': float(logS), 'particle_d_um': particle_d_um}

    def _get_phase_id(self, coord):
        idx = tuple(np.clip(((coord)/self.voxel_grid.resolution).astype(int), 0, np.array(self.voxel_grid.shape)-1))
        return 1 if self.voxel_grid.lipid_density[idx] > 0.3 else (2 if self.voxel_grid.steric[idx] > 0.5 else 0)

    def compute_temporal_descriptors(self, n=10):
        if not self.dosage_config.release_kinetics: return {}
        D, dx, res = 1e-6 * 1e16, self.voxel_grid.resolution, {}; dt = min(3600.0/n, (dx**2)/(6*D)*0.5); C = self.voxel_grid.steric.copy()
        def lap(f): return (ndimage.convolve(f, np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,-6,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]])) / dx**2)
        for t in range(n): C = np.maximum(C + D*dt*lap(C), 0); res[f't_{t}'] = {'C': C.copy(), 'time': t*dt}
        return res

    def compute_multi_molecule_superposition(self):
        for a in ['electrostatic', 'hydrophobic', 'steric', 'h_donor', 'h_acceptor', 'polarizability', 'solvent_accessible', 'reactive_sites', 'acid_density', 'base_density', 'acid_density_api', 'base_density_api', 'acid_density_exc', 'base_density_exc', 'amine_density_api', 'carbonyl_density_api', 'amine_density_exc', 'carbonyl_density_exc', 'ph_field']:
            getattr(self.voxel_grid, a).fill(0)
        apis = {m.name for m in self.apis}
        for mol in self.molecules:
            f, is_a = self.project_enhanced_molecule_fields(mol), mol.name in apis
            for k, v in f.items():
                if hasattr(self.voxel_grid, k): getattr(self.voxel_grid, k).__iadd__(v)
                if is_a and hasattr(self.voxel_grid, f'{k}_api'): getattr(self.voxel_grid, f'{k}_api').__iadd__(v)
                if not is_a and hasattr(self.voxel_grid, f'{k}_exc'): getattr(self.voxel_grid, f'{k}_exc').__iadd__(v)
        self._compute_solvent_accessibility()

    def _compute_solvent_accessibility(self):
        mask = (self.voxel_grid.steric < np.percentile(self.voxel_grid.steric[self.voxel_grid.steric>0], 25)) if np.any(self.voxel_grid.steric>0) else np.ones(self.voxel_grid.shape, dtype=bool)
        self.voxel_grid.solvent_accessible = mask.astype(float) * (1.0 + self.env_conditions.moisture)

    def compute_enhanced_interaction_metrics(self) -> Dict:
        vg, total = self.voxel_grid, np.prod(self.voxel_grid.shape); vsi = np.sum(vg.steric > 0.05) / total
        sci = np.sum(vg.steric > (np.percentile(vg.steric[vg.steric>0],80) if np.any(vg.steric>0) else 1.0)) / total
        ec = min(np.sum(vg.electrostatic > 0.1), np.sum(vg.electrostatic < -0.1)) / max(np.sum(np.abs(vg.electrostatic)>0.1), 1)
        ha, hbp = (min(1.0, np.mean(vg.hydrophobic[vg.hydrophobic>0])) if np.any(vg.hydrophobic>0) else 0), min(1.0, np.sum((vg.h_donor>0.1) & (vg.h_acceptor>0.1)) / (np.sum(vg.steric>0.1)+1e-6) * 2.0)
        voi = self.compute_voxel_overlap_integral(); self._last_voi, ri = voi, 1.0 - np.exp(-(sum(voi.values())))
        sai, vi, w = np.mean(vg.solvent_accessible), (min(1.0, np.mean(vg.viscosity[vg.steric>0.1])) if np.any(vg.steric>0.1) else 0), self.ocs_weights
        p_score = (100.0 - 100.0 * (w['ri'] * ri + w['sci'] * sci + w['ha'] * ha + w['ec'] * ec) + w['hbp'] * hbp - min(w['voi_cap'], voi.get('AcidBase_VOI', 0.0) / w['voi_scale']))
        lpi = 0.0
        if self.apis and self.excipients:
            b_s, l_s, r_s = Chem.MolFromSmarts('NC(=N)NC(=N)N'), Chem.MolFromSmarts('[Mg+2,Ca+2,Al+3,Zn+2]'), Chem.MolFromSmarts('[C;R;H1]([OH])[O;R]')
            for a in self.apis:
                for e in self.excipients:
                    if not (hasattr(a, 'smiles') and hasattr(e, 'smiles')): continue
                    ag, eg, c_AB = self.detect_functional_groups_rdkit(a.smiles), self.detect_functional_groups_rdkit(e.smiles), 1.0 - np.exp(-voi.get('AcidBase_VOI', 0))
                    arr, moist = np.exp(80000.0 / 8.314 * (1/298.15 - 1/(self.env_conditions.temperature + 273.15))), max(0.1, self.env_conditions.moisture / 0.4)
                    c_M = 1.0 - np.exp(-voi.get('Maillard_VOI', 0) * arr * moist)
                    if (ag.get('ester', 0) or ag.get('lactone', 0)) and ((Chem.MolFromSmiles(e.smiles) and b_s and Chem.MolFromSmiles(e.smiles).HasSubstructMatch(l_s)) or eg.get('base_strong', 0)): lpi += 62 * c_AB
                    if ag.get('primary_amine', 0) and ((Chem.MolFromSmiles(e.smiles) and r_s and Chem.MolFromSmiles(e.smiles).HasSubstructMatch(r_s)) or eg.get('reducing_sugar', 0)):
                        if not (Chem.MolFromSmiles(a.smiles) and b_s and Chem.MolFromSmiles(a.smiles).HasSubstructMatch(b_s)) or self.env_conditions.temperature > 30: lpi += 35 * c_M
        lpi += float(np.clip(voi.get('Disproportionation_VOI', 0.0) * 20.0, 0.0, 30.0)) + \
               float(np.clip(voi.get('NuclAdd_VOI', 0.0) * 15.0, 0.0, 25.0)) + \
               float(np.clip(voi.get('Photo_VOI', 0.0) * 12.0, 0.0, 20.0)) + \
               float(np.clip(voi.get('Polymorph_VOI', 0.0) * 10.0, 0.0, 15.0)) + \
               float(np.clip(voi.get('MetalOx_VOI', 0.0) * 15.0, 0.0, 20.0)) + \
               float(np.clip(voi.get('Transester_VOI', 0.0) * 10.0, 0.0, 18.0)) + \
               float(np.clip(voi.get('Complexation_VOI', 0.0) * 8.0, 0.0, 10.0)) + \
               float(np.clip(voi.get('Adsorption_VOI', 0.0) * 6.0, 0.0, 8.0)) + \
               float(np.clip(voi.get('Eutectic_VOI', 0.0) * 8.0, 0.0, 10.0))
        ocs = max(0.0, min(100.0, p_score - lpi)); res = {'VSI': vsi, 'SCI': sci, 'EC': ec, 'HA': ha, 'HBP': hbp, 'SAI': sai, 'RI': ri, 'LPI': lpi, 'VI': vi, 'OCS': ocs}
        res.update(voi); return res

    def compute_voxel_overlap_integral(self) -> Dict:
        R, T, tc, vg = 1.987e-3, self.env_conditions.temperature + 273.15, sum(m.concentration for m in self.molecules) + 1e-10, self.voxel_grid
        m_v = (np.sum(vg.amine_density_api * vg.carbonyl_density_exc + vg.amine_density_exc * vg.carbonyl_density_api) * np.exp(-4.0/(R*T)) * 10) / tc
        h_v = (np.sum(self.env_conditions.moisture * vg.reactive_sites) * np.exp(-5.0/(R*T)) * 5) / tc
        o_v = (np.sum(self.env_conditions.oxygen_level * vg.reactive_sites) * np.exp(-5.0/(R*T)) * 5) / tc
        a_v = (np.sum(vg.acid_density_api * vg.base_density_exc + vg.acid_density_exc * vg.base_density_api) * np.exp(-1.5/(R*T)) * 50) / tc
        e_b_s, e_a_s = float(np.sum(vg.base_density_exc)), float(np.sum(vg.acid_density_exc))
        pH_m, d_v = self.env_conditions.pH + float(np.clip(e_b_s - e_a_s, -3.0, 3.0)), 0.0
        for api in self.apis:
            if getattr(api, 'smiles', None): d_v += _compute_disprop_voi(classify_salt(api.smiles), pH_m, e_b_s, e_a_s, T_celsius=self.env_conditions.temperature)[0]
        n_v, p_v, m_o_v, poly_v, trans_v, cmplx_v, adsp_v, eut_v = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        r_d = float(np.mean(np.abs(vg.reactive_sites))) + 0.1
        m_d = float(np.mean(vg.metal_ion)) + 0.1
        p_d = float(np.mean(vg.peroxide_density)) + 0.05
        h_d = float(np.mean(vg.h_donor)) + 0.1
        for api in self.apis:
            for exc in self.excipients:
                api_smi, exc_smi = getattr(api, 'smiles', None), getattr(exc, 'smiles', None)
                n_v += compute_nucl_add_voi(api_smi, exc_smi, T_celsius=self.env_conditions.temperature, nucleophile_density=r_d, electrophile_density=r_d)[0]
                p_v += compute_photo_voi(api_smi, exc_smi, oxygen_level=self.env_conditions.oxygen_level)[0]
                m_o_v += compute_metal_ox_voi(api_smi, exc_smi, T_celsius=self.env_conditions.temperature, oxygen_level=self.env_conditions.oxygen_level, moisture=self.env_conditions.moisture, metal_density=m_d, peroxide_density=p_d)[0]
                poly_v += compute_polymorph_voi(api_smi, exc_smi, T_celsius=self.env_conditions.temperature, moisture=self.env_conditions.moisture, storage_months=max(self.env_conditions.storage_time, 1.0))[0]
                trans_v += compute_transester_voi(api_smi, exc_smi, T_celsius=self.env_conditions.temperature, pH=self.env_conditions.pH, moisture=self.env_conditions.moisture, alcohol_density=h_d)[0]
                cmplx_v += compute_complexation_voi(api_smi, exc_smi, T_celsius=self.env_conditions.temperature, pH=self.env_conditions.pH)[0]
                adsp_v += compute_adsorption_voi(api_smi, exc_smi, pH=self.env_conditions.pH)[0]
                eut_v += compute_eutectic_voi(api_smi, exc_smi, T_celsius=self.env_conditions.temperature)[0]
        return {
            'Maillard_VOI': m_v, 'Hydrolysis_VOI': h_v, 'Oxidation_VOI': o_v, 'AcidBase_VOI': a_v, 
            'Disproportionation_VOI': float(np.log1p(d_v)), 'NuclAdd_VOI': float(n_v), 'Photo_VOI': float(p_v), 
            'Polymorph_VOI': float(poly_v), 'MetalOx_VOI': float(m_o_v), 'Transester_VOI': float(trans_v),
            'Complexation_VOI': float(cmplx_v), 'Adsorption_VOI': float(adsp_v), 'Eutectic_VOI': float(eut_v)
        }

    def compute_voi_uncertainty(self, n=30):
        orig = {k: getattr(self.voxel_grid, k).copy() for k in ['reactive_sites', 'electrostatic', 'acid_density_api', 'base_density_api', 'acid_density_exc', 'base_density_exc']}
        samples = []
        for _ in range(n):
            for k, v in orig.items(): setattr(self.voxel_grid, k, v * (1 + np.random.normal(0, 0.05, v.shape)))
            samples.append(self.compute_voxel_overlap_integral())
        for k, v in orig.items(): setattr(self.voxel_grid, k, v)
        return {k: (np.mean(vs), np.std(vs), 1.96*np.std(vs)) for k, vs in {k: [s[k] for s in samples] for k in samples[0]}.items()}

    def classify_enhanced_compatibility(self, ocs, mechs):
        if any(m in mechs for m in ['Chemical Reaction Risk', 'Oxidation Risk']): return 'Incompatible', 'Critical risk detected'
        t = self.compatibility_thresholds
        if ocs >= t['compatible']: return 'Compatible', 'Stable'
        if ocs >= t['partially_compatible']: return 'Partially Compatible', f'Monitor: {", ".join(mechs[:2])}'
        return 'Incompatible', f'Poor stability: {", ".join(mechs[:2])}'

    def _ionisation_fraction(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return 0.0
        logP, fracs = Descriptors.MolLogP(mol), []
        a_p, b_p = Chem.MolFromSmarts('[CX3](=O)[OX2H1]'), Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
        if a_p and mol.HasSubstructMatch(a_p): fracs.append(1.0 / (1.0 + 10 ** ((4.0 + 0.3 * logP) - self.env_conditions.pH)))
        if b_p and mol.HasSubstructMatch(b_p): fracs.append(1.0 / (1.0 + 10 ** (self.env_conditions.pH - (9.0 - 0.4 * max(logP, 0.0)))))
        return float(max(fracs)) if fracs else 0.0

    def _logP_lipid_fraction(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return 0.0
        P = 10 ** Descriptors.MolLogP(mol)
        return float(P / (P + 1.0))

    def evaluate_docking_affinity(self, pdb_path: str, center: Tuple[float, float, float], radius: float = 10.0):
        """
        Tier 3: Performs a millisecond-scale docking affinity calculation using the C-based nibble engine.
        Implements the "Negative Space Matrix Multiplication" theory.
        """
        # 1. Initialize nibble engine (30x30x30 grid, 0.8A res)
        # Bounding box covers the pocket range
        nibble = NibbleEngine(dim_x=40, dim_y=40, dim_z=40, resolution=0.5)
        
        # 2. Load and Invert the PDB pocket with Thermodynamic Blur
        # Scaled by temperature (Higher T = more blur/flexibility)
        T_C = self.env_conditions.temperature
        blur = 1.2 + max(0.0, (T_C - 20.0) / 40.0) # ~1.2 at 20C, ~1.7 at 40C
        atoms = nibble.load_pdb_pocket(pdb_path, center, radius, blur_radius=blur)
        if atoms <= 0:
            return 0.0, "Pocket not found or empty"
            
        # 3. Project current drug population and calculate affinity
        # Start coords are center - radius for all axes
        start_coords = (center[0] - radius, center[1] - radius, center[2] - radius)
        
        # Translate molecules to the pocket center to avoid out-of-bounds projection
        for mol in self.apis + self.excipients:
            if len(mol.coords) > 0:
                mol_center = np.mean(mol.coords, axis=0)
                mol.coords = mol.coords + (np.array(center) - mol_center)
                
        affinity = nibble.project_molecule(self, start_coords)
        
        return affinity, f"Docking score: {affinity:.2f} (from {atoms} pocket atoms)"

    def _run_form_physics(self) -> 'DosageFormResult':
        cfg, env, T_K = self.dosage_config, self.env_conditions, self.env_conditions.temperature + 273.15
        f = cfg.form_type
        if f == DosageForm.ORAL_SOLID:
            a_T, X_cr = TemporalPhysics.wlf_time_temperature_superposition(env.temperature, 25.0), float(TemporalPhysics.avrami_crystallization(1e-12, 2.0, max(env.storage_time * 2592000, 1.0)))
            return DosageFormResult(form_type=f.value, ocs_modifier=-10.0*X_cr, modifier_label=f"Cryst. penalty ({X_cr:.2%} conv.)", extra_metrics={'wlf_shift_factor': float(a_T), 'crystallised_fraction': X_cr, 'storage_time_months': env.storage_time})
        elif f == DosageForm.SOLUTION:
            I, eps = max(env.ionic_strength, 1e-6), env.dielectric_constant
            d_r, m_i = float(np.clip(np.sqrt(eps / (78.5 * max(I, 0.01) / 0.15)), 0.2, 5.0)), float(np.mean([self._ionisation_fraction(a.smiles) for a in self.apis if getattr(a, 'smiles', None)])) if self.apis else 0.0
            i_b, i_p = 8.0 * m_i * (eps / 78.5), 3.0 * min(I / 0.5, 1.0)
            return DosageFormResult(form_type=f.value, ocs_modifier=i_b - i_p, modifier_label=f"Soln: ion. bonus {i_b:.1f}, ionic-p {i_p:.1f}", extra_metrics={'mean_ionisation_fraction': m_i, 'debye_length_ratio': d_r, 'ionic_strength': I, 'dielectric_constant': eps})
        elif f == DosageForm.LIPID_BASED:
            m_l = float(np.mean([self._logP_lipid_fraction(a.smiles) for a in self.apis if getattr(a, 'smiles', None)])) if self.apis else 0.5
            l_b, e_p = 12.0 * m_l * cfg.lipid_fraction, sum(8.0 * cfg.lipid_fraction for a in self.apis if getattr(a, 'smiles', None) and Chem.MolFromSmiles(a.smiles).HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2]')))
            fh = self.compute_flory_huggins_phase_behavior(); c_s, c_c = fh.get('chi_scalar', 1.0), fh.get('chi_critical', 0.9)
            p_p, g_i = 5.0 * max(0.0, c_s - c_c), MicrostructurePhysics.interface_energy(c_s, 35.0, 25.0)
            return DosageFormResult(form_type=f.value, ocs_modifier=l_b - e_p - p_p, modifier_label=f"LBF: lip-b {l_b:.1f}, ester-p {e_p:.1f}, ph-sep-p {p_p:.1f}", extra_metrics={'mean_lipid_partition_fraction': m_l, 'lipid_volume_fraction': cfg.lipid_fraction, 'flory_huggins_chi': c_s, 'chi_critical': c_c, 'interface_energy_mN_m': float(g_i), 'interfacial_stability_score': float(np.exp(-g_i / 20.0))})
        elif f == DosageForm.CONTROLLED_RELEASE:
            D_T = cfg.coating_diffusivity * np.exp(50000.0 / 8.314 * (1 / 298.15 - 1 / T_K))
            c_i = float(np.clip(1.0 - ProcessPhysics.fickian_coating_diffusion(D_T, cfg.coating_thickness_um, 50)[1][-1], 0.0, 1.0))
            c_r = float(np.mean(TemporalPhysics.fick_second_law_1d(1.0, D_T * 0.1, cfg.coating_thickness_um * 10, 30, 86400.0)[2][:, -1]))
            c_b, b_p, m_p = 10.0 * c_i * c_r, 8.0 * (1.0 - c_i), 5.0 * env.moisture * (1.0 - c_i)
            return DosageFormResult(form_type=f.value, ocs_modifier=c_b - b_p - m_p, modifier_label=f"CR: int-b {c_b:.1f}, burst-p {b_p:.1f}, moist-p {m_p:.1f}", extra_metrics={'coating_thickness_um': cfg.coating_thickness_um, 'coating_diffusivity_cm2s': float(D_T), 'coating_integrity_1h': c_i, 'core_retention_24h': c_r, 'temperature_celsius': env.temperature})
        elif f == DosageForm.MULTI_PHASE:
            p0 = (self.voxel_grid.hydrophobic - self.voxel_grid.hydrophobic.min()) / (np.ptp(self.voxel_grid.hydrophobic) + 1e-10)
            fh = self.compute_flory_huggins_phase_behavior(); c_s, c_c = fh.get('chi_scalar', 1.0), fh.get('chi_critical', 0.9)
            p_f = MicrostructurePhysics.cahn_hilliard_phase_field(c_s, p0[:, :, p0.shape[2]//2].copy(), 1.0, 0.005, 20)[-1]
            g_i, o_r = MicrostructurePhysics.interface_energy(c_s, 35.0, 30.0), float(np.clip(2.0 * MicrostructurePhysics.interface_energy(c_s, 35.0, 30.0) * 1e-3 * 1e-4 / (8.314 * T_K * cfg.droplet_diameter_um / 2.0 * 1e-6), 0.0, 1.0))
            s_b, o_p = 8.0 * float(np.exp(-max(0.0, c_s - c_c))), 5.0 * o_r
            return DosageFormResult(form_type=f.value, ocs_modifier=s_b - o_p, modifier_label=f"Multi-phase: stab-b {s_b:.1f}, Ost-p {o_p:.1f}", extra_metrics={'flory_huggins_chi': c_s, 'chi_critical': c_c, 'interfacial_stability_score': float(np.exp(-max(0.0, c_s - c_c))), 'interface_energy_mN_m': float(g_i), 'interface_fraction': int(np.sum((p_f > 0.2) & (p_f < 0.8))) / p_f.size, 'droplet_diameter_um': cfg.droplet_diameter_um, 'ostwald_ripening_risk': o_r})
        elif f == DosageForm.SUSPENSION:
            eta = cfg.viscosity_cP * 0.001
            v_s = DosageSpecificPhysics.stokes_sedimentation_velocity(cfg.particle_diameter_um / 2.0, cfg.particle_density_g_cm3, 1.0, eta)
            wet_comp = float(np.exp(-np.mean(np.abs(self.voxel_grid.hydrophobic)) / 2.0))
            # Suspending agent bonus (stabilizes network)
            stab_b = 5.0 * np.log1p(cfg.suspending_agent_conc)
            ocs_mod = 10.0 * wet_comp - 5.0 * np.log1p(v_s) + stab_b
            return DosageFormResult(form_type=f.value, ocs_modifier=ocs_mod, modifier_label=f"Suspension: wet-b {10*wet_comp:.1f}, sed-p {5*np.log1p(v_s):.1f}, stab-b {stab_b:.1f}", extra_metrics={'sedimentation_velocity_um_s': float(v_s), 'wetting_compatibility': wet_comp, 'particle_diameter_um': cfg.particle_diameter_um, 'particle_density_g_cm3': cfg.particle_density_g_cm3, 'viscosity_cP': cfg.viscosity_cP, 'suspending_agent_conc': cfg.suspending_agent_conc})
        elif f == DosageForm.INJECTABLE:
            osm = DosageSpecificPhysics.compute_osmolarity([a.concentration for a in self.apis], [1.0] * len(self.apis))
            sap = DosageSpecificPhysics.spatial_aggregation_propensity(self.voxel_grid.hydrophobic, self.voxel_grid.electrostatic)
            osm_err = abs(osm - cfg.target_osmolarity) / cfg.target_osmolarity
            ocs_mod = -20.0 * osm_err - 10.0 * np.log1p(max(sap, 0))
            return DosageFormResult(form_type=f.value, ocs_modifier=ocs_mod, modifier_label=f"Injectable: osm-p {20*osm_err:.1f}, agg-p {10*np.log1p(max(sap, 0)):.1f}", extra_metrics={'calculated_osmolarity': float(osm), 'aggregation_propensity': float(sap), 'target_osmolarity': cfg.target_osmolarity})
        return DosageFormResult(form_type=str(f), ocs_modifier=0.0, modifier_label=f"Unknown form {f}")

    def analyze_enhanced_formulation(self, include_uncertainty=True, include_voi_uncertainty=False, n_mc=50):
        self.compute_multi_molecule_superposition(); m = self.compute_enhanced_interaction_metrics()
        mechs = self.detect_enhanced_incompatibility_mechanisms(m)
        if self.compute_reaction_integrals() > 50: mechs.append('High Reaction Probability')
        voi_un = self.compute_voi_uncertainty() if include_voi_uncertainty else {}
        f_r = self._run_form_physics(); ocs_b = float(m['OCS']); ocs_f = float(np.clip(ocs_b + f_r.ocs_modifier, 0.0, 100.0))
        m.update({'OCS_base': ocs_b, 'OCS': ocs_f, 'form_ocs_modifier': f_r.ocs_modifier})
        cls, mch = self.classify_enhanced_compatibility(ocs_f, mechs)
        self.interaction_metrics, self.confidence_intervals = m, (self.monte_carlo_uncertainty_analysis(n_mc) if include_uncertainty else {})
        if f_r.is_stub and f_r.warnings: mechs.extend(f_r.warnings)
        return {'metrics': m, 'classification': cls, 'mechanism': mch, 'mechanisms': mechs, 'confidence_intervals': self.confidence_intervals, 'voi_uncertainty': voi_un, 'form_physics': f_r.__dict__}

    def analyze_complete_physics(self):
        std, fh, T_K, t_s = self.compute_enhanced_interaction_metrics(), self.compute_flory_huggins_phase_behavior(), self.env_conditions.temperature + 273.15, max(self.env_conditions.storage_time * 2592000, 1.0)
        return {'standard': std, 'process': {'amorph_risk': float(TemporalPhysics.avrami_crystallization(1e-12, 2.0, t_s)), 'wlf_shift_factor': float(TemporalPhysics.wlf_time_temperature_superposition(self.env_conditions.temperature, 25.0))}, 'microstructure': {'interface_energy': float(MicrostructurePhysics.interface_energy(fh.get('chi_scalar', 1.0), 35.0, 30.0)), 'flory_huggins_chi': fh.get('chi_scalar', 1.0)}, 'temporal': {'nucleation_rate': float(TemporalPhysics.classical_nucleation_theory(-1e6, 0.05, T_K)[2])}, 'form_physics': self._run_form_physics().__dict__}

    def compute_reaction_integrals(self):
        t = 0.0
        if not self.apis or not self.excipients: return 0.0
        for a in self.apis:
            r_a = self.project_enhanced_molecule_fields(a)['reactive_sites']
            for e in self.excipients: t += np.sum(r_a * self.project_enhanced_molecule_fields(e)['reactive_sites'])
        return t

    def optimize_molecular_orientation(self, mol, strategy='minimize_clash', steps=50):
        best_c, best_s = mol.coords.copy(), (1e9 if strategy == 'minimize_clash' else -1e9)
        for _ in range(steps):
            Q, R = np.linalg.qr(np.random.randn(3, 3)); center = np.mean(mol.coords, axis=0)
            new_c = np.dot(mol.coords - center, Q.T) + center + np.random.uniform(-2, 2, 3)
            idx = np.clip((new_c/self.voxel_grid.resolution).astype(int), 0, np.array(self.voxel_grid.shape)-1)
            s = np.sum(self.voxel_grid.steric[idx[:,0], idx[:,1], idx[:,2]])
            if (strategy == 'minimize_clash' and s < best_s) or (strategy == 'maximize_hbond' and s > best_s): best_s, best_c = s, new_c.copy()
        mol.coords = best_c; return mol

    def detect_enhanced_incompatibility_mechanisms(self, metrics):
        voi, mechs = self.compute_voxel_overlap_integral(), []
        def get_conf(v, t): return 1.0 / (1.0 + np.exp(-4.0 * (v/t - 1.2)))
        # Standard mechanisms
        for k, t, label in [
            ('Maillard_VOI', 0.1, 'Maillard Risk'), 
            ('Hydrolysis_VOI', 0.05, 'Hydrolysis Risk'), 
            ('Oxidation_VOI', 0.05, 'Oxidation Risk'), 
            ('AcidBase_VOI', 0.3, 'Acid-Base Risk'), 
            ('MetalOx_VOI', 0.2, 'Metal-Catalysed Oxidation Risk'), 
            ('Polymorph_VOI', 0.35, 'Polymorph Conversion Risk'), 
            ('Photo_VOI', 0.3, 'Photodegradation Risk'), 
            ('NuclAdd_VOI', 0.2, 'Nucleophilic Addition Risk'), 
            ('Disproportionation_VOI', 0.2, 'Disproportionation Risk')
        ]:
            if voi.get(k, 0) > t * 0.5: 
                mechs.append(f"{label} (VOI={voi[k]:.3f}, Conf={get_conf(voi[k], t):.1%})")
        
        # Tier 3 Advanced Mechanisms
        if voi.get('Transester_VOI', 0) > 0.1:
            conf = get_conf(voi['Transester_VOI'], 0.5)
            mechs.append(f"Transesterification Risk (VOI={voi['Transester_VOI']:.3f}, Conf={conf:.1%})")
        if voi.get('Complexation_VOI', 0) > 0.1:
            conf = get_conf(voi['Complexation_VOI'], 0.5)
            mechs.append(f"Complexation Risk (VOI={voi['Complexation_VOI']:.3f}, Conf={conf:.1%})")
        if voi.get('Adsorption_VOI', 0) > 0.1:
            conf = get_conf(voi['Adsorption_VOI'], 0.5)
            mechs.append(f"Surface Adsorption Risk (VOI={voi['Adsorption_VOI']:.3f}, Conf={conf:.1%})")
        if voi.get('Eutectic_VOI', 0) > 0.1:
            conf = get_conf(voi['Eutectic_VOI'], 0.5)
            mechs.append(f"Eutectic Formation Risk (VOI={voi['Eutectic_VOI']:.3f}, Conf={conf:.1%})")

        if metrics.get('SCI', 0) > 0.02: 
            mechs.append(f"Steric Clash (SCI={metrics['SCI']:.3f}, Conf={get_conf(metrics['SCI'], 0.05):.1%})")
        return mechs if mechs else ['No specific mechanism detected']

    def monte_carlo_uncertainty_analysis(self, n=50):
        o_e, res = EnvironmentalConditions(pH=self.env_conditions.pH, moisture=self.env_conditions.moisture, temperature=self.env_conditions.temperature), []
        for _ in range(n):
            self.env_conditions.temperature, self.env_conditions.moisture = o_e.temperature + np.random.normal(0, 1.0), np.clip(o_e.moisture + np.random.normal(0, 0.05), 0, 1)
            self.compute_multi_molecule_superposition(); res.append(self.compute_enhanced_interaction_metrics()['OCS'])
        self.env_conditions = o_e
        return {'mean': np.mean(res), 'std': np.std(res), 'ci_95': (np.percentile(res, 2.5), np.percentile(res, 97.5)), 'distribution': res}

    def perform_global_sensitivity_analysis(self, samples: int = 10) -> Dict:
        RANGES = {'pH': (3.0, 10.0), 'temperature': (15.0, 60.0), 'moisture': (0.0, 0.9), 'ionic_strength': (0.0, 1.0), 'oxygen_level': (0.0, 0.21), 'storage_time': (0.0, 24.0)}
        keys, p, delta, rng = list(RANGES.keys()), len(RANGES), 0.5, np.random.default_rng(42)
        o_e, ee_all, n_c = self.env_conditions, {k: [] for k in RANGES}, 0
        def _eval(xn):
            self.env_conditions = EnvironmentalConditions(**{k: float(RANGES[k][0] + xn[i]*(RANGES[k][1]-RANGES[k][0])) for i, k in enumerate(keys)})
            self.compute_multi_molecule_superposition(); return float(self.compute_enhanced_interaction_metrics()['OCS'])
        for _ in range(samples):
            x0 = rng.uniform(0.0, 1.0 - delta, size=p); y_p = _eval(x0); n_c += 1; x_c = x0.copy()
            for idx in rng.permutation(p):
                x_n = x_c.copy(); step = delta if (x_c[idx] + delta <= 1.0) else -delta; x_n[idx] += step
                y_n = _eval(x_n); n_c += 1; ee_all[keys[idx]].append((y_n - y_p) / (step * (RANGES[keys[idx]][1] - RANGES[keys[idx]][0]))); y_p, x_c = y_n, x_n
        self.env_conditions = o_e; self.compute_multi_molecule_superposition()
        m_s = {k: float(np.mean(np.abs(ee_all[k]))) for k in keys}; sig = {k: float(np.std(ee_all[k])) for k in keys}
        rank = sorted(keys, key=lambda k: m_s[k], reverse=True)
        return {'mu_star': m_s, 'sigma': sig, 'ranking': rank, 'primary_risk': rank[0], 'levers': {k: v / max(max(m_s.values()), 1e-9) for k, v in m_s.items()}, 'interaction_flags': {k: bool((sig[k] / max(m_s[k], 1e-9)) > 0.5) if m_s[k] > 0.01 else False for k in keys}, 'high_sensitivity': [k for k in keys if m_s[k] > 0.3], 'n_calls': n_c, 'n_trajectories': samples, 'param_ranges': RANGES}

    def get_stability_recommendations(self) -> List[str]:
        recs, env = [], self.env_conditions
        if not hasattr(self, 'interaction_metrics') or not self.interaction_metrics: self.compute_multi_molecule_superposition(); self.interaction_metrics = self.compute_enhanced_interaction_metrics()
        m, voi = self.interaction_metrics, self.compute_voxel_overlap_integral(); ocs = float(m.get('OCS', 50.0))
        recs.append(f"[{'INFO' if ocs>=80 else ('ADVISORY' if ocs>=60 else ('WARNING' if ocs>=40 else 'CRITICAL'))}] OCS={ocs:.1f} - " + ( "compatible" if ocs>=80 else ("marginal" if ocs>=60 else ("concern" if ocs>=40 else "high risk"))))
        if voi.get('Maillard_VOI', 0) > 0.05: recs.append(f"[{'CRITICAL' if voi['Maillard_VOI']>1 else 'WARNING'}] Maillard risk (VOI={voi['Maillard_VOI']:.3f})")
        if voi.get('Hydrolysis_VOI', 0) > 0.02: recs.append(f"[{'CRITICAL' if voi['Hydrolysis_VOI']>0.5 else 'WARNING'}] Hydrolysis risk (VOI={voi['Hydrolysis_VOI']:.3f})")
        if voi.get('Oxidation_VOI', 0) > 0.1 or env.oxygen_level > 0.18: recs.append(f"[{'WARNING' if voi['Oxidation_VOI']>0.1 else 'ADVISORY'}] Oxidation risk (VOI={voi['Oxidation_VOI']:.3f})")
        if voi.get('Disproportionation_VOI', 0) > 0.05: recs.append(f"[WARNING] Disproportionation risk (VOI={voi['Disproportionation_VOI']:.3f})")
        if voi.get('MetalOx_VOI', 0) > 0.05: 
            recs.append(f"[{'CRITICAL' if voi['MetalOx_VOI']>0.8 else 'WARNING'}] Metal-catalysed oxidation risk (VOI={voi['MetalOx_VOI']:.3f})")
            recs.append("[TIP] Haber-Weiss inhibition: use chelating agents (EDTA, Citrate) and screen excipients for trace Fe/Cu/peroxides.")
        if voi.get('Polymorph_VOI', 0) > 0.05: recs.append(f"[WARNING] Polymorph conversion risk (VOI={voi['Polymorph_VOI']:.3f})")
        if voi.get('Photo_VOI', 0) > 0.05: recs.append(f"[WARNING] Photodegradation risk (VOI={voi['Photo_VOI']:.3f})")
        if voi.get('NuclAdd_VOI', 0) > 0.05: recs.append(f"[WARNING] Nucleophilic addition risk (VOI={voi['NuclAdd_VOI']:.3f})")
        # ── Transesterification ───────────────────────────────────────────────
        te_voi = float(voi.get('Transester_VOI', 0.0))
        if te_voi > 0.1:
            te_pairs: List[str] = []
            for api in self.apis:
                for exc in self.excipients:
                    _, pairs = compute_transester_voi(
                        getattr(api, 'smiles', None),
                        getattr(exc, 'smiles', None),
                        T_celsius=env.temperature, pH=env.pH,
                        moisture=env.moisture,
                    )
                    te_pairs.extend(pairs)
            te_pairs = list(dict.fromkeys(te_pairs))
            severity = 'CRITICAL' if te_voi > 1.0 else 'WARNING'
            advice = []
            # Specific advice logic based on left_tier.txt
            if any(p and ('amine' in p.lower() or 'NH2' in p) for p in te_pairs):
                advice.append("aminolysis detected — primary amine API attacks ester excipient; replace polyol/PEG excipient or use non-ester binder (HPMC, PVP)")
            if any(p and ('PEG' in p or 'polyether' in p or 'CCO' in p) for p in te_pairs):
                advice.append("PEG transesterification — Aspirin-class APIs form PEG esters; use anhydrous processing, avoid PEG in direct contact, consider cellulose-based binder instead")
            if any(p and ('mannitol' in p.lower() or 'sorbitol' in p.lower() or 'OH' in p) for p in te_pairs):
                advice.append("polyol (mannitol/sorbitol) transesterification risk; use spray-dried dispersion to minimise contact, lower storage temperature, target pH < 6 in microenvironment")
            if not advice:
                advice.append("avoid co-formulation of ester APIs with polyol/PEG excipients; use pH-controlled microenvironment (citric acid buffer pH 5-6)")
            
            recs.append(f"[{severity}] Transesterification risk (VOI={te_voi:.3f}): active pairs: {te_pairs[:2]}. " + " | ".join(advice))

        if voi.get('Complexation_VOI', 0) > 0.1:
            cmplx_voi = float(voi['Complexation_VOI'])
            recs.append(
                f"[ADVISORY] Complexation risk (VOI={cmplx_voi:.3f}): "
                "API-excipient complex may alter dissolution rate and bioavailability. "
                "Verify in vitro dissolution at pH 1.2, 4.5, 6.8 (USP <711>). "
                "Cyclodextrin complexation: confirm Ka by phase solubility; "
                "PVP/HPMC: check drug release from ASD by USP dissolution. "
                "Ref: Loftsson & Brewster J. Pharm. Pharmacol. 1996."
            )
        if voi.get('Adsorption_VOI', 0) > 0.1:
            adsp_voi = float(voi['Adsorption_VOI'])
            recs.append(
                f"[ADVISORY] Surface Adsorption risk (VOI={adsp_voi:.3f}): "
                "API adsorption onto high-surface-area excipient reduces bioavailability. "
                "Specify low-surface-area grades (e.g., granular vs micronized), "
                "pre-saturate excipient with surfactant (SLS 0.1%), or use hydrophilic coating. "
                "Ref: Adeyeye & Brittain (2008) Preformulation in Solid Dosage Form Development."
            )
        if voi.get('Eutectic_VOI', 0) > 0.1:
            eut_voi = float(voi['Eutectic_VOI'])
            recs.append(
                f"[{'WARNING' if eut_voi > 0.5 else 'ADVISORY'}] Eutectic Formation risk (VOI={eut_voi:.3f}): "
                "Binary eutectic may depress melting point below storage/processing temperature. "
                "Perform DSC binary phase diagram screening; avoid HME above eutectic T; "
                "use non-lipid binders at >40°C storage. "
                "Ref: Tong P. et al. AAPS PharmSciTech (2011)."
            )
        if env.temperature > 30: recs.append(f"[WARNING] High temperature {env.temperature:.0f}C")
        if env.moisture > 0.4: recs.append(f"[WARNING] High moisture {env.moisture:.0%}")
        return recs if recs else ["[INFO] No critical risks identified."]

    def optimize_tablet_topology(self) -> Dict:
        cfg, env, T_K = self.dosage_config, self.env_conditions, self.env_conditions.temperature + 273.15
        D_b = cfg.coating_diffusivity * np.exp(50000.0 / 8.314 * (1.0 / 298.15 - 1.0 / T_K))
        chi = self.compute_flory_huggins_phase_behavior().get('chi_scalar', 1.0); D_c = D_b * (1.0 + max(0.0, chi - 0.5))
        archs = {'Monolithic': [1.0], 'Bilayer': [0.6, 0.4], 'Trilayer': [0.5, 0.3, 0.2], 'Coated-core': [0.2, 0.8]}
        res = {}
        for name, fracs in archs.items():
            l_r = []
            for i, f in enumerate(fracs):
                D_l, L_cm = D_c * (10.0 ** (i - len(fracs) // 2)), cfg.coating_thickness_um * f * 1e-4
                M_1h = float(1.0 - np.exp(-D_l * 3600.0 / L_cm ** 2))
                ser = sum(np.exp(-(2*k+1)**2 * np.pi**2 * D_l*0.1 * 86400.0 / (L_cm*10.0)**2) for k in range(5))
                l_r.append({'thickness_um': cfg.coating_thickness_um * f, 'D_layer': D_l, 'released_1h': M_1h, 'retained_24h': float(np.clip((8.0 / np.pi**2) * ser, 0.0, 1.0))})
            m_ret, m_rel = np.mean([x['retained_24h'] for x in l_r]), np.mean([x['released_1h'] for x in l_r])
            res[name] = {'n_layers': len(fracs), 'layer_results': l_r, 'mean_retained_24h': float(m_ret), 'mean_released_1h': float(m_rel), 'score': float(0.6 * m_ret + 0.4 * max(1.0 - abs(m_rel - 0.5) * 2.0, 0.0))}
        best = max(res, key=lambda k: res[k]['score'])
        return {'best_architecture': best, 'improvement': float(res[best]['score'] - res['Monolithic']['score']), 'all_architectures': res, 'D_effective': float(D_c), 'chi_scalar': float(chi)}
