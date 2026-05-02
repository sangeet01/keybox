# central orchestrator

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy import ndimage
from rdkit import Chem
from .models import (
    Molecule, MoleculeType, DosageForm, EnvironmentalConditions, 
    DosageFormConfig, DosageFormResult
)
from .voxel_grid import EnhancedVoxelGrid
from .physics import (
    ProcessPhysics, MicrostructurePhysics, 
    TemporalPhysics, DosageSpecificPhysics
)
from .mechanisms import (
    compute_voxel_overlap_integral, detect_enhanced_incompatibility_mechanisms,
    get_stability_recommendations, compute_transester_voi
)
from .thermodynamics import (
    compute_flory_huggins_phase_behavior, compute_dissolution_profile,
    compute_reaction_integrals, monte_carlo_uncertainty_analysis
)
from .dosage_forms import run_dosage_form_physics
from .nibble_bridge import NibbleEngine

class KeyBoxSystem:
    """
    KeyBoxSystem (Orchestrator V3):
    Coordinates spatial projection, chemical mechanism integration, 
    and form-specific physics dispatching.
    """
    def __init__(self, voxel_bounds: Tuple[float, float, float] = (20.0, 20.0, 20.0), 
                 resolution: float = 0.5, 
                 dosage_config: Optional[DosageFormConfig] = None,
                 env_conditions: Optional[EnvironmentalConditions] = None,
                 random_state: Optional[int] = None):
        self.voxel_grid = EnhancedVoxelGrid(voxel_bounds, resolution)
        self.apis: List[Molecule] = []
        self.excipients: List[Molecule] = []
        self.dosage_config = dosage_config or DosageFormConfig(DosageForm.ORAL_SOLID)
        self.env_conditions = env_conditions or EnvironmentalConditions()
        self.phase_diffusion_rates = {0: 1.0, 1: 0.1, 2: 0.01} # air, lipid, polymer
        self.interaction_metrics: Dict[str, float] = {}
        self.confidence_intervals: Dict[str, Tuple[float, float]] = {}
        self.random_state = random_state
        if random_state is not None: np.random.seed(random_state)

    def add_molecule(self, molecule: Molecule, is_api: bool = True):
        """Adds a molecule and updates the spatial superposition."""
        if is_api: self.apis.append(molecule)
        else: self.excipients.append(molecule)
        self.compute_multi_molecule_superposition()

    def compute_multi_molecule_superposition(self):
        """Re-calculates the aggregate voxel fields for all components."""
        for a in ['electrostatic', 'hydrophobic', 'steric', 'h_donor', 'h_acceptor', 'polarizability', 'solvent_accessible', 'reactive_sites', 'acid_density', 'base_density', 'acid_density_api', 'base_density_api', 'acid_density_exc', 'base_density_exc', 'amine_density_api', 'carbonyl_density_api', 'amine_density_exc', 'carbonyl_density_exc', 'ph_field']:
            getattr(self.voxel_grid, a).fill(0)
        
        for mol in self.apis:
            fields = self.project_enhanced_molecule_fields(mol)
            for k, v in fields.items():
                if hasattr(self.voxel_grid, k):
                    getattr(self.voxel_grid, k)[:] += v.reshape(self.voxel_grid.shape)
                    
        for mol in self.excipients:
            fields = self.project_enhanced_molecule_fields(mol)
            for k, v in fields.items():
                if hasattr(self.voxel_grid, k):
                    getattr(self.voxel_grid, k)[:] += v.reshape(self.voxel_grid.shape)

    def project_enhanced_molecule_fields(self, molecule: Molecule) -> Dict[str, np.ndarray]:
        """Maps atomic properties to voxel channels using physical kernels."""
        VC = self.voxel_grid._voxel_coords_flat; sh = self.voxel_grid.shape
        f_es, f_acid, f_base, f_ami, f_car = [np.zeros(sh).flatten() for _ in range(5)]
        f_hy, f_st, f_hd, f_ha, f_pol, f_rx, f_ph = [np.zeros(sh).flatten() for _ in range(7)]
        f_lip, f_vis, f_met, f_per = [np.zeros(sh).flatten() for _ in range(4)]
        
        coulomb_k, kappa, epsilon_lj, sigma2_2 = 138.9, 0.1, 0.5, 0.5 * 1.5**2
        lipid_f = 0.8 if molecule.mol_type == MoleculeType.LIPID else 0.0
        visc_f = 0.5 if molecule.mol_type == MoleculeType.POLYMER else 0.0
        is_basic = getattr(molecule, 'is_basic', False) or (molecule.base_groups is not None and (molecule.base_groups > 0.5).any())
        is_acidic = getattr(molecule, 'is_acidic', False) or (molecule.acid_groups is not None and (molecule.acid_groups > 0.5).any())
        ph_shift = 0.5 if is_basic else (-0.5 if is_acidic else 0.0)
        has_met = any(s in molecule.smiles for s in ['[Fe]', '[Cu]', '[Mn]', '[Ti]']) if molecule.smiles else False
        is_met_prone = any(s in molecule.smiles for s in ['C@@H', '[Si]']) if molecule.smiles else False
        has_per = '[OX2][OX2]' in molecule.smiles if molecule.smiles else False

        t_ch = molecule.taut_charges if molecule.taut_charges is not None else [molecule.charges]
        t_ac = molecule.taut_acid_groups if getattr(molecule, 'taut_acid_groups', None) is not None else [molecule.acid_groups]
        t_ba = molecule.taut_base_groups if getattr(molecule, 'taut_base_groups', None) is not None else [molecule.base_groups]
        t_am = molecule.taut_amine_groups if getattr(molecule, 'taut_amine_groups', None) is not None else [molecule.amine_groups]
        t_cb = molecule.taut_carbonyl_groups if getattr(molecule, 'taut_carbonyl_groups', None) is not None else [molecule.carbonyl_groups]
        t_hd = molecule.taut_h_donors if getattr(molecule, 'taut_h_donors', None) is not None else [molecule.h_donors]
        t_ha = molecule.taut_h_acceptors if getattr(molecule, 'taut_h_acceptors', None) is not None else [molecule.h_acceptors]
        GLOBAL_CUTOFF = 6.0
        from scipy.spatial import cKDTree
        
        confs = molecule.conformations if molecule.conformations is not None else [molecule.coords]
        weights = molecule.taut_weights if molecule.taut_weights is not None else [1.0/len(confs)] * len(confs)
        for t_idx, conf in enumerate(confs):
            w = weights[t_idx]
            if len(conf) > 100:
                tree = cKDTree(VC); sdm = tree.sparse_distance_matrix(cKDTree(conf.astype(np.float64)), GLOBAL_CUTOFF, output_type='coo_matrix')
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
        
        c = molecule.concentration
        return {'electrostatic': (f_es * c).reshape(sh), 'acid_density': (f_acid * c).reshape(sh), 'base_density': (f_base * c).reshape(sh), 'amine_density': (f_ami * c).reshape(sh), 'carbonyl_density': (f_car * c).reshape(sh), 'hydrophobic': (f_hy * c).reshape(sh), 'steric': (f_st * c).reshape(sh), 'h_donor': (f_hd * c).reshape(sh), 'h_acceptor': (f_ha * c).reshape(sh), 'polarizability': (f_pol * c).reshape(sh), 'reactive_sites': (f_rx * c).reshape(sh), 'lipid_density': (f_lip * c).reshape(sh), 'viscosity': (f_vis * c).reshape(sh), 'ph_field': (f_ph * c).reshape(sh), 'metal_ion': (f_met * c).reshape(sh), 'peroxide_density': (f_per * c).reshape(sh), 'ionic_field': np.zeros(sh)}

    def compute_enhanced_interaction_metrics(self) -> Dict[str, float]:
        """Calculates Tier 2 stability metrics from voxel fields."""
        G = self.voxel_grid; vol = np.prod(G.shape) * (G.resolution**3)
        SCI = float(np.sum(G.steric[G.steric > 1.0]) / vol)
        HBI = float(np.sum(G.h_donor * G.h_acceptor) / vol)
        ESI = float(np.sum(np.abs(G.electrostatic)) / vol)
        OCS = float(100.0 * np.exp(-(SCI * 50.0 + (1.0 - np.clip(HBI*10.0, 0, 1)) * 5.0 + ESI * 2.0)))
        return {'SCI': SCI, 'HBI': HBI, 'ESI': ESI, 'OCS': OCS}

    def analyze_enhanced_formulation(self, include_uncertainty=True, n_mc=50):
        """Complete formulation analysis pipeline."""
        m = self.compute_enhanced_interaction_metrics()
        voi = compute_voxel_overlap_integral(self)
        mechs = detect_enhanced_incompatibility_mechanisms(self, voi, m)
        recs = get_stability_recommendations(self, voi, m)
        
        voi_un = {} # Simplified
        ci = monte_carlo_uncertainty_analysis(self, n_mc) if include_uncertainty else {}
        f_r = run_dosage_form_physics(self.apis, self.excipients, self.env_conditions, self.voxel_grid, self.dosage_config, self)
        
        return {'metrics': m, 'classification': 'Compatible' if m['OCS'] > 70 else 'Risk', 'mechanism': mechs[0] if mechs else 'None', 'mechanisms': mechs, 'stability_recommendations': recs, 'confidence_intervals': ci, 'form_physics': f_r.__dict__}

    def analyze_complete_physics(self):
        """High-level physical state simulation."""
        std = self.analyze_enhanced_formulation()
        fh = compute_flory_huggins_phase_behavior(self.apis, self.excipients, self.env_conditions, self.voxel_grid)
        T_K = self.env_conditions.temperature + 273.15
        t_s = max(self.env_conditions.storage_time * 2592000, 1.0)
        
        return {
            'standard': std, 
            'process': {'amorph_risk': float(TemporalPhysics.avrami_crystallization(1e-12, 2.0, t_s))}, 
            'microstructure': {'interface_energy': float(MicrostructurePhysics.interface_energy(fh.get('chi_scalar', 1.0), 35.0, 30.0)), 'flory_huggins_chi': fh.get('chi_scalar', 1.0)}, 
            'temporal': {'nucleation_rate': float(TemporalPhysics.classical_nucleation_theory(-1e6, 0.05, T_K)[2])}, 
            'form_physics': run_dosage_form_physics(self.apis, self.excipients, self.env_conditions, self.voxel_grid, self.dosage_config, self).__dict__
        }

    def evaluate_docking_affinity(self, pdb_path: str, center: Tuple[float, float, float], radius: float = 10.0):
        """
        Two-phase docking cascade (theory.txt):
          Phase I  Coarse  1.0A: Steric + electrostatic fast filter  (~10 us)
          Phase II Sniper  0.25A: Full 11-channel + B-factor Gaussian (~5 ms)
        """
        T_C = self.env_conditions.temperature
        # Blur scales with temperature (Arrhenius thermal averaging, theory.txt sec 5)
        blur_base = 1.2 + max(0.0, (T_C - 20.0) / 40.0)
        start_coords = (center[0] - radius, center[1] - radius, center[2] - radius)

        # Translate molecules to the pocket center to avoid out-of-bounds projection
        for mol in self.apis + self.excipients:
            if len(mol.coords) > 0:
                mol_center = np.mean(mol.coords, axis=0)
                mol.coords = mol.coords + (np.array(center) - mol_center)

        # Phase I — Coarse Broadsword: 1.0A, box = (2*radius / 1.0) voxels per side
        box_c = max(10, int(2 * radius / 1.0))
        coarse = NibbleEngine(dim_x=box_c, dim_y=box_c, dim_z=box_c, resolution=1.0)
        atoms = coarse.load_pdb_pocket(pdb_path, center, radius, blur_radius=blur_base)
        if atoms <= 0:
            return 0.0, "Pocket not found or empty"
        coarse_score = coarse.project_molecule(self, start_coords)

        # Phase II — Sniper: 0.25A, box = (2*radius / 0.25) voxels per side
        # B-factor blur is applied inside nibble_load_pdb_pocket per-atom
        box_s = max(20, int(2 * radius / 0.25))
        sniper = NibbleEngine(dim_x=box_s, dim_y=box_s, dim_z=box_s, resolution=0.25)
        sniper.load_pdb_pocket(pdb_path, center, radius, blur_radius=blur_base)
        sniper_score = sniper.project_molecule(self, start_coords)

        report = (f"Phase I  Coarse : {coarse_score:.1f} | "
                  f"Phase II Sniper : {sniper_score:.2f} | "
                  f"Pocket atoms    : {atoms}")
        return sniper_score, report

    def optimize_tablet_topology(self) -> Dict:
        """Topological optimization of multi-layer tablet architectures."""
        cfg, env, T_K = self.dosage_config, self.env_conditions, self.env_conditions.temperature + 273.15
        D_b = cfg.coating_diffusivity * np.exp(50000.0 / 8.314 * (1.0 / 298.15 - 1.0 / T_K))
        fh = compute_flory_huggins_phase_behavior(self.apis, self.excipients, self.env_conditions, self.voxel_grid)
        chi = fh.get('chi_scalar', 1.0); D_c = D_b * (1.0 + max(0.0, chi - 0.5))
        archs = {'Monolithic': [1.0], 'Bilayer': [0.6, 0.4], 'Trilayer': [0.5, 0.3, 0.2], 'Coated-core': [0.2, 0.8]}
        res = {}
        for name, fracs in archs.items():
            l_r = []
            for i, f in enumerate(fracs):
                D_l, L_cm = D_c * (10.0 ** (i - len(fracs) // 2)), cfg.coating_thickness_um * f * 1e-4
                M_1h = float(1.0 - np.exp(-D_l * 3600.0 / L_cm ** 2))
                ser = sum(np.exp(-(2*k+1)**2 * np.pi**2 * D_l*0.1 * 86400.0 / (L_cm*10.0)**2) for k in range(5))
                l_r.append({'retained_24h': float(np.clip((8.0 / np.pi**2) * ser, 0.0, 1.0))})
            m_ret = np.mean([x['retained_24h'] for x in l_r])
            res[name] = {'score': float(m_ret)}
        return {'best_architecture': max(res, key=lambda k: res[k]['score']), 'all_architectures': res}

    def _get_phase_id(self, coord):
        idx = tuple(np.clip(((coord)/self.voxel_grid.resolution).astype(int), 0, np.array(self.voxel_grid.shape)-1))
        return 1 if self.voxel_grid.lipid_density[idx] > 0.3 else (2 if self.voxel_grid.steric[idx] > 0.5 else 0)
