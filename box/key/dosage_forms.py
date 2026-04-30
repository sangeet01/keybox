import numpy as np
from typing import List, Dict, Any, Tuple
from rdkit import Chem
from .models import (
    Molecule, EnvironmentalConditions, 
    DosageFormConfig, DosageFormResult, DosageForm
)
from .voxel_grid import EnhancedVoxelGrid
from .physics import (
    ProcessPhysics, MicrostructurePhysics, 
    TemporalPhysics, DosageSpecificPhysics
)

def _ionisation_fraction(smiles: str, pH: float) -> float:
    """Estimates ionized fraction using basic SMARTS matching."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if not mol: return 0.5
    acid_p, base_p = Chem.MolFromSmarts('[CX3](=O)[OX2H1]'), Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]')
    if mol.HasSubstructMatch(acid_p): return 1.0 / (1.0 + 10**(4.5 - pH))
    if mol.HasSubstructMatch(base_p): return 1.0 / (1.0 + 10**(pH - 9.0))
    return 0.1

def _logP_lipid_fraction(smiles: str) -> float:
    """Estimates lipid partition fraction based on LogP."""
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if not mol: return 0.5
    P = 10 ** Descriptors.MolLogP(mol)
    return float(P / (P + 1.0))

def run_dosage_form_physics(apis: List[Molecule], excipients: List[Molecule], env: EnvironmentalConditions, grid: EnhancedVoxelGrid, 
                           cfg: DosageFormConfig, engine_ref) -> DosageFormResult:
    """Dispatches physical simulations specific to the dosage form."""
    T_K = env.temperature + 273.15
    f = cfg.form_type
    
    if f == DosageForm.ORAL_SOLID:
        a_T = TemporalPhysics.wlf_time_temperature_superposition(env.temperature, 25.0)
        X_cr = float(TemporalPhysics.avrami_crystallization(1e-12, 2.0, max(env.storage_time * 2592000, 1.0)))
        return DosageFormResult(
            form_type=f.value, 
            ocs_modifier=-10.0 * X_cr, 
            modifier_label=f"Cryst. penalty ({X_cr:.2%} conv.)", 
            extra_metrics={'wlf_shift_factor': float(a_T), 'crystallised_fraction': X_cr, 'storage_time_months': env.storage_time}
        )
        
    elif f == DosageForm.SOLUTION:
        I, eps = max(env.ionic_strength, 1e-6), env.dielectric_constant
        d_r = float(np.clip(np.sqrt(eps / (78.5 * max(I, 0.01) / 0.15)), 0.2, 5.0))
        m_i = float(np.mean([_ionisation_fraction(a.smiles, env.pH) for a in apis if getattr(a, 'smiles', None)])) if apis else 0.0
        i_b, i_p = 8.0 * m_i * (eps / 78.5), 3.0 * min(I / 0.5, 1.0)
        return DosageFormResult(
            form_type=f.value, 
            ocs_modifier=i_b - i_p, 
            modifier_label=f"Soln: ion. bonus {i_b:.1f}, ionic-p {i_p:.1f}", 
            extra_metrics={'mean_ionisation_fraction': m_i, 'debye_length_ratio': d_r, 'ionic_strength': I, 'dielectric_constant': eps}
        )
        
    elif f == DosageForm.LIPID_BASED:
        m_l = float(np.mean([_logP_lipid_fraction(a.smiles) for a in apis if getattr(a, 'smiles', None)])) if apis else 0.5
        l_b = 12.0 * m_l * cfg.lipid_fraction
        e_p = sum(8.0 * cfg.lipid_fraction for a in apis if getattr(a, 'smiles', None) and Chem.MolFromSmiles(a.smiles).HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2]')))
        
        # This requires Flory-Huggins, which we moved to thermodynamics.py
        from .thermodynamics import compute_flory_huggins_phase_behavior
        fh = compute_flory_huggins_phase_behavior(apis, excipients, env, grid)
        c_s, c_c = fh.get('chi_scalar', 1.0), fh.get('chi_critical', 0.9)
        p_p, g_i = 5.0 * max(0.0, c_s - c_c), MicrostructurePhysics.interface_energy(c_s, 35.0, 25.0)
        
        return DosageFormResult(
            form_type=f.value, 
            ocs_modifier=l_b - e_p - p_p, 
            modifier_label=f"LBF: lip-b {l_b:.1f}, ester-p {e_p:.1f}, ph-sep-p {p_p:.1f}", 
            extra_metrics={'mean_lipid_partition_fraction': m_l, 'lipid_volume_fraction': cfg.lipid_fraction, 'flory_huggins_chi': c_s, 'chi_critical': c_c, 'interface_energy_mN_m': float(g_i), 'interfacial_stability_score': float(np.exp(-g_i / 20.0))}
        )
        
    elif f == DosageForm.CONTROLLED_RELEASE:
        D_T = cfg.coating_diffusivity * np.exp(50000.0 / 8.314 * (1 / 298.15 - 1 / T_K))
        c_i = float(np.clip(1.0 - ProcessPhysics.fickian_coating_diffusion(D_T, cfg.coating_thickness_um, 50)[1][-1], 0.0, 1.0))
        c_r = float(np.mean(TemporalPhysics.fick_second_law_1d(1.0, D_T * 0.1, cfg.coating_thickness_um * 10, 30, 86400.0)[2][:, -1]))
        c_b, b_p, m_p = 10.0 * c_i * c_r, 8.0 * (1.0 - c_i), 5.0 * env.moisture * (1.0 - c_i)
        return DosageFormResult(
            form_type=f.value, 
            ocs_modifier=c_b - b_p - m_p, 
            modifier_label=f"CR: int-b {c_b:.1f}, burst-p {b_p:.1f}, moist-p {m_p:.1f}", 
            extra_metrics={'coating_thickness_um': cfg.coating_thickness_um, 'coating_diffusivity_cm2s': float(D_T), 'coating_integrity_1h': c_i, 'core_retention_24h': c_r, 'temperature_celsius': env.temperature}
        )
        
    elif f == DosageForm.MULTI_PHASE:
        p0 = (grid.hydrophobic - grid.hydrophobic.min()) / (np.ptp(grid.hydrophobic) + 1e-10)
        from .thermodynamics import compute_flory_huggins_phase_behavior
        fh = compute_flory_huggins_phase_behavior(apis, excipients, env, grid)
        c_s, c_c = fh.get('chi_scalar', 1.0), fh.get('chi_critical', 0.9)
        
        p_f = MicrostructurePhysics.cahn_hilliard_phase_field(c_s, p0[:, :, p0.shape[2]//2].copy(), 1.0, 0.005, 20)[-1]
        g_i, o_r = MicrostructurePhysics.interface_energy(c_s, 35.0, 30.0), float(np.clip(2.0 * MicrostructurePhysics.interface_energy(c_s, 35.0, 30.0) * 1e-3 * 1e-4 / (8.314 * T_K * cfg.droplet_diameter_um / 2.0 * 1e-6), 0.0, 1.0))
        s_b, o_p = 8.0 * float(np.exp(-max(0.0, c_s - c_c))), 5.0 * o_r
        return DosageFormResult(
            form_type=f.value, 
            ocs_modifier=s_b - o_p, 
            modifier_label=f"Multi-phase: stab-b {s_b:.1f}, Ost-p {o_p:.1f}", 
            extra_metrics={'flory_huggins_chi': c_s, 'chi_critical': c_c, 'interfacial_stability_score': float(np.exp(-max(0.0, c_s - c_c))), 'interface_energy_mN_m': float(g_i), 'interface_fraction': int(np.sum((p_f > 0.2) & (p_f < 0.8))) / p_f.size, 'droplet_diameter_um': cfg.droplet_diameter_um, 'ostwald_ripening_risk': o_r}
        )
        
    elif f == DosageForm.SUSPENSION:
        eta = cfg.viscosity_cP * 0.001
        v_s = DosageSpecificPhysics.stokes_sedimentation_velocity(cfg.particle_diameter_um / 2.0, cfg.particle_density_g_cm3, 1.0, eta)
        wet_comp = float(np.exp(-np.mean(np.abs(grid.hydrophobic)) / 2.0))
        stab_b = 5.0 * np.log1p(cfg.suspending_agent_conc)
        ocs_mod = 10.0 * wet_comp - 5.0 * np.log1p(v_s) + stab_b
        return DosageFormResult(
            form_type=f.value, 
            ocs_modifier=ocs_mod, 
            modifier_label=f"Suspension: wet-b {10*wet_comp:.1f}, sed-p {5*np.log1p(v_s):.1f}, stab-b {stab_b:.1f}", 
            extra_metrics={'sedimentation_velocity_um_s': float(v_s), 'wetting_compatibility': wet_comp, 'particle_diameter_um': cfg.particle_diameter_um, 'particle_density_g_cm3': cfg.particle_density_g_cm3, 'viscosity_cP': cfg.viscosity_cP, 'suspending_agent_conc': cfg.suspending_agent_conc}
        )
        
    elif f == DosageForm.INJECTABLE:
        osm = DosageSpecificPhysics.compute_osmolarity([a.concentration for a in apis], [1.0] * len(apis))
        sap = DosageSpecificPhysics.spatial_aggregation_propensity(grid.hydrophobic, grid.electrostatic)
        osm_err = abs(osm - cfg.target_osmolarity) / cfg.target_osmolarity
        ocs_mod = -20.0 * osm_err - 10.0 * np.log1p(max(sap, 0))
        return DosageFormResult(
            form_type=f.value, 
            ocs_modifier=ocs_mod, 
            modifier_label=f"Injectable: osm-p {20*osm_err:.1f}, agg-p {10*np.log1p(max(sap, 0)):.1f}", 
            extra_metrics={'calculated_osmolarity': float(osm), 'aggregation_propensity': float(sap), 'target_osmolarity': cfg.target_osmolarity}
        )
        
    return DosageFormResult(form_type=str(f), ocs_modifier=0.0, modifier_label=f"Unknown form {f}")
