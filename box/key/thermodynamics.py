import numpy as np
from typing import Dict, List, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFreeSASA, AllChem
from .models import Molecule, EnvironmentalConditions
from .voxel_grid import EnhancedVoxelGrid
from .physics import compute_hansen_chi
from .mechanisms import compute_voxel_overlap_integral

def compute_flory_huggins_phase_behavior(apis: List[Molecule], excipients: List[Molecule], env: EnvironmentalConditions, grid: EnhancedVoxelGrid) -> Dict:
    """Calculates phase separation risk and mixing energy."""
    R, T = 8.314, env.temperature + 273.15
    poly_d, solv_d = grid.viscosity, grid.solvent_accessible
    total = poly_d + solv_d + 1e-10; phi_p, phi_s = poly_d / total, solv_d / total
    hansen_results, N_p_def, N_s_def = [], 100.0, 1.0
    for api in apis:
        for exc in excipients:
            api_smi, exc_smi = getattr(api, 'smiles', None), getattr(exc, 'smiles', None)
            if api_smi and exc_smi:
                hr = compute_hansen_chi(api_smi, exc_smi, T_celsius=env.temperature)
                if hr: hansen_results.append(hr)
    if hansen_results:
        chi_scalar = np.mean([r['chi'] for r in hansen_results])
        chi_crit = np.mean([r['chi_critical'] for r in hansen_results])
        N_p_def = np.mean([r['N_p'] for r in hansen_results])
        N_s_def = np.mean([r['N_s'] for r in hansen_results])
    else: 
        chi_scalar, chi_crit = (0.5 + 100.0 / T), 0.5 * (1 / np.sqrt(N_p_def) + 1 / np.sqrt(N_s_def))**2
    
    h_norm = grid.hydrophobic / (np.mean(np.abs(grid.hydrophobic)) + 1e-10)
    chi_field = chi_scalar * (1.0 + 0.1 * np.abs(h_norm - np.mean(h_norm)))
    dg = ((phi_p / N_p_def) * np.log(phi_p + 1e-10) + (phi_s / N_s_def) * np.log(phi_s + 1e-10) + chi_field * phi_p * phi_s) * R * T
    risk = (chi_field > chi_crit).astype(float)
    from scipy import ndimage
    labels, n = ndimage.label(risk > 0.5)
    return {'chi': chi_field, 'chi_scalar': float(chi_scalar), 'chi_critical': float(chi_crit), 'dg': dg, 'risk': risk, 'labels': labels, 'n': n, 'hansen_pairs': hansen_results}

def compute_dissolution_profile(apis: List[Molecule], excipients: List[Molecule], env: EnvironmentalConditions, grid: EnhancedVoxelGrid, 
                                time_points: int = 100, total_time: float = 7200.0, particle_d_um: float = 10.0, 
                                rho_g_cm3: float = 1.3, V_mL: float = 900.0, mass_mg: float = 500.0, h_cm: float = 0.003) -> Dict:
    """Noyes-Whitney dissolution model using SASA and voxel topology."""
    api_smiles = next((getattr(api, 'smiles', None) for api in apis if getattr(api, 'smiles', None)), None)
    exc_smiles = next((getattr(exc, 'smiles', None) for exc in excipients if getattr(exc, 'smiles', None)), None)
    
    if not api_smiles:
        from scipy import ndimage
        A_shell = np.sum((grid.steric > 0.1) & ~ndimage.binary_erosion(grid.steric > 0.1)) * (grid.resolution * 1e-8) ** 2
        k, t = (1e-5 * A_shell) / h_cm, np.linspace(0, total_time, time_points); dt = t[1] - t[0]
        M_d, C, M_t = np.zeros(time_points), np.zeros(time_points), np.sum(grid.steric) * 1e-15; M_r = M_t
        for i in range(1, time_points):
            if M_r <= 0: M_d[i], C[i] = M_t, M_t / V_mL; continue
            dm = min(k * max(1.0 - C[i-1], 0) * dt, M_r); M_d[i], M_r = M_d[i-1] + dm, M_r - dm; C[i] = M_d[i] / V_mL
        p = M_d / (M_t + 1e-10) * 100; t50 = float(t[np.argmax(p >= 50)]) if np.max(p) >= 50 else total_time
        return {'time': t, 'pct_dissolved': p, 'T50_s': t50, 'D_cm2s': 1e-5, 'Cs_mg_mL': 1.0, 'A_eff_cm2': float(A_shell), 'f_polar': 0.5, 'wetting_factor': 1.0, 'logS_ESOL': None, 'note': 'Legacy voxel-shell fallback'}

    mol_api = Chem.MolFromSmiles(api_smiles); mw = float(Descriptors.MolWt(mol_api)) if mol_api else 250.0
    T_K, kB, Na, eta = env.temperature + 273.15, 1.380649e-23, 6.022e23, 0.00089
    r_mol = (3 * mw / (4 * np.pi * Na * rho_g_cm3 * 1e6)) ** (1/3)
    D = float(kB * T_K / (6 * np.pi * eta * r_mol) * 1e4)
    logP = Descriptors.MolLogP(mol_api) if mol_api else 0.0
    rings = mol_api.GetRingInfo().NumRings() if mol_api else 0
    n_sp3 = sum(1 for a in mol_api.GetAtoms() if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.rdchem.HybridizationType.SP3) if mol_api else 0
    n_c = sum(1 for a in mol_api.GetAtoms() if a.GetAtomicNum() == 6) if mol_api else 1
    logS = 0.16 - 0.63*logP - 0.0062*mw + 0.066*rings - 0.74*(n_sp3/n_c); Cs_n = float(10**logS * mw); Cs, pH = Cs_n, env.pH
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

def compute_reaction_integrals(apis: List[Molecule], excipients: List[Molecule], system: Any) -> float:
    """Integrated reactive site overlap across population."""
    t = 0.0
    if not apis or not excipients: return 0.0
    for a in apis:
        r_a = system.project_enhanced_molecule_fields(a)['reactive_sites']
        for e in excipients: 
            r_e = system.project_enhanced_molecule_fields(e)['reactive_sites']
            t += np.sum(r_a * r_e)
    return float(t)

def monte_carlo_uncertainty_analysis(system: Any, n: int = 50) -> Dict:
    """Stochastic sensitivity analysis of VOI signals."""
    results = []
    base_coords = {id(mol): mol.coords.copy() for mol in system.apis + system.excipients}
    
    for _ in range(n):
        for mol in system.apis + system.excipients:
            mol.coords = base_coords[id(mol)] + np.random.normal(0, 0.2, mol.coords.shape)
        system.compute_multi_molecule_superposition()
        results.append(compute_voxel_overlap_integral(system))
        
    for mol in system.apis + system.excipients:
        mol.coords = base_coords[id(mol)]
    
    keys = results[0].keys()
    stats = {}
    for k in keys:
        vals = [r[k] for r in results]
        stats[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'relative_std': float(np.std(vals) / (np.mean(vals) + 1e-10))}
    return stats
