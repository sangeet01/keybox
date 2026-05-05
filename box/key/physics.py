import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from rdkit import Chem
from rdkit.Chem import Descriptors

class MultiComponentSystem:
    """N-component Flory-Huggins with ternary interactions"""
    
    @staticmethod
    def compute_ternary_chi(chi_12, chi_13, chi_23, phi_1, phi_2, phi_3):
        """Ternary Flory-Huggins interaction parameter"""
        chi_123 = (phi_1*phi_2*chi_12 + phi_1*phi_3*chi_13 + phi_2*phi_3*chi_23) / \
                  (phi_1*phi_2 + phi_1*phi_3 + phi_2*phi_3 + 1e-10)
        return chi_123
    
    @staticmethod
    def competitive_adsorption(excipients, api_surface_area):
        """Langmuir competitive adsorption model"""
        K_ads = [exc.get('adsorption_constant', 1.0) for exc in excipients]
        C = [exc.get('concentration', 1.0) for exc in excipients]
        
        theta = []
        for i in range(len(excipients)):
            numerator = K_ads[i] * C[i]
            denominator = 1 + sum(K_ads[j] * C[j] for j in range(len(excipients)))
            theta.append(numerator / denominator)
        
        return np.array(theta)
    
    @staticmethod
    def reaction_network(components, rate_constants, time_points=100):
        """Multi-step reaction kinetics: A+B->C, C+D->E"""
        n = len(components)
        C = np.array([comp['concentration'] for comp in components])
        
        dC_dt = np.zeros((time_points, n))
        dC_dt[0] = C
        
        dt = 1.0
        for t in range(1, time_points):
            rates = np.zeros(n)
            for rxn in rate_constants:
                i, j, k = rxn['reactants'][0], rxn['reactants'][1], rxn['product']
                rate = rxn['k'] * dC_dt[t-1, i] * dC_dt[t-1, j]
                rates[i] -= rate
                rates[j] -= rate
                rates[k] += rate
            
            dC_dt[t] = dC_dt[t-1] + rates * dt
            dC_dt[t] = np.maximum(dC_dt[t], 0)
        
        return dC_dt

class ProcessPhysics:
    """Link molecular properties to manufacturing processes"""
    
    @staticmethod
    def griffith_fracture_criterion(chi, elastic_modulus, surface_energy):
        gamma = surface_energy * (1 + chi)
        crack_length = 0.1
        sigma_critical = np.sqrt(2 * elastic_modulus * gamma / (np.pi * crack_length))
        return sigma_critical
    
    @staticmethod
    def percolation_threshold(phi_api, phi_excipients, coordination_number=6):
        p_c = 1.0 / (coordination_number - 1)
        if phi_api > p_c:
            return True, phi_api / p_c
        return False, 0.0
    
    @staticmethod
    def compression_induced_amorphization(pressure, chi, Tg):
        P_critical = 100
        amorphization_risk = (pressure / P_critical) * chi
        T_rise = 0.1 * pressure
        if T_rise > (Tg - 273.15):
            amorphization_risk *= 2
        return min(amorphization_risk, 1.0)

    @staticmethod
    def fickian_coating_diffusion(D, thickness, time_points=100):
        """Coating barrier effectiveness via Fick's law"""
        L = thickness  # um
        t = np.linspace(0, 3600, time_points)  # 1 hour
        M_rel = 1 - np.exp(-D * t / (L**2))
        return t, M_rel

class MicrostructurePhysics:
    """Phase-field and percolation models for spatial heterogeneity"""
    
    @staticmethod
    def cahn_hilliard_phase_field(chi, phi, dx=1.0, dt=0.01, steps=100):
        M = 1.0
        kappa = 0.5
        phi_history = [phi.copy()]
        
        for _ in range(steps):
            mu = chi * (1 - 2*phi)
            lap_phi = MicrostructurePhysics._laplacian_2d(phi, dx)
            mu_field = mu - kappa * lap_phi
            lap_mu = MicrostructurePhysics._laplacian_2d(mu_field, dx)
            phi += M * dt * lap_mu
            phi = np.clip(phi, 0, 1)
            phi_history.append(phi.copy())
        
        return phi_history
    
    @staticmethod
    def _laplacian_2d(field, dx):
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        ) / (dx**2)
        return lap
    
    @staticmethod
    def interface_energy(chi, surface_tension_api, surface_tension_exc):
        gamma_12 = surface_tension_api + surface_tension_exc - \
                   2 * np.sqrt(surface_tension_api * surface_tension_exc) * (1 - chi)
        return max(gamma_12, 0)
    
    @staticmethod
    def particle_packing_coordination(phi_solid, particle_size_ratio):
        phi_rcp = 0.64
        if phi_solid < phi_rcp:
            z = 6 * (phi_solid / phi_rcp)
        else:
            z = 6 + 2 * (phi_solid - phi_rcp) / (1 - phi_rcp)
        z *= (1 + 0.1 * particle_size_ratio)
        return z

class TemporalPhysics:
    """Kinetics, diffusion, nucleation, and aging"""
    
    @staticmethod
    def classical_nucleation_theory(delta_G_bulk, gamma, T):
        k_B = 1.38e-23
        r_critical = -2 * gamma / delta_G_bulk if delta_G_bulk < 0 else np.inf
        if delta_G_bulk < 0:
            delta_G_star = 16 * np.pi * gamma**3 / (3 * delta_G_bulk**2)
        else:
            delta_G_star = np.inf
        J0 = 1e20
        J = J0 * np.exp(-delta_G_star / (k_B * T)) if delta_G_star < np.inf else 0
        return r_critical, delta_G_star, J
    
    @staticmethod
    def wlf_time_temperature_superposition(T, T_ref, C1=17.44, C2=51.6):
        log_aT = -C1 * (T - T_ref) / (C2 + T - T_ref)
        a_T = 10**log_aT
        return a_T
    
    @staticmethod
    def avrami_crystallization(k, n, t):
        X = 1 - np.exp(-k * t**n)
        return X

    @staticmethod
    def haber_weiss_fenton_rate(metal_density, peroxide_density, oxygen_level, T):
        """
        Simplified radical generation rate: Rate = k * [M] * [H2O2] * [O2]
        Ea ~ 40-60 kJ/mol (Fenton)
        """
        R = 8.314
        Ea = 50000.0
        k_ref = 1e4
        k = k_ref * np.exp(-Ea / (R * T))
        rate = k * metal_density * peroxide_density * (oxygen_level / 0.21)
        return rate

    @staticmethod
    def fick_second_law_1d(C0, D, L, time_points=100, total_time=3600):
        nx = 50
        x = np.linspace(0, L, nx)
        dx = x[1] - x[0]
        t = np.linspace(0, total_time, time_points)
        dt = t[1] - t[0]
        dt_max = dx**2 / (2 * D)
        if dt > dt_max:
            dt = dt_max * 0.5
            t = np.linspace(0, total_time, int(total_time / dt))
        C = np.zeros((len(t), nx))
        C[0] = C0
        for i in range(1, len(t)):
            C[i, 1:-1] = C[i-1, 1:-1] + D * dt / dx**2 * (
                C[i-1, 2:] - 2*C[i-1, 1:-1] + C[i-1, :-2]
            )
            C[i, 0] = C[i, 1]; C[i, -1] = C[i, -2]
        return x, t, C

class DosageSpecificPhysics:
    """Tier 3: Specialized physics for Suspension and Injectable forms"""
    
    @staticmethod
    def stokes_sedimentation_velocity(r_um, rho_p, rho_f, eta):
        """
        v_s = 2r^2(rho_p - rho_f)g / (9*eta)
        rho in g/cm3, r in um, eta in Pa*s, v_s in um/s
        """
        g = 9.81
        r_m = r_um * 1e-6
        delta_rho = (rho_p - rho_f) * 1000.0  # kg/m3
        v_s_m = (2 * r_m**2 * delta_rho * g) / (9 * eta)
        return v_s_m * 1e6  # um/s
    
    @staticmethod
    def ostwald_ripening_rate(D_s, gamma, C_inf, V_m, T):
        """
        LSW theory: dR^3/dt = 8*D_s*gamma*C_inf*V_m / (9*R*T)
        """
        R = 8.314
        rate = (8 * D_s * gamma * C_inf * V_m) / (9 * R * T)
        return rate

    @staticmethod
    def compute_osmolarity(concentrations, vant_hoff_factors):
        """
        Osmolarity = Sum(c_i * phi_i) * 1000  (mOsm/L)
        """
        return np.sum(np.array(concentrations) * np.array(vant_hoff_factors)) * 1000.0
    
    @staticmethod
    def spatial_aggregation_propensity(hydrophobic_field, charge_field):
        """
        SAP proxy: spatial overlap of hydrophobic patches and neutral charge
        """
        sap_field = hydrophobic_field * (1.0 - np.abs(charge_field))
        return np.sum(sap_field)

def _estimate_hansen_params(smiles: str):
    """
    Estimate Hansen solubility parameters (MPa^0.5) from SMILES via RDKit descriptors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    logP  = Descriptors.MolLogP(mol)
    tpsa  = Descriptors.TPSA(mol)
    mw    = max(Descriptors.MolWt(mol), 1.0)
    hbd   = Descriptors.NumHDonors(mol)
    hba   = Descriptors.NumHAcceptors(mol)
    rings = mol.GetRingInfo().NumRings()

    cooh_pat = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    phen_pat = Chem.MolFromSmarts('[cX3][OX2H]')
    n_cooh   = len(mol.GetSubstructMatches(cooh_pat)) if cooh_pat else 0
    n_phenol = len(mol.GetSubstructMatches(phen_pat)) if phen_pat else 0

    delta_d = 17.0 - 0.95 * logP + 0.4 * rings
    delta_p = 13.5 * np.sqrt(max(tpsa / mw, 0.0))
    delta_h = (1.8 * 3.0 * np.sqrt(max(hbd + 0.5 * hba, 0.0))
               + 6.0 * n_cooh + 3.5 * n_phenol)
    delta_t = np.sqrt(delta_d**2 + delta_p**2 + delta_h**2)
    return delta_d, delta_p, delta_h, delta_t, mw

def compute_hansen_chi(smiles_api: str, smiles_exc: str,
                        T_celsius: float = 25.0,
                        V_ref: float = 100.0) -> Optional[Dict]:
    """
    Compute Flory-Huggins chi from Hansen solubility parameters.
    """
    p_api = _estimate_hansen_params(smiles_api)
    p_exc = _estimate_hansen_params(smiles_exc)
    if p_api is None or p_exc is None:
        return None

    d1, p1, h1, t1, mw1 = p_api
    d2, p2, h2, t2, mw2 = p_exc
    T = T_celsius + 273.15
    R = 8.314  # J/(mol*K)

    delta_sq = (d1 - d2)**2 + 0.25 * (p1 - p2)**2 + 0.25 * (h1 - h2)**2
    chi = V_ref * delta_sq / (R * T)

    N_p = max(mw2 / V_ref, 1.0)   
    N_s = max(mw1 / V_ref, 1.0)   
    chi_critical = 0.5 * (1.0 / np.sqrt(N_p) + 1.0 / np.sqrt(N_s))**2

    if chi < chi_critical:
        risk_label = 'Miscible'
    elif chi < 1.0:
        risk_label = 'Marginally miscible'
    elif chi < 2.0:
        risk_label = 'Tendency to phase separate'
    else:
        risk_label = 'Phase separation expected'

    return {
        'chi':          float(chi),
        'chi_critical': float(chi_critical),
        'risk_label':   risk_label,
        'delta_api':    {'d': d1, 'p': p1, 'h': h1, 'total': t1},
        'delta_exc':    {'d': d2, 'p': p2, 'h': h2, 'total': t2},
        'delta_sq':     float(delta_sq),
        'N_p':          float(N_p),
        'N_s':          float(N_s),
    }
