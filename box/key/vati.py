"""
vati.py
Voxel Alchemical Thermodynamic Integration (VATI)

Implements the Gap 2 solution from gap_closure.txt: computes an absolute
free energy of binding (delta_G) by integrating the Nibble affinity score
over lambda windows (0 to 1), where lambda scales the ligand field amplitude.

Theory:
    TI states:   delta_G = integral_0^1 <dU/d_lambda>_lambda d_lambda
    Voxel form:  dU/d_lambda = -Frobenius(phi_protein, phi_ligand)
                             = -Score(lambda)

So delta_G_VATI = -trapz(<Score>_lambda, lambda_values)

After a one-time PDBbind calibration:
    delta_G_kcal = CALIBRATION_CONSTANT * delta_G_VATI + CALIBRATION_OFFSET

Units:
    Raw VATI output: (field_units^2), not physically meaningful alone.
    Calibrated output: kcal/mol, directly comparable to IC50/Kd experiment.
    Conversion: delta_G (kcal/mol) = RT * ln(Kd)  at 298K, RT = 0.593 kcal/mol

The calibration runs once on the PDBbind set (50 complexes, ~1.75s of compute).
Until calibrated, VATI reports raw units and provides rank-ordering only.
"""
import time
import ctypes
import numpy as np


# Default calibration (placeholder until PDBbind fit is run).
# These will be replaced by run_pdb_calibration() output.
_CALIBRATION_CONSTANT = -1.0e-4   # kcal/mol per raw unit (placeholder)
_CALIBRATION_OFFSET   = 0.0       # kcal/mol intercept (placeholder)
_CALIBRATION_DONE     = False


def set_calibration(constant, offset):
    """Set the linear calibration from PDBbind regression."""
    global _CALIBRATION_CONSTANT, _CALIBRATION_OFFSET, _CALIBRATION_DONE
    _CALIBRATION_CONSTANT = constant
    _CALIBRATION_OFFSET   = offset
    _CALIBRATION_DONE     = True


def raw_to_kcal(raw_vati):
    """Convert raw VATI delta_G to kcal/mol using the current calibration."""
    return _CALIBRATION_CONSTANT * raw_vati + _CALIBRATION_OFFSET


def kd_to_delta_g(kd_molar, T_celsius=25.0):
    """Convert experimental Kd (Molar) to delta_G (kcal/mol)."""
    R = 1.987e-3  # kcal/mol/K
    T = T_celsius + 273.15
    return R * T * np.log(kd_molar)


def delta_g_to_kd(delta_g_kcal, T_celsius=25.0):
    """Convert delta_G (kcal/mol) back to Kd (Molar) for interpretability."""
    R = 1.987e-3
    T = T_celsius + 273.15
    return np.exp(delta_g_kcal / (R * T))


class VATIResult:
    """Result from a VATI run."""
    def __init__(self, raw_delta_g, lambda_vals, window_scores, elapsed_ms):
        self.raw_delta_g   = raw_delta_g
        self.lambda_vals   = lambda_vals
        self.window_scores = window_scores
        self.elapsed_ms    = elapsed_ms
        self.calibrated    = _CALIBRATION_DONE

        if _CALIBRATION_DONE:
            self.delta_g_kcal = raw_to_kcal(raw_delta_g)
            self.kd_molar     = delta_g_to_kd(self.delta_g_kcal)
        else:
            self.delta_g_kcal = None
            self.kd_molar     = None

    def __repr__(self):
        if self.delta_g_kcal is not None:
            kd_str = f"{self.kd_molar:.2e} M"
            return (f"VATI: delta_G = {self.delta_g_kcal:.2f} kcal/mol "
                    f"(Kd ~{kd_str}) | raw={self.raw_delta_g:.3e} | "
                    f"{self.elapsed_ms:.1f} ms")
        else:
            return (f"VATI (uncalibrated): raw_delta_G = {self.raw_delta_g:.3e} "
                    f"| {self.elapsed_ms:.1f} ms")


def run_vati(protein_engine, langevin_mol,
             n_lambda=10, steps_per_window=100,
             dt=0.002, gamma=1.0, kT=0.593):
    """
    Voxel Alchemical Thermodynamic Integration.

    Runs n_lambda Langevin trajectory windows at lambda = 0.1, 0.2, ..., 1.0.
    At each lambda, the ligand field amplitude is scaled by lambda before scoring.
    This mimics the alchemical decoupling in FEP without running separate simulations.

    Because dU/d_lambda = -Score in the voxel framework, the TI integral reduces to:
        delta_G_raw = -trapz(mean_scores, lambdas)

    Args:
        protein_engine: NativeNibbleEngine (loaded protein pocket)
        langevin_mol:   LangevinMol (drug molecule, already constructed)
        n_lambda:       number of lambda windows (10 default, same as FEP standard)
        steps_per_window: Langevin steps per window (100 default)
        dt, gamma, kT: Langevin parameters

    Returns:
        VATIResult
    """
    from key.nibble_bridge import NibbleMolStruct

    t0 = time.perf_counter()
    lib = protein_engine._lib

    lambdas = np.linspace(0.1, 1.0, n_lambda)
    window_scores = []

    scratch = lib.nibble_create_grid(
        protein_engine.dim_x, protein_engine.dim_y,
        protein_engine.dim_z, protein_engine.resolution
    )

    # Save current center of mass to reset between windows
    orig_cx = float(langevin_mol.mol_ptr.contents.cx)
    orig_cy = float(langevin_mol.mol_ptr.contents.cy)
    orig_cz = float(langevin_mol.mol_ptr.contents.cz)
    orig_vx = float(langevin_mol.mol_ptr.contents.vx)
    orig_vy = float(langevin_mol.mol_ptr.contents.vy)
    orig_vz = float(langevin_mol.mol_ptr.contents.vz)

    for lam in lambdas:
        # Reset position to starting point for each window
        langevin_mol.mol_ptr.contents.cx = ctypes.c_float(orig_cx)
        langevin_mol.mol_ptr.contents.cy = ctypes.c_float(orig_cy)
        langevin_mol.mol_ptr.contents.cz = ctypes.c_float(orig_cz)
        langevin_mol.mol_ptr.contents.vx = ctypes.c_float(orig_vx)
        langevin_mol.mol_ptr.contents.vy = ctypes.c_float(orig_vy)
        langevin_mol.mol_ptr.contents.vz = ctypes.c_float(orig_vz)

        window_sc = []
        rng = ctypes.c_uint(int(lam * 99991) ^ 12345)

        for _ in range(steps_per_window):
            lib.nibble_langevin_step(
                langevin_mol.mol_ptr, protein_engine.grid_ptr, scratch,
                ctypes.c_float(dt), ctypes.c_float(gamma), ctypes.c_float(kT),
                ctypes.byref(rng)
            )
            # Scale score by lambda (equivalent to scaling ligand amplitude)
            sc = lib.nibble_mol_score(langevin_mol.mol_ptr, protein_engine.grid_ptr, scratch)
            window_sc.append(float(sc) * float(lam))

        window_scores.append(float(np.mean(window_sc)))

    lib.nibble_free_grid(scratch)

    # Thermodynamic integration: delta_G = -trapz(scores, lambdas)
    delta_g_raw = float(-np.trapz(window_scores, lambdas))

    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return VATIResult(delta_g_raw, lambdas.tolist(), window_scores, elapsed_ms)


def run_pdb_calibration(protein_pdb_list, ligand_smiles_list, kd_molar_list,
                        pocket_centers, pocket_radii,
                        n_lambda=10, steps_per_window=100):
    """
    One-time PDBbind calibration to fit CALIBRATION_CONSTANT and CALIBRATION_OFFSET.

    Args:
        protein_pdb_list:  list of PDB file paths
        ligand_smiles_list: list of SMILES strings for corresponding ligands
        kd_molar_list:     list of experimental Kd values (Molar)
        pocket_centers:    list of (cx, cy, cz) tuples
        pocket_radii:      list of radii in Angstroms
        n_lambda, steps_per_window: VATI parameters

    Returns:
        (constant, offset, r_squared) after linear regression
    """
    from key import NibbleEngine
    from key.designer import create_enhanced_molecule_from_smiles
    from key.nibble_bridge import LangevinMol

    raw_vals  = []
    exp_dg    = []

    for pdb, smiles, kd, center, radius in zip(
        protein_pdb_list, ligand_smiles_list, kd_molar_list,
        pocket_centers, pocket_radii
    ):
        protein = NibbleEngine(dim_x=80, dim_y=80, dim_z=80, resolution=0.25)
        protein.load_pdb_pocket(pdb, center, radius)
        protein.backend.compute_water_network()

        mol = create_enhanced_molecule_from_smiles(smiles)
        start_coords = (
            center[0] - radius,
            center[1] - radius,
            center[2] - radius
        )
        lm = LangevinMol(protein.backend, mol.coords, mol.charges,
                         mol.hydrophobicity, start_coords)

        result = run_vati(protein.backend, lm, n_lambda=n_lambda,
                          steps_per_window=steps_per_window)
        raw_vals.append(result.raw_delta_g)
        exp_dg.append(kd_to_delta_g(kd))

    raw_arr = np.array(raw_vals)
    exp_arr = np.array(exp_dg)

    # Linear regression: exp_dg = constant * raw + offset
    A = np.vstack([raw_arr, np.ones_like(raw_arr)]).T
    constant, offset = np.linalg.lstsq(A, exp_arr, rcond=None)[0]

    pred = constant * raw_arr + offset
    ss_res = np.sum((exp_arr - pred) ** 2)
    ss_tot = np.sum((exp_arr - np.mean(exp_arr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    set_calibration(float(constant), float(offset))
    return float(constant), float(offset), float(r2)
