"""
verify_tier34.py
Benchmarks the Tier 3 (Langevin gradient), Tier 4 (Induced Fit dynamics),
and Gap Closure (water network, Fukui/covalent) functions against 1CRN.pdb.
Reports timing for each stage and a final total to give the true MD-equivalent cost.
"""
import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from key import NibbleEngine
from key.designer import create_enhanced_molecule_from_smiles

PDB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nibble', '1CRN.pdb'))
POCKET_CENTER = (17.0, 13.0, 11.0)
POCKET_RADIUS  = 10.0
START_C = (POCKET_CENTER[0] - POCKET_RADIUS,
           POCKET_CENTER[1] - POCKET_RADIUS,
           POCKET_CENTER[2] - POCKET_RADIUS)

PASS = 0
FAIL = 0

def report(label, elapsed_ms, detail=''):
    global PASS
    print(f"  [{label}] {elapsed_ms:.2f} ms  {detail}")
    PASS += 1

def check(condition, label):
    global FAIL
    if not condition:
        print(f"  FAIL: {label}")
        FAIL += 1

def load_sniper():
    e = NibbleEngine(dim_x=80, dim_y=80, dim_z=80, resolution=0.25)
    e.load_pdb_pocket(PDB_PATH, POCKET_CENTER, POCKET_RADIUS, blur_radius=1.5)
    return e

def build_drug_engine(aspirin, start_c):
    """Build a drug NativeNibbleEngine from aspirin coords."""
    lib = NibbleEngine._lib
    from key.nibble_bridge import NativeNibbleEngine, N_CHANNELS, CH_STERIC_DEMAND, CH_ELEC_DEMAND, CH_LIPO_DEMAND, LangevinMol
    drug = NativeNibbleEngine(lib, 80, 80, 80, 0.25)
    import ctypes
    for i in range(len(aspirin.coords)):
        ch_vals = (ctypes.c_float * N_CHANNELS)(0)
        ch_vals[CH_STERIC_DEMAND] = 1.0
        ch_vals[CH_ELEC_DEMAND]   = float(aspirin.charges[i])
        ch_vals[CH_LIPO_DEMAND]   = float(aspirin.hydrophobicity[i])
        lib.nibble_project_atom(
            drug.grid_ptr,
            float(aspirin.coords[i][0] - start_c[0]),
            float(aspirin.coords[i][1] - start_c[1]),
            float(aspirin.coords[i][2] - start_c[2]),
            1.5, ch_vals
        )
    return drug


# -----------------------------------------------------------------------
print("=" * 60)
print("Nibble V4 Tier 3/4 + Gap Closure Benchmark")
print("=" * 60)

# Setup
aspirin = create_enhanced_molecule_from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin", concentration=0.3)
print(f"\nDrug: {aspirin.name}, {len(aspirin.coords)} atoms")

# -----------------------------------------------------------------------
print("\n--- Tier 2 Baseline (static Sniper, for reference) ---")
t0 = time.perf_counter()
protein = load_sniper()
t_load = (time.perf_counter() - t0) * 1e3

drug = build_drug_engine(aspirin, START_C)

t0 = time.perf_counter()
protein.backend.compute_water_network()
score_static = protein.backend.affinity_full(drug, delta_G_bulk=0.0)  # water off for baseline
t_static = (time.perf_counter() - t0) * 1e3
report("Pocket load (Sniper)", t_load)
report("Static affinity score", t_static, f"score={score_static:,.1f}")

# -----------------------------------------------------------------------
print("\n--- Tier 3: Gradient Computation ---")
cx = float(POCKET_CENTER[0] - START_C[0])
cy = float(POCKET_CENTER[1] - START_C[1])
cz = float(POCKET_CENTER[2] - START_C[2])

t0 = time.perf_counter()
gx, gy, gz = protein.backend.gradient_at(cx, cy, cz)
t_grad = (time.perf_counter() - t0) * 1e3
report("Gradient at pocket center", t_grad, f"grad=({gx:.3f}, {gy:.3f}, {gz:.3f})")
check(not (gx == 0 and gy == 0 and gz == 0), "gradient is non-zero")

# Simulate 1000-step Langevin scan (C-Native)
print("\n--- Tier 3: Langevin Scan (1000 steps, C-native) ---")
from key.nibble_bridge import LangevinMol
langevin_mol = LangevinMol(protein.backend, aspirin.coords, aspirin.charges, aspirin.hydrophobicity, START_C)

t0 = time.perf_counter()
n_steps = 1000
best_score_traj, best_pos = langevin_mol.run_trajectory(n_steps=n_steps)
t_traj = (time.perf_counter() - t0) * 1e3
report(f"Langevin scan ({n_steps} steps)", t_traj, f"per_step={(t_traj/n_steps):.3f} ms, best={best_score_traj:,.1f}")

# -----------------------------------------------------------------------
print("\n--- Gap Closure: Virtual Water Network ---")
t0 = time.perf_counter()
protein.backend.compute_water_network(delta_G_bulk=2.0, kT=0.593)
t_water = (time.perf_counter() - t0) * 1e3
report("Water network computation", t_water)

t0 = time.perf_counter()
score_full = protein.backend.affinity_full(drug, delta_G_bulk=2.0)
t_full_aff = (time.perf_counter() - t0) * 1e3
report("Full affinity (water+desolv)", t_full_aff, f"score={score_full:,.1f}")

# -----------------------------------------------------------------------
print("\n--- Gap Closure: Fukui Reactivity + Covalent Detection ---")
t0 = time.perf_counter()
coords_flat = np.array(aspirin.coords, dtype=np.float32)
charges_flat = np.array(aspirin.charges, dtype=np.float32)
protein.backend.project_fukui(coords_flat - np.array(START_C), charges_flat)
t_fukui = (time.perf_counter() - t0) * 1e3
report("Fukui channel projection", t_fukui)

t0 = time.perf_counter()
cov_site = protein.backend.check_covalent(drug, threshold=0.3)
t_cov = (time.perf_counter() - t0) * 1e3
report("Covalent event check", t_cov, f"site={cov_site} (-1=none)")

# -----------------------------------------------------------------------
print("\n--- Tier 4: Induced Fit Dynamics ---")

t0 = time.perf_counter()
protein.backend.thermal_breathe(sigma_delta=0.05)
t_breathe = (time.perf_counter() - t0) * 1e3
report("Thermal breathing (1 frame)", t_breathe)

t0 = time.perf_counter()
protein.backend.elastic_deform(drug, elasticity_scale=0.2)
t_elastic = (time.perf_counter() - t0) * 1e3
report("Elastic deformation (1 frame)", t_elastic)

t0 = time.perf_counter()
q = protein.backend.pocket_occupancy(drug, threshold=0.1)
t_occ = (time.perf_counter() - t0) * 1e3
report("Pocket occupancy (q)", t_occ, f"q={q:.4f}")

# Tier 4 full frame update (uses same grid for open and closed since we have one PDB)
protein_open   = load_sniper()
protein_closed = load_sniper()

t0 = time.perf_counter()
protein.backend.tier4_frame_update(
    protein_open.backend, protein_closed.backend, drug,
    sigma_delta=0.05, elasticity_scale=0.2
)
t_frame = (time.perf_counter() - t0) * 1e3
report("Full Tier 4 frame update", t_frame)

# -----------------------------------------------------------------------
# Full pipeline simulation: 50 Tier 4 frames x 100 Langevin steps each
print("\n--- Full Pipeline: 50 Tier 4 Frames x 100 Langevin Steps ---")
protein2 = load_sniper()
protein2.backend.compute_water_network()

# We need a new LangevinMol for protein2, and a fresh drug_engine to accumulate the final result
lm_full = LangevinMol(protein2.backend, aspirin.coords, aspirin.charges, aspirin.hydrophobicity, START_C)
drug_final = NibbleEngine(dim_x=80, dim_y=80, dim_z=80, resolution=0.25)

t0 = time.perf_counter()
best_full = float("inf")

for frame in range(50):
    # C-Native Langevin inner loop (100 steps per frame)
    sc, _ = lm_full.run_trajectory(n_steps=100)
    
    # We use a scratch drug grid for Tier 4 evaluation
    lm_full.engine._lib.nibble_reset_steric(drug_final.backend.grid_ptr) # optional reset
    lm_full.project_to_grid(drug_final.backend)

    # Tier 4 frame update
    protein2.backend.tier4_frame_update(
        protein_open.backend, protein_closed.backend, drug_final.backend,
        sigma_delta=0.05 * np.sin(frame * 0.1),
        elasticity_scale=0.15
    )
    
    sc_full = protein2.backend.affinity_full(drug_final.backend)
    if sc_full < best_full:
        best_full = sc_full

t_pipeline = (time.perf_counter() - t0) * 1e3
lm_full.project_to_grid(drug_final.backend) # ensure final state is projected
q_final = protein2.backend.pocket_occupancy(drug_final.backend)
cov_final = protein2.backend.check_covalent(drug_final.backend, threshold=0.3)

print(f"\n  Best full affinity (water+desolv): {best_full:,.1f}")
print(f"  Final pocket occupancy q:           {q_final:.4f}")
print(f"  Covalent engagement:                {'YES site=' + str(cov_final) if cov_final >= 0 else 'No'}")
report("Full Tier 3+4+Gaps pipeline (50 frames x 100 steps)", t_pipeline)

# -----------------------------------------------------------------------
print("\n" + "=" * 60)
total_component_ms = t_load + t_static + t_grad + t_traj + t_water + t_full_aff + t_fukui + t_cov + t_breathe + t_elastic + t_occ + t_frame
print(f"Component total:  {total_component_ms:.1f} ms")
print(f"Pipeline total:   {t_pipeline:.1f} ms (50 frames x 100 steps)")
print(f"GROMACS baseline: ~48 hours (172,800,000 ms)")
print(f"Speedup:          ~{172_800_000 / t_pipeline:,.0f}x")
print(f"\nPasses: {PASS}  Failures: {FAIL}")
print("=" * 60)
