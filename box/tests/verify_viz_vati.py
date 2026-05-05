"""
verify_viz_vati.py
Tests the trajectory visualizer and VATI module.
Produces 'aspirin_trajectory.html' and reports VATI delta_G.
"""
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from key import NibbleEngine
from key.designer import create_enhanced_molecule_from_smiles
from key.nibble_bridge import LangevinMol
from key.trajectory_viewer import TrajectoryCapture
from key.vati import run_vati

PDB_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nibble', '1CRN.pdb'))
POCKET_CTR  = (17.0, 13.0, 11.0)
POCKET_RAD  = 10.0
START_C     = (POCKET_CTR[0] - POCKET_RAD, POCKET_CTR[1] - POCKET_RAD, POCKET_CTR[2] - POCKET_RAD)
OUT_DIR     = os.path.dirname(__file__)

print("=" * 60)
print("KeyBox: Trajectory Viewer + VATI Benchmark")
print("=" * 60)

# Load protein
print("\nLoading protein pocket...")
t0 = time.perf_counter()
protein = NibbleEngine(dim_x=80, dim_y=80, dim_z=80, resolution=0.25)
protein.load_pdb_pocket(PDB_PATH, POCKET_CTR, POCKET_RAD, blur_radius=1.5)
protein.backend.compute_water_network()
print(f"  Loaded in {(time.perf_counter()-t0)*1e3:.1f} ms")

# Build drug
aspirin = create_enhanced_molecule_from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin", concentration=0.3)
print(f"  Drug: {aspirin.name}, {len(aspirin.coords)} atoms")

# Trajectory Capture
print("\n--- Trajectory Capture (200 steps, record every 5) ---")
lm = LangevinMol(protein.backend, aspirin.coords, aspirin.charges, aspirin.hydrophobicity, START_C)
cap = TrajectoryCapture(protein.backend, lm, n_steps=200, record_every=5)
best_sc, best_pos = cap.run()
print(f"  Best score: {best_sc:,.1f}  at grid pos {best_pos}")
print(f"  Frames captured: {len(cap.frames_cx)}")

html_path = os.path.join(OUT_DIR, "aspirin_trajectory.html")
cap.export_html(html_path, start_coords=START_C, resolution=0.25)

# VATI
print("\n--- VATI (10 lambda windows, 50 steps each) ---")
lm2 = LangevinMol(protein.backend, aspirin.coords, aspirin.charges, aspirin.hydrophobicity, START_C)
result = run_vati(protein.backend, lm2, n_lambda=10, steps_per_window=50)
print(f"  {result}")
print(f"  Window scores: {[f'{s:.1f}' for s in result.window_scores]}")

print("\n" + "=" * 60)
print("DONE. Open aspirin_trajectory.html to view the binding animation.")
print("=" * 60)
