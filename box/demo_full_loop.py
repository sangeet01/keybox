"""
KeyBox Full Loop Demo (v0.8.0)

End-to-end demonstration:
  Step 1  - BBD optimization: find best pH / Temperature / Moisture
  Step 2  - Combined Mixture-Process: optimize Filler/Binder/Disintegrant composition
  Step 3  - Run voxel physics at the optimal formulation point
  Step 4  - Export field maps to viz_output/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

SEP = "=" * 60

# ============================================================
# STEP 1: BOX-BEHNKEN DESIGN (Process Factors)
# ============================================================

print(SEP)
print("STEP 1: Box-Behnken Process Optimization")
print(SEP)

from key import KeyBoxOptimizer
from key.optimizer import generate_box_behnken

bbd_opt = KeyBoxOptimizer(
    api_smiles="CC(=O)Oc1ccccc1C(=O)O",
    api_name="Aspirin"
)

# Three independent process factors
bbd_opt.add_factor("pH",          low=4.0,  high=8.0)
bbd_opt.add_factor("Temperature", low=25.0, high=50.0)
bbd_opt.add_factor("Moisture",    low=0.2,  high=0.8)

# One fixed excipient at standard level
bbd_opt.add_fixed_excipient("MCC", "C1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O")

print("[INFO] Running BBD (15 design points)...")
bbd_results = bbd_opt.run_bbd_optimization(resolution=2.0, verbose=True)

print()
print("  BBD Design Table (first 5 rows):")
print(bbd_results["design_df"].head(5).to_string(index=False))

if "predicted_ocs" in bbd_results:
    print(f"\n  Predicted Optimal OCS : {bbd_results['predicted_ocs']:.2f} / 100")
    p = bbd_results["optimal_params"]
    print(f"  Optimal pH            : {p.get('pH', 'N/A'):.2f}")
    print(f"  Optimal Temperature   : {p.get('Temperature', 'N/A'):.1f} C")
    print(f"  Optimal Moisture      : {p.get('Moisture', 'N/A'):.2f} RH")

best_ph   = bbd_results["optimal_params"].get("pH", 7.0)          if "optimal_params" in bbd_results else 7.0
best_temp = bbd_results["optimal_params"].get("Temperature", 25.0) if "optimal_params" in bbd_results else 25.0
best_mois = bbd_results["optimal_params"].get("Moisture", 0.4)     if "optimal_params" in bbd_results else 0.4

# ============================================================
# STEP 2: COMBINED MIXTURE-PROCESS OPTIMIZATION
# ============================================================

print()
print(SEP)
print("STEP 2: Combined Mixture-Process Optimization")
print(SEP)

mix_opt = KeyBoxOptimizer(
    api_smiles="CC(=O)Oc1ccccc1C(=O)O",
    api_name="Aspirin"
)

# Mixture components (sum = 1.0 enforced)
mix_opt.add_mixture_component(
    "Lactose",
    "C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O)O"
)
mix_opt.add_mixture_component("PVP",    "CC(C(=O)N1CCCC1)C")
mix_opt.add_mixture_component("Starch", "C(C1C(C(C(C(O1)O)O)O)O)O")

# One process factor (fix Temperature from BBD result)
mix_opt.add_factor("Temperature", low=25.0, high=50.0)

print("[INFO] Running Combined Optimization...")
mix_results = mix_opt.run_combined_optimization(resolution=2.0, verbose=True)

print()
if "optimal_params" in mix_results:
    p = mix_results["optimal_params"]
    mix_keys = ["Lactose", "PVP", "Starch"]
    total = sum(p.get(k, 0) for k in mix_keys)
    print("  Optimal Mixture Composition:")
    for k in mix_keys:
        print(f"    {k:<10}: {p.get(k,0)*100:.1f}%")
    print(f"    {'Total':<10}: {total*100:.1f}%  (Simplex check)")
    print(f"\n  Predicted OCS : {mix_results['predicted_ocs']:.2f} / 100")

    best_lactose = p.get("Lactose", 0.5)
    best_pvp     = p.get("PVP",     0.3)
    best_starch  = p.get("Starch",  0.2)
else:
    best_lactose, best_pvp, best_starch = 0.5, 0.3, 0.2

# ============================================================
# STEP 3: VOXEL PHYSICS AT OPTIMAL FORMULATION
# ============================================================

print()
print(SEP)
print("STEP 3: Voxel Field Analysis at Optimal Point")
print(SEP)

from key import KeyBoxSystem, EnvironmentalConditions
from key.designer import create_enhanced_molecule_from_smiles

system = KeyBoxSystem(
    voxel_bounds=(20., 20., 20.),
    resolution=1.5,
    random_state=42
)

# Set optimal environment from BBD
system.env_conditions = EnvironmentalConditions(
    pH=best_ph,
    temperature=best_temp,
    moisture=best_mois
)

# Add API
aspirin = create_enhanced_molecule_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
system.add_molecule(aspirin, is_api=True)

# Add optimal mixture excipients at their optimal proportions
LACTOSE_SMILES = "C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O)O"
PVP_SMILES     = "CC(C(=O)N1CCCC1)C"
STARCH_SMILES  = "C(C1C(C(C(C(O1)O)O)O)O)O"

lactose = create_enhanced_molecule_from_smiles(LACTOSE_SMILES, "Lactose")
pvp     = create_enhanced_molecule_from_smiles(PVP_SMILES,     "PVP")
starch  = create_enhanced_molecule_from_smiles(STARCH_SMILES,  "Starch")

lactose.concentration = best_lactose
pvp.concentration     = best_pvp
starch.concentration  = best_starch

system.add_molecule(lactose, is_api=False)
system.add_molecule(pvp,     is_api=False)
system.add_molecule(starch,  is_api=False)

print("[INFO] Running full voxel field analysis with bootstrap MC (n=20)...")
results = system.analyze_enhanced_formulation(include_uncertainty=True, n_mc=20)

m = results["metrics"]
ci = results.get("confidence_intervals", {})

print()
print("  VOXEL FIELD METRICS")
print(f"  {'OCS':<20}: {m['OCS']:.2f} / 100")
print(f"  {'Classification':<20}: {results['classification']}")
print(f"  {'VSI (Steric)':<20}: {m['VSI']:.4f}")
print(f"  {'EC (Electrostatic)':<20}: {m['EC']:.4f}")
print(f"  {'HBP (H-Bond)':<20}: {m['HBP']:.4f}")
print(f"  {'HA (Hydrophobic)':<20}: {m['HA']:.4f}")
print(f"  {'RI (React Index)':<20}: {m['RI']:.4f}")
print(f"  {'Maillard VOI':<20}: {m.get('Maillard_VOI',0):.5f}")
print(f"  {'Hydrolysis VOI':<20}: {m.get('Hydrolysis_VOI',0):.5f}")
print(f"  {'AcidBase VOI':<20}: {m.get('AcidBase_VOI',0):.5f}")

if ci:
    print(f"\n  BOOTSTRAP MC UNCERTAINTY (n=20)")
    print(f"  Mean OCS : {ci.get('mean', 0):.2f}")
    print(f"  Std Dev  : {ci.get('std', 0):.2f}")
    lo, hi = ci.get('ci_95', (0, 0))
    print(f"  95% CI   : [{lo:.2f}, {hi:.2f}]")

print()
print("  DETECTED MECHANISMS:")
for mech in results.get("mechanisms", []):
    print(f"    - {mech}")

# ============================================================
# STEP 4: FIELD EXPORT
# ============================================================

print()
print(SEP)
print("STEP 4: Exporting Voxel Field Maps to viz_output/")
print(SEP)

os.makedirs("viz_output", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vg = system.voxel_grid
z_mid = vg.shape[2] // 2

fields_to_export = {
    "steric":       ("Steric Field (LJ Repulsion)",  "plasma"),
    "electrostatic":("Electrostatic Field (Debye-Huckel)", "RdBu_r"),
    "hydrophobic":  ("Hydrophobic Field",             "YlOrBr"),
    "h_donor":      ("H-Bond Donor Field",            "Blues"),
    "h_acceptor":   ("H-Bond Acceptor Field",         "Greens"),
    "reactive_sites":("Reactive Sites Field",         "hot"),
}

for fname, (title, cmap) in fields_to_export.items():
    data = getattr(vg, fname)[:, :, z_mid]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data.T, origin="lower", cmap=cmap)
    plt.colorbar(im, ax=ax, label="Field Intensity")
    ax.set_title(f"{title}\nz={z_mid} slice", fontsize=11)
    ax.set_xlabel("X voxels")
    ax.set_ylabel("Y voxels")
    fig.tight_layout()
    fig.savefig(f"viz_output/{fname}.png", dpi=150)
    plt.close()
    print(f"  [SAVED] viz_output/{fname}.png")

# VOI interaction density map
voi_map = (vg.amine_density_api[:,:,z_mid] * vg.carbonyl_density_exc[:,:,z_mid] +
           vg.acid_density_api[:,:,z_mid]  * vg.base_density_exc[:,:,z_mid])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(voi_map.T, origin="lower", cmap="hot")
plt.colorbar(im, ax=ax, label="VOI Density")
ax.set_title("VOI Hotspot Map (Amine*Carbonyl + Acid*Base)", fontsize=10)
ax.set_xlabel("X voxels"); ax.set_ylabel("Y voxels")
fig.tight_layout()
fig.savefig("viz_output/voi_hotspots.png", dpi=150)
plt.close()
print("  [SAVED] viz_output/voi_hotspots.png")

print()
print(SEP)
print("FULL LOOP COMPLETE")
print(SEP)
print(f"  Final OCS        : {m['OCS']:.2f}")
print(f"  Classification   : {results['classification']}")
print(f"  Fields exported  : {len(fields_to_export)+1} PNG files in viz_output/")
print(SEP)
