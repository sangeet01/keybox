"""
Combined Mixture-Process Optimization Example (v0.8.0)

Demonstrates KeyBoxOptimizer for simultaneous optimization of:
  - Mixture Composition (Filler, Binder, Disintegrant) -- must sum to 100%
  - Process Factors (Temperature, Moisture) -- independent
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key import KeyBoxOptimizer

def main():
    print("=" * 60)
    print("KeyBox: Combined Mixture-Process Optimization")
    print("=" * 60)

    # Initialize optimizer with API
    opt = KeyBoxOptimizer(
        api_smiles="CC(=O)Oc1ccccc1C(=O)O",
        api_name="Aspirin"
    )

    # --- MIXTURE: proportions must sum to 1.0 ---
    print("\n[1] Defining Mixture Components (composition space)...")
    opt.add_mixture_component(
        "Lactose_Filler",
        "C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O[C@@H]2[C@H]"
        "(O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O)O"
    )
    opt.add_mixture_component("PVP_Binder", "C1CN(C(=O)C=C1)C")
    opt.add_mixture_component("Starch_Disintegrant", "C(C1C(C(C(C(O1)O)O)O)O)O")

    # --- PROCESS: independent environmental factors ---
    print("[2] Defining Process Factors (independent variables)...")
    opt.add_factor("Temperature", low=25.0, high=50.0)
    opt.add_factor("Moisture",    low=0.2,  high=0.8)

    # --- RUN COMBINED OPTIMIZATION ---
    print("\n[3] Running Combined Optimization...\n")
    results = opt.run_combined_optimization(resolution=2.0, verbose=True)

    # --- REPORT ---
    print("\n" + "=" * 60)
    print("OPTIMAL FORMULATION")
    print("=" * 60)

    if "optimal_params" in results:
        params = results["optimal_params"]
        print(f"\nPredicted OCS: {results['predicted_ocs']:.2f} / 100")
        print("\nProcess Conditions:")
        print(f"  Temperature : {params.get('Temperature', 'N/A'):.1f} C")
        print(f"  Moisture    : {params.get('Moisture', 'N/A'):.2f} RH")
        print("\nMixture Composition (sum must = 1.00):")
        mix_keys = ["Lactose_Filler", "PVP_Binder", "Starch_Disintegrant"]
        total = sum(params.get(k, 0) for k in mix_keys)
        for k in mix_keys:
            pct = params.get(k, 0) * 100
            print(f"  {k:<25}: {pct:.1f}%")
        print(f"  {'Total':<25}: {total*100:.1f}%")

        # Simplex constraint verification
        print(f"\n[CHECK] Simplex constraint satisfied: {abs(total - 1.0) < 1e-6}")
    else:
        print("\n[INFO] scikit-learn not found -- raw design table returned.")
        print(results["design_df"].to_string())

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
