"""
KeyBox Test Suite: Full System Validation (v0.8.0)

Tests:
  1. Module imports
  2. Molecule creation from SMILES
  3. Basic compatibility analysis (OCS)
  4. Simplex normalization constraint
  5. BBD optimization (3-factor)
  6. Combined Mixture-Process optimization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

def run(label, fn):
    try:
        fn()
        print(f"  [PASS] {label}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label} -> {e}")
        return False


def test_imports():
    from key import (MoleculeType, DosageForm, Molecule, EnvironmentalConditions,
                     KeyBoxSystem, VoxelVisualizer, create_enhanced_molecule_from_smiles,
                     ENHANCED_EXCIPIENT_LIBRARY, KeyBoxOptimizer, FormulationOptimizer,
                     generate_box_behnken)


def test_molecule_creation():
    from key.designer import create_enhanced_molecule_from_smiles
    mol = create_enhanced_molecule_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
    assert mol.coords is not None
    assert len(mol.coords) > 0
    assert mol.concentration == 1.0


def test_basic_compatibility():
    from key import KeyBoxSystem, EnvironmentalConditions
    from key.designer import create_enhanced_molecule_from_smiles
    system = KeyBoxSystem(voxel_bounds=(20., 20., 20.), resolution=1.5, random_state=42)
    api = create_enhanced_molecule_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
    exc = create_enhanced_molecule_from_smiles("OCC1OC(O)C(O)C(O)C1O", "Lactose")
    system.add_molecule(api, is_api=True)
    system.add_molecule(exc, is_api=False)
    results = system.analyze_enhanced_formulation(include_uncertainty=False)
    ocs = results['metrics']['OCS']
    assert 0.0 <= ocs <= 100.0, f"OCS out of range: {ocs}"


def test_simplex_normalization():
    from key import KeyBoxOptimizer
    opt = KeyBoxOptimizer("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
    opt.add_mixture_component("Lactose", "OCC1OC(O)C(O)C(O)C1O")
    opt.add_mixture_component("PVP", "C1CN(C(=O)C=C1)C")
    opt.add_mixture_component("Starch", "C(C1C(C(C(C(O1)O)O)O)O)O")

    from key.optimizer import KeyBoxOptimizer as KO
    simplex = opt._generate_simplex_centroid(3)
    for row in simplex:
        assert abs(row.sum() - 1.0) < 1e-9, f"Simplex row does not sum to 1.0: {row}"

    # Test decode normalization
    coded = np.array([0.0, 0.0, 0.8, 0.5, 0.3])  # 2 process + 3 mixture
    opt.add_factor("Temperature", 25.0, 50.0)
    opt.add_factor("Moisture", 0.2, 0.8)
    params = opt._decode_design_point(coded)
    mix_sum = params["Lactose"] + params["PVP"] + params["Starch"]
    assert abs(mix_sum - 1.0) < 1e-9, f"Decoded mixture sum != 1.0: {mix_sum}"


def test_bbd_design_shape():
    from key.optimizer import generate_box_behnken
    design = generate_box_behnken(3, center_points=3)
    assert design.shape == (15, 3), f"Unexpected BBD shape: {design.shape}"
    # Center point check
    centers = design[12:]
    assert np.all(centers == 0.0)


def test_combined_optimization():
    from key import KeyBoxOptimizer
    opt = KeyBoxOptimizer("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
    opt.add_mixture_component("Lactose", "OCC1OC(O)C(O)C(O)C1O")
    opt.add_mixture_component("PVP", "C1CN(C(=O)C=C1)C")
    opt.add_factor("Temperature", 25.0, 50.0)
    results = opt.run_combined_optimization(resolution=2.0, verbose=False)
    assert "design_df" in results
    df = results["design_df"]
    # Each row mixture columns must sum to 1.0
    mix_sums = df[["Lactose", "PVP"]].sum(axis=1)
    assert (mix_sums - 1.0).abs().max() < 1e-6, "Simplex violated in combined design"
    if "predicted_ocs" in results:
        assert 0 <= results["predicted_ocs"] <= 100


def main():
    print("=" * 60)
    print("KeyBox Full Test Suite (v0.8.0)")
    print("=" * 60)

    tests = [
        ("Module imports",             test_imports),
        ("Molecule creation (SMILES)", test_molecule_creation),
        ("Basic compatibility (OCS)",  test_basic_compatibility),
        ("Simplex normalization",       test_simplex_normalization),
        ("BBD design matrix shape",    test_bbd_design_shape),
        ("Combined optimization",      test_combined_optimization),
    ]

    passed = sum(run(label, fn) for label, fn in tests)
    total = len(tests)

    print()
    print("=" * 60)
    print(f"Results: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
