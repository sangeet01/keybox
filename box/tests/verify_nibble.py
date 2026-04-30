"""
verify_nibble.py
End-to-end verification of the Nibble docking engine.
Uses a real aspirin molecule created from SMILES via the KeyBox designer.
No mocks anywhere.
"""
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from key import KeyBoxSystem, NibbleEngine
from key.designer import create_enhanced_molecule_from_smiles
from key.models import DosageForm, DosageFormConfig

PDB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nibble', '1CRN.pdb'))
# Crambin (1CRN) pocket center
POCKET_CENTER = (17.0, 13.0, 11.0)
POCKET_RADIUS  = 10.0


def test_backend():
    """Report the active backend. C-Native requires building nibble.dll via build_nibble.py."""
    engine = NibbleEngine(dim_x=20, dim_y=20, dim_z=20, resolution=1.0)
    print(f"  Backend : {engine.mode}")
    if engine.mode == "C-Native":
        print("  C-Native DLL loaded.")
    else:
        print("  NumPy-Fallback active (run build_nibble.py to enable C-Native).")


def test_coarse_phase():
    """Phase I (theory.txt sec 5): Coarse 1.0A screen."""
    t0 = time.perf_counter()
    engine = NibbleEngine(dim_x=20, dim_y=20, dim_z=20, resolution=1.0)
    atoms = engine.load_pdb_pocket(PDB_PATH, POCKET_CENTER, POCKET_RADIUS, blur_radius=1.2)
    elapsed = (time.perf_counter() - t0) * 1e6
    assert atoms > 0, f"PDB load returned {atoms}"
    print(f"  Coarse pocket  : {atoms} atoms, loaded in {elapsed:.0f} us")
    return engine


def test_sniper_phase():
    """Phase II (theory.txt sec 5): Sniper 0.25A with B-factor Gaussian blur."""
    t0 = time.perf_counter()
    engine = NibbleEngine(dim_x=80, dim_y=80, dim_z=80, resolution=0.25)
    atoms = engine.load_pdb_pocket(PDB_PATH, POCKET_CENTER, POCKET_RADIUS, blur_radius=1.5)
    elapsed = (time.perf_counter() - t0) * 1e3
    assert atoms > 0
    print(f"  Sniper  pocket : {atoms} atoms, loaded in {elapsed:.1f} ms")
    return engine


def test_affinity_real_molecule():
    """
    End-to-end: create aspirin from SMILES, build a KeyBoxSystem,
    call evaluate_docking_affinity (two-phase cascade) against 1CRN.pdb.
    Zero mocks.
    """
    # Real molecule from SMILES via KeyBox designer
    aspirin = create_enhanced_molecule_from_smiles(
        "CC(=O)Oc1ccccc1C(=O)O",
        name="Aspirin",
        concentration=0.3
    )
    assert aspirin is not None and aspirin.coords is not None
    print(f"  Drug    : {aspirin.name}, {len(aspirin.coords)} atoms")

    # Real KeyBoxSystem (no mock)
    system = KeyBoxSystem(
        voxel_bounds=(20.0, 20.0, 20.0),
        resolution=0.5,
        dosage_config=DosageFormConfig(DosageForm.ORAL_SOLID),
        random_state=42
    )
    system.add_molecule(aspirin, is_api=True)

    # Two-phase cascade (Coarse -> Sniper) as per theory.txt
    t0 = time.perf_counter()
    score, report = system.evaluate_docking_affinity(PDB_PATH, POCKET_CENTER, POCKET_RADIUS)
    elapsed = (time.perf_counter() - t0) * 1e3

    print(f"  Affinity report : {report}")
    print(f"  Total time      : {elapsed:.1f} ms")
    assert isinstance(score, float)


def run():
    tests = [
        ("C-Native backend",       test_backend),
        ("Phase I Coarse (1.0A)",  test_coarse_phase),
        ("Phase II Sniper (0.25A)",test_sniper_phase),
        ("Full affinity (aspirin)",test_affinity_real_molecule),
    ]

    print("=" * 60)
    print("Nibble Engine Verification (theory.txt compliance)")
    print("=" * 60)
    passed = 0
    for label, fn in tests:
        print(f"\n[{label}]")
        try:
            fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL -> {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")
    print("=" * 60)
    return passed == len(tests)


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
