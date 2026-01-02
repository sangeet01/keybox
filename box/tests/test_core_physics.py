"""
Core Module Import Test

Validates that all KeyBox open-source modules can be imported successfully.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test all module imports"""
    print("=" * 60)
    print("KeyBox Module Import Test")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import data structures
    print("\n[TEST] Importing data structures...")
    try:
        from key import MoleculeType, DosageForm, Molecule, EnvironmentalConditions
        print("  OK: Data structures imported")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Import core engine
    print("\n[TEST] Importing core engine...")
    try:
        from key import KeyBoxSystem
        print("  OK: KeyBoxSystem imported")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Import visualizer
    print("\n[TEST] Importing visualizer...")
    try:
        from key.visualizer import VoxelVisualizer
        print("  OK: VoxelVisualizer imported")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Import designer
    print("\n[TEST] Importing designer...")
    try:
        from key.designer import ENHANCED_EXCIPIENT_LIBRARY
        print("  OK: Designer module imported")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Create KeyBoxSystem instance
    print("\n[TEST] Creating KeyBoxSystem instance...")
    try:
        from key import KeyBoxSystem
        system = KeyBoxSystem(random_state=42)
        print("  OK: KeyBoxSystem instance created")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
