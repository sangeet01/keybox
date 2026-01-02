"""
Basic API-Excipient Compatibility Analysis Example

Demonstrates core KeyBox functionality for predicting formulation incompatibilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key import KeyBoxSystem, EnvironmentalConditions
from key.designer import create_enhanced_molecule_from_smiles

def main():
    print("=" * 60)
    print("KeyBox: Basic Compatibility Analysis")
    print("=" * 60)
    
    # Initialize system
    system = KeyBoxSystem(
        voxel_bounds=(20.0, 20.0, 20.0),
        resolution=1.0,
        random_state=42
    )
    
    # Add API (Aspirin)
    aspirin = create_enhanced_molecule_from_smiles(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        name="Aspirin"
    )
    system.add_molecule(aspirin, is_api=True)
    
    # Add Excipient (Lactose)
    lactose = create_enhanced_molecule_from_smiles(
        smiles="C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
        name="Lactose"
    )
    system.add_molecule(lactose, is_api=False)
    
    # Set environmental conditions (accelerated stability)
    system.env_conditions = EnvironmentalConditions(
        pH=7.0,
        moisture=0.6,  # 60% RH
        temperature=40.0,  # 40°C
        storage_time=6.0  # 6 months
    )
    
    print("\nFormulation:")
    print(f"  API: Aspirin")
    print(f"  Excipient: Lactose")
    print(f"\nConditions:")
    print(f"  pH: {system.env_conditions.pH}")
    print(f"  Moisture: {system.env_conditions.moisture * 100:.0f}% RH")
    print(f"  Temperature: {system.env_conditions.temperature}°C")
    print(f"  Storage Time: {system.env_conditions.storage_time} months")
    
    # Analyze formulation
    print("\nAnalyzing...")
    results = system.analyze_enhanced_formulation()
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nCompatibility: {results['classification']}")
    print(f"OCS Score: {results['metrics']['OCS']:.3f}")
    
    print(f"\nKey Metrics:")
    print(f"  VSI (Voxel Steric Index): {results['metrics']['VSI']:.3f}")
    print(f"  EC (Electrostatic Clash): {results['metrics']['EC']:.3f}")
    print(f"  HBP (H-Bond Potential): {results['metrics']['HBP']:.3f}")
    
    if results['mechanisms']:
        print(f"\nDetected Mechanisms:")
        for mech in results['mechanisms']:
            print(f"  - {mech}")
    else:
        print(f"\nNo incompatibility mechanisms detected")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
