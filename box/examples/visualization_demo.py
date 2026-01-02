"""
Visualization Demo Example

Demonstrates KeyBox's FEA-style visualization capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from key import KeyBoxSystem, EnvironmentalConditions
from key.visualizer import VoxelVisualizer
from key.designer import create_enhanced_molecule_from_smiles

def main():
    print("=" * 60)
    print("KeyBox: Visualization Demo")
    print("=" * 60)
    
    # Initialize system
    system = KeyBoxSystem(
        voxel_bounds=(20.0, 20.0, 20.0),
        resolution=1.0,
        random_state=42
    )
    
    # Add molecules
    aspirin = create_enhanced_molecule_from_smiles(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        name="Aspirin"
    )
    system.add_molecule(aspirin, is_api=True)
    
    ethylene_glycol = create_enhanced_molecule_from_smiles(
        smiles="C(CO)O",
        name="Ethylene_Glycol"
    )
    system.add_molecule(ethylene_glycol, is_api=False)
    
    # Set conditions
    system.env_conditions = EnvironmentalConditions(
        pH=7.0,
        moisture=0.6,
        temperature=40.0,
        storage_time=6.0
    )
    
    print("\nGenerating visualizations...")
    
    # Create output directory
    output_dir = "viz_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    viz = VoxelVisualizer(system)
    
    # 1. Field slice visualization
    print("  - Electrostatic field slice...")
    viz.plot_field_slice(
        'electrostatic',
        z_slice=10,
        save_path=f"{output_dir}/electrostatic_slice.png"
    )
    
    # 2. Interaction heatmap
    print("  - Interaction heatmap...")
    viz.plot_interaction_heatmap(
        save_path=f"{output_dir}/interaction_heatmap.png"
    )
    
    # 3. VOI spatial distribution
    print("  - VOI hotspots...")
    viz.plot_voi_spatial_distribution(
        save_path=f"{output_dir}/voi_hotspots.png"
    )
    
    print(f"\nOK: Visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
