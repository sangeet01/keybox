"""
KeyBox Visualizer: Advanced 3D Voxel Field Visualization
Publication-grade rendering with vector fields and mechanism overlays
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .models import DosageForm
from .core_engine import EnhancedVoxelGrid, KeyBoxSystem


class VoxelVisualizer:
    """Advanced voxel field visualizer with vector fields and mechanism detection"""
    
    def __init__(self, system: KeyBoxSystem):
        self.system = system
        self.grid = system.voxel_grid
        
    def plot_field_slice(self, field_name: str, z_slice: Optional[int] = None, 
                        save_path: Optional[str] = None):
        """2D slice of any field channel"""
        field = getattr(self.grid, field_name)
        z = z_slice or self.grid.shape[2] // 2
        
        plt.figure(figsize=(8, 6))
        plt.imshow(field[:, :, z].T, origin='lower', cmap='viridis')
        plt.colorbar(label=field_name)
        plt.title(f'{field_name} (z={z})')
        plt.xlabel('X'); plt.ylabel('Y')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_3d_isosurface(self, field_name: str, threshold: float = 0.5,
                          save_path: Optional[str] = None):
        """3D isosurface rendering"""
        field = getattr(self.grid, field_name)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points above threshold
        mask = field > threshold
        coords = np.where(mask)
        
        if len(coords[0]) > 0:
            ax.scatter(coords[0], coords[1], coords[2], 
                      c=field[mask], cmap='plasma', alpha=0.6, s=20)
        
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'{field_name} > {threshold}')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_interaction_heatmap(self, save_path: Optional[str] = None):
        """Multi-channel interaction heatmap"""
        channels = ['electrostatic', 'hydrophobic', 'steric', 'h_donor', 
                   'h_acceptor', 'reactive_sites']
        z = self.grid.shape[2] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, ch in enumerate(channels):
            field = getattr(self.grid, ch)
            im = axes[i].imshow(field[:, :, z].T, origin='lower', cmap='RdYlBu_r')
            axes[i].set_title(ch)
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def animate_temporal_evolution(self, field_name: str, n_frames: int = 50,
                                   save_path: Optional[str] = None):
        """Animate field evolution over time"""
        # Compute temporal snapshots
        snapshots = []
        z = self.grid.shape[2] // 2
        
        for t in range(n_frames):
            # Simulate diffusion
            field = getattr(self.grid, field_name).copy()
            field = self._apply_diffusion(field, steps=t)
            snapshots.append(field[:, :, z])
        
        # Create animation
        fig, ax = plt.subplots(figsize=(8, 6))
        
        vmin = min(s.min() for s in snapshots)
        vmax = max(s.max() for s in snapshots)
        
        im = ax.imshow(snapshots[0].T, origin='lower', cmap='viridis',
                      vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=field_name)
        title = ax.set_title(f'{field_name} - Frame 0')
        
        def update(frame):
            im.set_array(snapshots[frame].T)
            title.set_text(f'{field_name} - Frame {frame}')
            return [im, title]
        
        anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
        
        if save_path:
            writer = PillowWriter(fps=10)
            anim.save(save_path, writer=writer)
            plt.close()
        else:
            plt.show()
    
    def plot_voi_spatial_distribution(self, save_path: Optional[str] = None):
        """Visualize VOI hotspots"""
        # Compute interaction density
        voi_map = (self.grid.amine_density_api * self.grid.carbonyl_density_exc + 
                   self.grid.amine_density_exc * self.grid.carbonyl_density_api +
                   self.grid.acid_density_api * self.grid.base_density_exc +
                   self.grid.acid_density_exc * self.grid.base_density_api)
        
        z = self.grid.shape[2] // 2
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 2D heatmap
        im1 = ax1.imshow(voi_map[:, :, z].T, origin='lower', cmap='hot')
        ax1.set_title('VOI Hotspots (2D Slice)')
        plt.colorbar(im1, ax=ax1, label='Interaction Density')
        
        # 3D scatter
        ax2 = fig.add_subplot(122, projection='3d')
        if np.any(voi_map > 0):
            mask = voi_map > np.percentile(voi_map[voi_map > 0], 75)
            coords = np.where(mask)
            
            if len(coords[0]) > 0:
                ax2.scatter(coords[0], coords[1], coords[2], 
                           c=voi_map[mask], cmap='hot', alpha=0.7, s=30)
        ax2.set_title('VOI Hotspots (3D)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _apply_diffusion(self, field: np.ndarray, steps: int = 1) -> np.ndarray:
        """Simple diffusion simulation"""
        from scipy.ndimage import gaussian_filter
        sigma = 0.5 + 0.1 * steps
        return gaussian_filter(field, sigma=sigma)
    
    def plot_vector_field(self, field_name: str, z_slice: Optional[int] = None,
                         stride: int = 2, save_path: Optional[str] = None):
        """Vector field visualization showing gradients (FEA-style)"""
        field = getattr(self.grid, field_name)
        z = z_slice or self.grid.shape[2] // 2
        
        # Compute gradients
        gy, gx = np.gradient(field[:, :, z])
        
        # Create meshgrid for arrows
        x = np.arange(0, field.shape[0], stride)
        y = np.arange(0, field.shape[1], stride)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Sample gradients
        U = gx[::stride, ::stride]
        V = gy[::stride, ::stride]
        magnitude = np.sqrt(U**2 + V**2)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Background field
        im = ax.imshow(field[:, :, z].T, origin='lower', cmap='gray', alpha=0.3)
        
        # Vector field
        quiver = ax.quiver(X, Y, U, V, magnitude, cmap='plasma', 
                          scale=None, width=0.003, alpha=0.8)
        
        plt.colorbar(quiver, ax=ax, label='Gradient Magnitude')
        ax.set_title(f'{field_name} Vector Field (z={z})')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_mechanism_overlay(self, save_path: Optional[str] = None):
        """Mechanism-specific spatial overlays with confidence contours"""
        z = self.grid.shape[2] // 2
        
        # Compute mechanism maps
        maillard = (self.grid.amine_density_api[:, :, z] * 
                   self.grid.carbonyl_density_exc[:, :, z] +
                   self.grid.amine_density_exc[:, :, z] * 
                   self.grid.carbonyl_density_api[:, :, z])
        
        hydrolysis = self.grid.reactive_sites[:, :, z] * self.system.env_conditions.moisture
        
        acid_base = (self.grid.acid_density_api[:, :, z] * 
                    self.grid.base_density_exc[:, :, z] +
                    self.grid.acid_density_exc[:, :, z] * 
                    self.grid.base_density_api[:, :, z])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Maillard
        im1 = axes[0, 0].imshow(maillard.T, origin='lower', cmap='Reds', alpha=0.8)
        if np.max(maillard) > 0:
            levels = np.percentile(maillard[maillard > 0], [50, 75, 90])
            axes[0, 0].contour(maillard.T, levels=levels, colors='darkred', 
                             linewidths=1.5, alpha=0.6)
        axes[0, 0].set_title('Maillard Reaction Risk')
        plt.colorbar(im1, ax=axes[0, 0], label='Amine×Carbonyl')
        
        # Hydrolysis
        im2 = axes[0, 1].imshow(hydrolysis.T, origin='lower', cmap='Blues', alpha=0.8)
        if np.max(hydrolysis) > 0:
            levels = np.percentile(hydrolysis[hydrolysis > 0], [50, 75, 90])
            axes[0, 1].contour(hydrolysis.T, levels=levels, colors='darkblue',
                             linewidths=1.5, alpha=0.6)
        axes[0, 1].set_title('Hydrolysis Risk')
        plt.colorbar(im2, ax=axes[0, 1], label='Reactive×Moisture')
        
        # Acid-Base
        im3 = axes[1, 0].imshow(acid_base.T, origin='lower', cmap='Purples', alpha=0.8)
        if np.max(acid_base) > 0:
            levels = np.percentile(acid_base[acid_base > 0], [50, 75, 90])
            axes[1, 0].contour(acid_base.T, levels=levels, colors='indigo',
                             linewidths=1.5, alpha=0.6)
        axes[1, 0].set_title('Acid-Base Interaction')
        plt.colorbar(im3, ax=axes[1, 0], label='Acid×Base')
        
        # Combined risk map
        combined = maillard + hydrolysis + acid_base
        im4 = axes[1, 1].imshow(combined.T, origin='lower', cmap='hot', alpha=0.8)
        if np.max(combined) > 0:
            levels = np.percentile(combined[combined > 0], [50, 75, 90])
            CS = axes[1, 1].contour(combined.T, levels=levels, colors='white',
                                   linewidths=2, alpha=0.8)
            axes[1, 1].clabel(CS, inline=True, fontsize=8, fmt='%1.0f%%')
        axes[1, 1].set_title('Combined Risk Map')
        plt.colorbar(im4, ax=axes[1, 1], label='Total Risk')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_cross_section_multiplane(self, field_name: str, 
                                     save_path: Optional[str] = None):
        """Orthogonal XY, XZ, YZ slices (medical imaging style)"""
        field = getattr(self.grid, field_name)
        
        # Slice positions
        x_mid = self.grid.shape[0] // 2
        y_mid = self.grid.shape[1] // 2
        z_mid = self.grid.shape[2] // 2
        
        fig = plt.figure(figsize=(15, 5))
        
        # XY plane (z=mid)
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(field[:, :, z_mid].T, origin='lower', cmap='viridis')
        ax1.set_title(f'XY Plane (z={z_mid})')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1)
        
        # XZ plane (y=mid)
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(field[:, y_mid, :].T, origin='lower', cmap='viridis')
        ax2.set_title(f'XZ Plane (y={y_mid})')
        ax2.set_xlabel('X'); ax2.set_ylabel('Z')
        plt.colorbar(im2, ax=ax2)
        
        # YZ plane (x=mid)
        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(field[x_mid, :, :].T, origin='lower', cmap='viridis')
        ax3.set_title(f'YZ Plane (x={x_mid})')
        ax3.set_xlabel('Y'); ax3.set_ylabel('Z')
        plt.colorbar(im3, ax=ax3)
        
        fig.suptitle(f'{field_name} - Orthogonal Cross-Sections', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_interactive_3d_plotly(self, field_name: str, threshold: float = 0.5,
                                  save_path: Optional[str] = None):
        """Interactive 3D visualization with Plotly (rotatable, zoomable)"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[WARNING] plotly not installed. Run: pip install plotly")
            return
        
        field = getattr(self.grid, field_name)
        mask = field > threshold
        coords = np.where(mask)
        
        if len(coords[0]) == 0:
            print(f"[WARNING] No voxels above threshold {threshold}")
            return
        
        # Create 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[0],
            y=coords[1],
            z=coords[2],
            mode='markers',
            marker=dict(
                size=3,
                color=field[mask],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title=field_name),
                opacity=0.7
            ),
            text=[f'{field_name}: {v:.3f}' for v in field[mask]],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f'{field_name} > {threshold} (Interactive 3D)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"[INTERACTIVE] Saved to {save_path} (open in browser)")
        else:
            fig.show()
    
    def animate_phase_separation(self, chi: float = 0.6, n_frames: int = 100,
                                save_path: Optional[str] = None):
        """Animate Cahn-Hilliard phase separation (spinodal decomposition)"""
        from .core_engine import MicrostructurePhysics
        
        # Initialize random composition field
        nx, ny = 50, 50
        phi = 0.5 + 0.05 * np.random.randn(nx, ny)
        
        # Run Cahn-Hilliard
        phi_history = MicrostructurePhysics.cahn_hilliard_phase_field(
            chi, phi, dx=1.0, dt=0.01, steps=n_frames
        )
        
        # Create animation
        fig, ax = plt.subplots(figsize=(8, 7))
        
        im = ax.imshow(phi_history[0].T, origin='lower', cmap='RdBu_r',
                      vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Volume Fraction')
        title = ax.set_title(f'Phase Separation (chi={chi:.2f}) - Frame 0/{n_frames}')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        
        def update(frame):
            im.set_array(phi_history[frame].T)
            title.set_text(f'Phase Separation (chi={chi:.2f}) - Frame {frame}/{n_frames}')
            return [im, title]
        
        anim = FuncAnimation(fig, update, frames=len(phi_history), 
                           interval=50, blit=True)
        
        if save_path:
            writer = PillowWriter(fps=20)
            anim.save(save_path, writer=writer)
            plt.close()
            print(f"[ANIMATION] Phase separation saved to {save_path}")
        else:
            plt.show()
    
    def plot_comparative_dashboard(self, system2: KeyBoxSystem, 
                                  labels: Tuple[str, str] = ('Formulation A', 'Formulation B'),
                                  save_path: Optional[str] = None):
        """Side-by-side comparison of two formulations"""
        # Compute metrics for both systems
        metrics1 = self.system.compute_enhanced_interaction_metrics()
        metrics2 = system2.compute_enhanced_interaction_metrics()
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Row 1: Field comparisons
        z = self.grid.shape[2] // 2
        
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.grid.steric[:, :, z].T, origin='lower', cmap='viridis')
        ax1.set_title(f'{labels[0]} - Steric')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(system2.voxel_grid.steric[:, :, z].T, origin='lower', cmap='viridis')
        ax2.set_title(f'{labels[1]} - Steric')
        plt.colorbar(im2, ax=ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff = self.grid.steric[:, :, z] - system2.voxel_grid.steric[:, :, z]
        im3 = ax3.imshow(diff.T, origin='lower', cmap='RdBu_r')
        ax3.set_title('Difference (A - B)')
        plt.colorbar(im3, ax=ax3)
        
        # Row 2: Mechanism overlays
        voi1 = (self.grid.amine_density_api[:, :, z] * self.grid.carbonyl_density_exc[:, :, z] +
                self.grid.acid_density_api[:, :, z] * self.grid.base_density_exc[:, :, z])
        voi2 = (system2.voxel_grid.amine_density_api[:, :, z] * system2.voxel_grid.carbonyl_density_exc[:, :, z] +
                system2.voxel_grid.acid_density_api[:, :, z] * system2.voxel_grid.base_density_exc[:, :, z])
        
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(voi1.T, origin='lower', cmap='hot')
        ax4.set_title(f'{labels[0]} - VOI')
        plt.colorbar(im4, ax=ax4)
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(voi2.T, origin='lower', cmap='hot')
        ax5.set_title(f'{labels[1]} - VOI')
        plt.colorbar(im5, ax=ax5)
        
        ax6 = fig.add_subplot(gs[1, 2])
        voi_diff = voi1 - voi2
        im6 = ax6.imshow(voi_diff.T, origin='lower', cmap='RdBu_r')
        ax6.set_title('VOI Difference')
        plt.colorbar(im6, ax=ax6)
        
        # Row 3: Metrics radar chart + bar comparison
        ax7 = fig.add_subplot(gs[2, :2], projection='polar')
        
        categories = ['OCS', 'SCI', 'EC', 'HA', 'HBP', 'RI']
        values1 = [metrics1.get(k, 0) for k in categories]
        values2 = [metrics2.get(k, 0) for k in categories]
        
        # Normalize to 0-1
        values1_norm = [v/100 if k == 'OCS' else v for k, v in zip(categories, values1)]
        values2_norm = [v/100 if k == 'OCS' else v for k, v in zip(categories, values2)]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values1_norm += values1_norm[:1]
        values2_norm += values2_norm[:1]
        angles += angles[:1]
        
        ax7.plot(angles, values1_norm, 'o-', linewidth=2, label=labels[0], color='blue')
        ax7.fill(angles, values1_norm, alpha=0.25, color='blue')
        ax7.plot(angles, values2_norm, 'o-', linewidth=2, label=labels[1], color='red')
        ax7.fill(angles, values2_norm, alpha=0.25, color='red')
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(categories)
        ax7.set_ylim(0, 1)
        ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax7.set_title('Metrics Comparison', y=1.08)
        
        # Bar chart for OCS
        ax8 = fig.add_subplot(gs[2, 2])
        x = [0, 1]
        ocs_vals = [metrics1['OCS'], metrics2['OCS']]
        colors = ['green' if v >= 60 else 'orange' if v >= 40 else 'red' for v in ocs_vals]
        ax8.bar(x, ocs_vals, color=colors, alpha=0.7)
        ax8.set_xticks(x)
        ax8.set_xticklabels(labels, rotation=15)
        ax8.set_ylabel('OCS Score')
        ax8.set_title('Overall Compatibility')
        ax8.axhline(60, color='green', linestyle='--', alpha=0.5, label='Compatible')
        ax8.axhline(40, color='orange', linestyle='--', alpha=0.5, label='Partial')
        ax8.legend(fontsize=8)
        
        fig.suptitle('Comparative Formulation Dashboard', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def export_all_fields(self, output_dir: str = 'viz_output', include_phase2: bool = False):
        """Export all field channels as images"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fields = ['electrostatic', 'hydrophobic', 'steric', 'h_donor', 
                 'h_acceptor', 'reactive_sites', 'acid_density', 'base_density']
        
        for field in fields:
            if hasattr(self.grid, field):
                self.plot_field_slice(field, save_path=f'{output_dir}/{field}.png')
        
        self.plot_interaction_heatmap(save_path=f'{output_dir}/heatmap.png')
        self.plot_voi_spatial_distribution(save_path=f'{output_dir}/voi_hotspots.png')
        
        # Advanced visualizations
        self.plot_vector_field('electrostatic', save_path=f'{output_dir}/vector_field.png')
        self.plot_mechanism_overlay(save_path=f'{output_dir}/mechanism_overlay.png')
        self.plot_cross_section_multiplane('steric', save_path=f'{output_dir}/cross_sections.png')
        
        count = len(fields) + 5
        
        # Phase 2 features (optional)
        if include_phase2:
            self.plot_interactive_3d_plotly('reactive_sites', threshold=0.3,
                                           save_path=f'{output_dir}/interactive_3d.html')
            self.animate_phase_separation(chi=0.6, n_frames=100,
                                        save_path=f'{output_dir}/phase_separation.gif')
            count += 2
            print(f"[VISUALIZER] Exported {count} visualizations (including Phase 2) to {output_dir}/")
        else:
            print(f"[VISUALIZER] Exported {count} visualizations to {output_dir}/")


def quick_visualize(system: KeyBoxSystem, output_dir: str = 'viz_output'):
    """One-line visualization export"""
    viz = VoxelVisualizer(system)
    viz.export_all_fields(output_dir)
    return viz
