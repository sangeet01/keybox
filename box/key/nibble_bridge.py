"""
Nibble Bridge (C-Native Docking Engine Interface)
-------------------------------------------------
Provides the Python ctypes bindings for the high-performance Nibble C-backend. 
Implements "Negative Space Matrix Multiplication" for O(1) thermal-averaged docking.
Automatically falls back to a vectorized NumPy implementation if the optimized 
C-native shared library cannot be loaded.
"""
import ctypes
import os
import numpy as np
from typing import Tuple, List, Optional, Dict
import time

# --- C Constants ---
N_CHANNELS = 16
CH_STERIC_DEMAND    = 0
CH_ELEC_DEMAND      = 1
CH_HBA_DEMAND       = 2
CH_HBD_DEMAND       = 3
CH_LIPO_DEMAND      = 4
CH_AROM_DEMAND      = 5
CH_METAL_DEMAND     = 6
CH_CATION_DEMAND    = 7
CH_ANION_DEMAND     = 8
CH_PHOBIC_CORE      = 9
CH_SOLVENT_EXPO     = 10
# Gap Closure channels (V4)
CH_FLEXIBILITY      = 11
CH_NUCLEOPHILICITY  = 12
CH_ELECTROPHILICITY = 13
CH_WATER_CONSERVED  = 14
CH_WATER_DYNAMIC    = 15

# --- C Structure Definitions ---
class NibbleGridStruct(ctypes.Structure):
    _fields_ = [
        ("dim_x", ctypes.c_int),
        ("dim_y", ctypes.c_int),
        ("dim_z", ctypes.c_int),
        ("resolution", ctypes.c_float),
        ("data", ctypes.POINTER(ctypes.c_float))
    ]

class NibbleMolStruct(ctypes.Structure):
    _fields_ = [
        ("cx", ctypes.c_float), ("cy", ctypes.c_float), ("cz", ctypes.c_float),
        ("vx", ctypes.c_float), ("vy", ctypes.c_float), ("vz", ctypes.c_float),
        ("qw", ctypes.c_float), ("qx", ctypes.c_float), ("qy", ctypes.c_float), ("qz", ctypes.c_float),
        ("wx", ctypes.c_float), ("wy", ctypes.c_float), ("wz", ctypes.c_float),
        ("n_atoms", ctypes.c_int),
        ("local_coords", ctypes.POINTER(ctypes.c_float)),
        ("ch_vals", ctypes.POINTER(ctypes.c_float))
    ]

# --- NumPy Fallback Implementation ---
class NumPyNibbleEngine:
    """
    A pure NumPy implementation of the Nibble engine logic.
    Provides O(1) matrix operations via NumPy's C-level vectorization.
    Used as a high-performance fallback when nibble.dll cannot be loaded.
    """
    def __init__(self, dim_x=30, dim_y=30, dim_z=30, resolution=0.8):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.resolution = resolution
        self.shape = (dim_x, dim_y, dim_z, N_CHANNELS)
        self.data = np.zeros(self.shape, dtype=np.float32)
        self.reset_steric()

    def reset_steric(self):
        self.data[..., CH_STERIC_DEMAND] = 1.0

    def project_atom(self, x, y, z, radius, channels):
        """Gaussian projection into the voxel grid."""
        # Calculate grid coords
        res = self.resolution
        ix, iy, iz = int(x/res), int(y/res), int(z/res)
        rv = int(radius/res) + 1
        
        # Slicing bounds
        x0, x1 = max(0, ix-rv), min(self.dim_x, ix+rv+1)
        y0, y1 = max(0, iy-rv), min(self.dim_y, iy+rv+1)
        z0, z1 = max(0, iz-rv), min(self.dim_z, iz+rv+1)
        
        # Grid coordinates for the sub-patch
        gx, gy, gz = np.meshgrid(
            np.arange(x0, x1) * res,
            np.arange(y0, y1) * res,
            np.arange(z0, z1) * res,
            indexing='ij'
        )
        
        # Distances
        d2 = (gx - x)**2 + (gy - y)**2 + (gz - z)**2
        mask = d2 < (radius * 2)**2
        
        if not np.any(mask):
            return

        # Gaussian weights
        weights = np.exp(-d2[mask] / (2.0 * radius**2))
        
        # Update channels
        for c_idx, val in enumerate(channels):
            if val != 0:
                self.data[x0:x1, y0:y1, z0:z1, c_idx][mask] += val * weights
        
        # Clip to prevent numerical explosion
        self.data[x0:x1, y0:y1, z0:z1, :][mask] = np.clip(
            self.data[x0:x1, y0:y1, z0:z1, :][mask], -200.0, 200.0
        )

    def load_pdb_pocket(self, pdb_path, center, range_val, blur_radius):
        """Parses PDB and projects demands."""
        self.reset_steric()
        atoms_found = 0
        cx, cy, cz = center
        sx, sy, sz = cx - range_val, cy - range_val, cz - range_val
        
        if not os.path.exists(pdb_path):
            return 0
            
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        ax = float(line[30:38])
                        ay = float(line[38:46])
                        az = float(line[46:54])
                        
                        dist2 = (ax-cx)**2 + (ay-cy)**2 + (az-cz)**2
                        if dist2 > range_val**2:
                            continue
                            
                        # B-factor adaptive blur
                        atom_blur = blur_radius
                        try:
                            b_val = float(line[60:66])
                            if b_val > 0:
                                b_sigma = np.sqrt(b_val / 78.957)
                                atom_blur *= (1.0 + min(2.0, b_sigma))
                        except: pass
                        
                        element = line[76:78].strip() or line[12:14].strip()
                        demands = np.zeros(N_CHANNELS)
                        demands[CH_STERIC_DEMAND] = -1.0
                        
                        if 'O' in element:
                            demands[CH_HBA_DEMAND] = 1.0
                            demands[CH_ELEC_DEMAND] = -0.6
                        elif 'N' in element:
                            demands[CH_HBD_DEMAND] = 1.0
                            demands[CH_ELEC_DEMAND] = 0.4
                        elif 'C' in element:
                            demands[CH_LIPO_DEMAND] = 0.7
                        
                        self.project_atom(ax-sx, ay-sy, az-sz, atom_blur, demands)
                        atoms_found += 1
                    except: continue
        return atoms_found

    def compute_affinity(self, other_engine):
        """Calculates binding affinity via dot product."""
        ps = self.data[..., CH_STERIC_DEMAND]
        ds = other_engine.data[..., CH_STERIC_DEMAND]
        
        # Hard clash penalty
        clash_mask = (ds > 0.1) & (ps < 0.05)
        score = -500.0 * np.sum(ds[clash_mask])
        
        # Channel overlap
        # We exclude steric from the sum since it's handled above or by direct mult
        score += np.sum(self.data * other_engine.data)
        return float(score)

# --- Native C implementation Bridge ---
class NativeNibbleEngine:
    def __init__(self, lib, dim_x=30, dim_y=30, dim_z=30, resolution=0.8):
        self._lib = lib
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.resolution = resolution
        self.grid_ptr = self._lib.nibble_create_grid(dim_x, dim_y, dim_z, resolution)

    def __del__(self):
        if hasattr(self, 'grid_ptr') and self.grid_ptr:
            self._lib.nibble_free_grid(self.grid_ptr)

    def load_pdb_pocket(self, pdb_path, center, range_val, blur_radius):
        return self._lib.nibble_load_pdb_pocket(
            self.grid_ptr, pdb_path.encode('utf-8'),
            float(center[0]), float(center[1]), float(center[2]),
            float(range_val), float(blur_radius)
        )

    def project_molecule_data(self, coords, charges, hydrophobicity, start_coords):
        # Create temporary drug grid
        drug_ptr = self._lib.nibble_create_grid(self.dim_x, self.dim_y, self.dim_z, self.resolution)
        for i in range(len(coords)):
            ch_vals = (ctypes.c_float * N_CHANNELS)(0)
            ch_vals[CH_STERIC_DEMAND] = 1.0
            ch_vals[CH_ELEC_DEMAND] = float(charges[i])
            ch_vals[CH_LIPO_DEMAND] = float(hydrophobicity[i])
            
            self._lib.nibble_project_atom(
                drug_ptr,
                float(coords[i][0] - start_coords[0]),
                float(coords[i][1] - start_coords[1]),
                float(coords[i][2] - start_coords[2]),
                1.5, ch_vals
            )
        affinity = self._lib.nibble_compute_affinity(self.grid_ptr, drug_ptr)
        self._lib.nibble_free_grid(drug_ptr)
        return float(affinity)

    # --- Gap Closure: Water Network ---

    def compute_water_network(self, delta_G_bulk=2.0, kT=0.593):
        """Compute and store CH_WATER_DYNAMIC occupancy field in the protein grid."""
        self._lib.nibble_compute_water_network(self.grid_ptr, ctypes.c_float(delta_G_bulk),
                                               ctypes.c_float(kT))

    def affinity_full(self, drug_engine, delta_G_bulk=2.0):
        """Frobenius affinity plus water-bridge bonus minus desolvation penalty."""
        return float(self._lib.nibble_compute_affinity_full(
            self.grid_ptr, drug_engine.grid_ptr, ctypes.c_float(delta_G_bulk)
        ))

    # --- Gap Closure: Fukui Covalent Channels ---

    def project_fukui(self, coords, partial_charges, blur_radius=1.5):
        """Project Fukui nucleophilicity/electrophilicity channels from partial charges."""
        coords_arr = np.array(coords, dtype=np.float32).flatten()
        charges_arr = np.array(partial_charges, dtype=np.float32)
        n = len(charges_arr)
        self._lib.nibble_project_fukui(
            self.grid_ptr, n,
            coords_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            charges_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(blur_radius)
        )

    def check_covalent(self, drug_engine, threshold=0.5):
        """Returns linear voxel index of covalent event site, or -1 if none."""
        return int(self._lib.nibble_check_covalent(
            self.grid_ptr, drug_engine.grid_ptr, ctypes.c_float(threshold)
        ))

    # --- Tier 3: Gradient ---

    def gradient_at(self, x, y, z):
        """Returns (gx, gy, gz) gradient of affinity field at position (x,y,z)."""
        gx = ctypes.c_float(0)
        gy = ctypes.c_float(0)
        gz = ctypes.c_float(0)
        self._lib.nibble_gradient(
            self.grid_ptr, ctypes.c_float(x), ctypes.c_float(y), ctypes.c_float(z),
            ctypes.byref(gx), ctypes.byref(gy), ctypes.byref(gz)
        )
        return float(gx.value), float(gy.value), float(gz.value)

    def run_trajectory(self, coords, charges, hydrophobicity, start_coords, 
                       n_steps=1000, dt=0.002, gamma=1.0, kT=0.593):
        """Runs the Langevin trajectory entirely in C for maximum performance."""
        n_atoms = len(coords)
        local_coords = np.zeros((n_atoms, 3), dtype=np.float32)
        ch_vals = np.zeros((n_atoms, N_CHANNELS), dtype=np.float32)
        
        for i in range(n_atoms):
            local_coords[i][0] = coords[i][0] - start_coords[0]
            local_coords[i][1] = coords[i][1] - start_coords[1]
            local_coords[i][2] = coords[i][2] - start_coords[2]
            
            ch_vals[i][CH_STERIC_DEMAND] = 1.0
            ch_vals[i][CH_ELEC_DEMAND] = charges[i]
            ch_vals[i][CH_LIPO_DEMAND] = hydrophobicity[i]

        mol_ptr = self._lib.nibble_mol_create(
            n_atoms, 
            local_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ch_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        scratch = self._lib.nibble_create_grid(self.dim_x, self.dim_y, self.dim_z, self.resolution)
        
        bcx = ctypes.c_float()
        bcy = ctypes.c_float()
        bcz = ctypes.c_float()
        
        best_score = self._lib.nibble_trajectory(
            mol_ptr, self.grid_ptr, scratch,
            ctypes.c_int(n_steps), ctypes.c_float(dt), ctypes.c_float(gamma), ctypes.c_float(kT),
            ctypes.byref(bcx), ctypes.byref(bcy), ctypes.byref(bcz)
        )
        
        self._lib.nibble_free_grid(scratch)
        self._lib.nibble_mol_free(mol_ptr)
        
        return float(best_score), (float(bcx.value), float(bcy.value), float(bcz.value))

    # --- Tier 4: Induced Fit ---

    def thermal_breathe(self, sigma_delta):
        """Apply thermal breathing (Mechanism 1): time-varying Gaussian blur."""
        self._lib.nibble_thermal_breathe(self.grid_ptr, ctypes.c_float(sigma_delta))

    def elastic_deform(self, ligand_engine, elasticity_scale=0.2):
        """Apply elastic deformation from ligand contact pressure (Mechanism 2)."""
        self._lib.nibble_elastic_deform(
            self.grid_ptr, ligand_engine.grid_ptr, ctypes.c_float(elasticity_scale)
        )

    def interpolate_from(self, grid_open_engine, grid_closed_engine, alpha):
        """Interpolate this grid between open and closed conformers (Mechanism 3)."""
        self._lib.nibble_interpolate_grids(
            self.grid_ptr, grid_open_engine.grid_ptr,
            grid_closed_engine.grid_ptr, ctypes.c_float(alpha)
        )

    def pocket_occupancy(self, ligand_engine, threshold=0.1):
        """Returns fractional pocket occupancy q from ligand grid overlap."""
        return float(self._lib.nibble_pocket_occupancy(
            self.grid_ptr, ligand_engine.grid_ptr, ctypes.c_float(threshold)
        ))

    def tier4_frame_update(self, grid_open_engine, grid_closed_engine,
                            ligand_engine, sigma_delta=0.05, elasticity_scale=0.2):
        """Full Tier 4 frame update: interpolate + breathe + elastic deform."""
        self._lib.nibble_tier4_frame_update(
            self.grid_ptr,
            grid_open_engine.grid_ptr,
            grid_closed_engine.grid_ptr,
            ligand_engine.grid_ptr,
            ctypes.c_float(sigma_delta),
            ctypes.c_float(elasticity_scale)
        )

    def score_displacement_curve(self, coords, charges, hydrophobicity, start_coords,
                                  output_file='displacement_curve.html',
                                  max_shift=5.0, step_size=0.5):
        """
        Spatial Convergence Curve: validates docking geometry by scanning the ligand
        along each axis and re-scoring at each displacement.

        A real binding event produces a sharp energy well (peak at 0 displacement).
        A non-binding molecule produces a flat or noisy curve.
        This is the MD-RMSD equivalent for voxel-based docking.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Error: 'plotly' is required. Install with 'pip install plotly'")
            return

        coords = np.array(coords)
        charges = np.array(charges)
        hydrophobicity = np.array(hydrophobicity)

        shifts = np.arange(-max_shift, max_shift + step_size * 0.5, step_size)
        axis_labels = ['X displacement (A)', 'Y displacement (A)', 'Z displacement (A)']
        colors = ['#7eb8a4', '#c4875a', '#8b9cc8']
        results = []

        for axis in range(3):
            axis_scores = []
            for shift in shifts:
                shifted = coords.copy()
                shifted[:, axis] += shift
                s = self.project_molecule_data(shifted, charges, hydrophobicity, start_coords)
                axis_scores.append(s)
            results.append(axis_scores)

        # Find the baseline (score at shift=0)
        zero_idx = int(len(shifts) // 2)
        baseline = results[0][zero_idx]

        fig = go.Figure()
        for axis in range(3):
            fig.add_trace(go.Scatter(
                x=shifts,
                y=results[axis],
                mode='lines+markers',
                name=axis_labels[axis],
                line=dict(color=colors[axis], width=2),
                marker=dict(size=5)
            ))

        # Mark optimal position
        fig.add_vline(
            x=0.0, line_width=1, line_dash='dash', line_color='rgba(255,255,255,0.4)',
            annotation_text='Optimal', annotation_position='top right',
            annotation_font_color='rgba(255,255,255,0.6)'
        )

        fig.update_layout(
            title=f'KeyBox Spatial Convergence Curve  |  Score at 0: {baseline:,.1f} (raw)',
            xaxis_title='Ligand Displacement from Optimal Position (A)',
            yaxis_title='Nibble Affinity Score (dimensionless Frobenius sum)',
            plot_bgcolor='rgb(11, 12, 14)',
            paper_bgcolor='rgb(11, 12, 14)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.25)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(255,255,255,0.1)')
        )
        fig.write_html(output_file)
        print(f"  Displacement curve saved to: {output_file}")
        return results

    def export_cube(self, filename: str, channel: int = CH_STERIC_DEMAND, atoms: Optional[List[Tuple[int, float, float, float]]] = None, start_coords: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Exports a specific voxel channel to a Gaussian Cube (.cube) file.
        This allows 3D isosurface visualization in tools like PyMOL, VMD, or UCSF ChimeraX.
        
        Args:
            filename: Output path (e.g., 'pocket_steric.cube')
            channel: Which channel to export (e.g., CH_STERIC_DEMAND)
            atoms: Optional list of (atomic_num, x, y, z) to include inside the visualization
            start_coords: The physical (x,y,z) origin of the grid
        """
        BOHR = 1.8897259886 # Conversion from Angstroms to Bohr units
        res = self.resolution * BOHR
        origin = [c * BOHR for c in start_coords]
        
        n_atoms = len(atoms) if atoms else 0
        
        # Extract grid data into a numpy array
        n = self.dim_x * self.dim_y * self.dim_z * N_CHANNELS
        arr = np.ctypeslib.as_array(self.grid_ptr.contents.data, shape=(n,))
        arr = arr.reshape((self.dim_x, self.dim_y, self.dim_z, N_CHANNELS))
            
        field = arr[:, :, :, channel]
        
        with open(filename, 'w') as f:
            f.write("KeyBox NibbleEngine Export\n")
            f.write(f"Channel: {channel}\n")
            f.write(f"{n_atoms:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
            f.write(f"{self.dim_x:5d} {res:12.6f} 0.000000 0.000000\n")
            f.write(f"{self.dim_y:5d} 0.000000 {res:12.6f} 0.000000\n")
            f.write(f"{self.dim_z:5d} 0.000000 0.000000 {res:12.6f}\n")
            
            if atoms:
                for atomic_num, x, y, z in atoms:
                    # Write: atomic_num, charge (0.0), x, y, z
                    f.write(f"{atomic_num:5d} 0.000000 {x*BOHR:12.6f} {y*BOHR:12.6f} {z*BOHR:12.6f}\n")
            
            # Format the output grid exactly as specified by the Gaussian cube standard
            # It loops x, then y, then z. Max 6 values per line.
            flat_field = field.flatten() # By default flattens in C order (x, y, z)
            for i in range(0, len(flat_field), 6):
                chunk = flat_field[i:i+6]
                line = "".join([f"{val:13.5E}" for val in chunk])
                f.write(line + "\n")

    def export_plotly_html(self, filename: str, channel: int = CH_STERIC_DEMAND, isovalue: float = 0.1, protein_atoms=None, ligand_atoms=None, score=None, start_coords=(0.0, 0.0, 0.0)):
        """
        Exports a specific voxel channel as an interactive 3D HTML visualization using Plotly,
        overlaying protein and ligand atoms if provided.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Error: 'plotly' is required for export_plotly_html. Install with 'pip install plotly'")
            return
            
        # Extract grid data
        n = self.dim_x * self.dim_y * self.dim_z * N_CHANNELS
        arr = np.ctypeslib.as_array(self.grid_ptr.contents.data, shape=(n,))
        arr = arr.reshape((self.dim_x, self.dim_y, self.dim_z, N_CHANNELS))
        field = arr[:, :, :, channel]
        
        # Create coordinates aligned with the physical pocket (start_coords)
        X, Y, Z = np.mgrid[0:self.dim_x, 0:self.dim_y, 0:self.dim_z]
        X = X * self.resolution + start_coords[0]
        Y = Y * self.resolution + start_coords[1]
        Z = Z * self.resolution + start_coords[2]
        
        data = []
        
        # 1. The Voxel Isosurface (Translucent Cloud)
        data.append(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=field.flatten(),
            isomin=isovalue,
            isomax=field.max(),
            surface_count=3,
            colorscale='Viridis',
            opacity=0.3, # Make it translucent so we can see atoms inside
            caps=dict(x_show=False, y_show=False)
        ))
        
        # 2. Protein Atoms
        if protein_atoms:
            px = [a[0] for a in protein_atoms]
            py = [a[1] for a in protein_atoms]
            pz = [a[2] for a in protein_atoms]
            data.append(go.Scatter3d(
                x=px, y=py, z=pz,
                mode='markers',
                marker=dict(size=4, color='lightgrey', opacity=0.8),
                name='Protein'
            ))
            
        # 3. Ligand Atoms
        if ligand_atoms is not None and len(ligand_atoms) > 0:
            lx = [a[0] for a in ligand_atoms]
            ly = [a[1] for a in ligand_atoms]
            lz = [a[2] for a in ligand_atoms]
            data.append(go.Scatter3d(
                x=lx, y=ly, z=lz,
                mode='markers',
                marker=dict(size=8, color='red', symbol='cross'),
                name='Ligand'
            ))
            
        fig = go.Figure(data=data)
        
        title_text = f"KeyBox Docking Viewer | Channel {channel}"
        if score is not None:
            title_text += f" | Affinity Score: {score:,.2f}"
            
        fig.update_layout(
            title=title_text,
            scene=dict(xaxis_title='X (A)', yaxis_title='Y (A)', zaxis_title='Z (A)'),
            plot_bgcolor='rgb(11, 12, 14)',
            paper_bgcolor='rgb(11, 12, 14)',
            font=dict(color='white')
        )
        fig.write_html(filename)

class LangevinMol:
    """Stateful C-native molecule for Langevin trajectories."""
    def __init__(self, engine, mol, start_coords):
        """
        Parameters
        ----------
        engine      : NativeNibbleEngine
        mol         : Molecule (KeyBox Molecule dataclass)
        start_coords: (sx, sy, sz) grid origin in absolute coordinates
        """
        self.engine = engine
        self._lib   = engine._lib
        coords      = mol.coords
        n_atoms     = len(coords)

        # Center of mass in absolute coordinates
        cx_abs = float(np.mean(coords[:, 0]))
        cy_abs = float(np.mean(coords[:, 1]))
        cz_abs = float(np.mean(coords[:, 2]))

        # Center of mass in grid-local coordinates
        cx_grid = cx_abs - start_coords[0]
        cy_grid = cy_abs - start_coords[1]
        cz_grid = cz_abs - start_coords[2]

        # Atom positions relative to center of mass (grid-local)
        local_coords = np.zeros((n_atoms, 3), dtype=np.float32)
        local_coords[:, 0] = coords[:, 0] - start_coords[0] - cx_grid
        local_coords[:, 1] = coords[:, 1] - start_coords[1] - cy_grid
        local_coords[:, 2] = coords[:, 2] - start_coords[2] - cz_grid

        # Full 16-channel array via shared helper
        ch_vals = NibbleEngine._mol_to_channel_array(mol)

        self.mol_ptr = self._lib.nibble_mol_create(
            n_atoms,
            local_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ch_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        self.mol_ptr.contents.cx = cx_grid
        self.mol_ptr.contents.cy = cy_grid
        self.mol_ptr.contents.cz = cz_grid

    def __del__(self):
        if hasattr(self, 'mol_ptr') and self.mol_ptr:
            self._lib.nibble_mol_free(self.mol_ptr)

    def run_trajectory(self, n_steps=10, dt=0.002, gamma=1.0, kT=0.593):
        scratch = self._lib.nibble_create_grid(self.engine.dim_x, self.engine.dim_y, self.engine.dim_z, self.engine.resolution)
        
        bcx = ctypes.c_float()
        bcy = ctypes.c_float()
        bcz = ctypes.c_float()
        
        best_score = self._lib.nibble_trajectory(
            self.mol_ptr, self.engine.grid_ptr, scratch,
            ctypes.c_int(n_steps), ctypes.c_float(dt), ctypes.c_float(gamma), ctypes.c_float(kT),
            ctypes.byref(bcx), ctypes.byref(bcy), ctypes.byref(bcz)
        )
        
        self._lib.nibble_free_grid(scratch)
        return float(best_score), (float(bcx.value), float(bcy.value), float(bcz.value))

    def project_to_grid(self, drug_engine):
        self._lib.nibble_mol_project(self.mol_ptr, drug_engine.grid_ptr)

# --- Main Interface ---
class NibbleEngine:
    """
    Orchestrator for the Nibble docking engine.
    Automatically selects the best available backend (Native C or NumPy).
    """
    _lib = None
    _tried_load = False

    def __init__(self, dim_x=30, dim_y=30, dim_z=30, resolution=0.8):
        self._init_lib()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.resolution = resolution
        
        if self._lib:
            self.backend = NativeNibbleEngine(self._lib, dim_x, dim_y, dim_z, resolution)
            self.mode = "C-Native"
        else:
            self.backend = NumPyNibbleEngine(dim_x, dim_y, dim_z, resolution)
            self.mode = "NumPy-Fallback"

    @classmethod
    def _init_lib(cls):
        if cls._tried_load: return
        cls._tried_load = True
        
        lib_name = 'nibble.dll' if os.name == 'nt' else ('libnibble.so' if os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'nibble', 'libnibble.so')) else 'nibble.so')
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nibble', lib_name))
        
        try:
            cls._lib = ctypes.CDLL(lib_path)
            # Setup argtypes
            cls._lib.nibble_create_grid.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
            cls._lib.nibble_create_grid.restype = ctypes.POINTER(NibbleGridStruct)
            cls._lib.nibble_free_grid.argtypes = [ctypes.POINTER(NibbleGridStruct)]
            
            cls._lib.nibble_reset_steric.argtypes = [ctypes.POINTER(NibbleGridStruct)]
            
            cls._lib.nibble_load_pdb_pocket.argtypes = [
                ctypes.POINTER(NibbleGridStruct), ctypes.c_char_p, 
                ctypes.c_float, ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_float
            ]
            cls._lib.nibble_load_pdb_pocket.restype = ctypes.c_int
            
            cls._lib.nibble_compute_affinity.argtypes = [
                ctypes.POINTER(NibbleGridStruct), ctypes.POINTER(NibbleGridStruct)
            ]
            cls._lib.nibble_compute_affinity.restype = ctypes.c_float
            
            cls._lib.nibble_update_local.argtypes = [
                ctypes.POINTER(NibbleGridStruct), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.POINTER(ctypes.c_float)
            ]
            
            cls._lib.nibble_project_atom.argtypes = [
                ctypes.POINTER(NibbleGridStruct), ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_float, ctypes.POINTER(ctypes.c_float)
            ]

            # --- Tier 3: Langevin Trajectory ---
            _G = ctypes.POINTER(NibbleGridStruct)
            _F = ctypes.c_float
            _PF = ctypes.POINTER(ctypes.c_float)

            cls._lib.nibble_gradient.argtypes = [_G, _F, _F, _F, _PF, _PF, _PF]
            cls._lib.nibble_gradient.restype = None

            cls._lib.nibble_compute_water_network.argtypes = [_G, _F, _F]
            cls._lib.nibble_compute_water_network.restype = None

            cls._lib.nibble_compute_affinity_full.argtypes = [_G, _G, _F]
            cls._lib.nibble_compute_affinity_full.restype = ctypes.c_float

            cls._lib.nibble_desolvation_penalty.argtypes = [_G, _G, _F, _F]
            cls._lib.nibble_desolvation_penalty.restype = ctypes.c_float

            cls._lib.nibble_project_fukui.argtypes = [
                _G, ctypes.c_int, _PF, _PF, _F
            ]
            cls._lib.nibble_project_fukui.restype = None

            cls._lib.nibble_check_covalent.argtypes = [_G, _G, _F]
            cls._lib.nibble_check_covalent.restype = ctypes.c_int

            _MOL = ctypes.POINTER(NibbleMolStruct)
            cls._lib.nibble_mol_create.argtypes = [ctypes.c_int, _PF, _PF]
            cls._lib.nibble_mol_create.restype = _MOL

            cls._lib.nibble_mol_free.argtypes = [_MOL]
            cls._lib.nibble_mol_free.restype = None

            cls._lib.nibble_trajectory.argtypes = [_MOL, _G, _G, ctypes.c_int, _F, _F, _F, _PF, _PF, _PF]
            cls._lib.nibble_trajectory.restype = ctypes.c_float
            
            cls._lib.nibble_mol_project.argtypes = [_MOL, _G]
            cls._lib.nibble_mol_project.restype = None

            # --- Tier 4: Induced Fit ---
            cls._lib.nibble_thermal_breathe.argtypes = [_G, _F]
            cls._lib.nibble_thermal_breathe.restype = None

            cls._lib.nibble_elastic_deform.argtypes = [_G, _G, _F]
            cls._lib.nibble_elastic_deform.restype = None

            cls._lib.nibble_interpolate_grids.argtypes = [_G, _G, _G, _F]
            cls._lib.nibble_interpolate_grids.restype = None

            cls._lib.nibble_pocket_occupancy.argtypes = [_G, _G, _F]
            cls._lib.nibble_pocket_occupancy.restype = ctypes.c_float

            cls._lib.nibble_tier4_frame_update.argtypes = [_G, _G, _G, _G, _F, _F]
            cls._lib.nibble_tier4_frame_update.restype = None
        except Exception:
            cls._lib = None

    def load_pdb_pocket(self, pdb_path, center, range_val=10.0, blur_radius=1.5):
        return self.backend.load_pdb_pocket(pdb_path, center, range_val, blur_radius)

    @staticmethod
    def _mol_to_channel_array(mol) -> np.ndarray:
        """
        Build a per-atom channel array [n_atoms, N_CHANNELS] from a Molecule.

        Previously only CH_STERIC, CH_ELEC, CH_LIPO were populated --
        leaving 13 channels dark on the drug side. This meant H-bond,
        aromaticity, metal coordination, and Fukui scoring were all
        one-sided (pocket only), making them useless for discrimination.

        Channel mapping:
          0  STERIC           always 1.0 (drug atom present)
          1  ELEC             Gasteiger partial charge
          2  HBA_DEMAND       drug H-bond acceptor
          3  HBD_DEMAND       drug H-bond donor
          4  LIPO             hydrophobicity index (Crippen)
          5  AROM             aromatic atom flag
          6  METAL            metal-binding group proxy
          7  CATION           protonated/quaternary N
          8  ANION            carboxylate/phosphate O
          9  PHOBIC_CORE      buried hydrophobic (high hydrophobicity + low charge)
         10  SOLVENT_EXPO     polarizability as solvent exposure proxy
         11  FLEXIBILITY      0.0 (flexibility is a pocket property, not drug)
         12  NUCLEOPHILICITY  Fukui f-: |q|^2 for negatively charged atoms
         13  ELECTROPHILICITY Fukui f+: |q|^2 for positively charged atoms
         14  WATER_CONSERVED  0.0 (drug doesn't carry crystallographic waters)
         15  WATER_DYNAMIC    0.0 (handled by desolvation penalty separately)
        """
        from rdkit import Chem

        n = len(mol.coords)
        ch = np.zeros((n, N_CHANNELS), dtype=np.float32)

        charges        = np.array(mol.charges,        dtype=np.float32)
        hydrophobicity = np.array(mol.hydrophobicity,  dtype=np.float32)
        h_donors       = np.array(mol.h_donors,        dtype=np.float32) if mol.h_donors       is not None else np.zeros(n, dtype=np.float32)
        h_acceptors    = np.array(mol.h_acceptors,     dtype=np.float32) if mol.h_acceptors     is not None else np.zeros(n, dtype=np.float32)
        polarizability = np.array(mol.polarizability,  dtype=np.float32) if mol.polarizability  is not None else np.zeros(n, dtype=np.float32)

        # CH_STERIC: always 1.0
        ch[:, CH_STERIC_DEMAND] = 1.0

        # CH_ELEC: partial charge
        ch[:, CH_ELEC_DEMAND] = charges

        # CH_HBA / CH_HBD
        ch[:, CH_HBA_DEMAND] = h_acceptors
        ch[:, CH_HBD_DEMAND] = h_donors

        # CH_LIPO
        ch[:, CH_LIPO_DEMAND] = hydrophobicity

        # CH_AROM: aromatic atoms from SMILES
        if mol.smiles:
            try:
                rdmol = Chem.MolFromSmiles(mol.smiles)
                if rdmol is not None:
                    rdmol_h = Chem.AddHs(rdmol)
                    n_rdkit = rdmol_h.GetNumAtoms()
                    arom_flags = np.array(
                        [1.0 if rdmol_h.GetAtomWithIdx(i).GetIsAromatic() else 0.0
                         for i in range(min(n_rdkit, n))],
                        dtype=np.float32
                    )
                    ch[:len(arom_flags), CH_AROM_DEMAND] = arom_flags
            except Exception:
                pass

        # CH_METAL: metal-binding group proxy
        # reactive_groups flags N and O; gate by negative charge (lone pair donors)
        if mol.reactive_groups is not None:
            rg = np.array(mol.reactive_groups, dtype=np.float32)
            ch[:, CH_METAL_DEMAND] = rg * np.clip(-charges, 0.0, 1.0)

        # CH_CATION: protonated N (basic group + positive charge)
        if mol.base_groups is not None:
            bg = np.array(mol.base_groups, dtype=np.float32)
            ch[:, CH_CATION_DEMAND] = bg * np.clip(charges, 0.0, 1.0)

        # CH_ANION: carboxylate/phosphate O (acid group + negative charge)
        if mol.acid_groups is not None:
            ag = np.array(mol.acid_groups, dtype=np.float32)
            ch[:, CH_ANION_DEMAND] = ag * np.clip(-charges, 0.0, 1.0)

        # CH_PHOBIC_CORE: high hydrophobicity + near-zero charge
        charge_penalty = np.exp(-4.0 * charges**2)
        ch[:, CH_PHOBIC_CORE] = np.clip(hydrophobicity, 0.0, None) * charge_penalty

        # CH_SOLVENT_EXPO: polarizability normalised
        pol_max = polarizability.max()
        pol_norm = polarizability / (pol_max + 1e-6) if pol_max > 0 else polarizability
        ch[:, CH_SOLVENT_EXPO] = pol_norm

        # CH_FLEXIBILITY: 0.0 -- flexibility is a pocket property
        ch[:, CH_FLEXIBILITY] = 0.0

        # CH_NUCLEOPHILICITY / CH_ELECTROPHILICITY: Fukui proxy
        # Matches nibble_project_fukui: threshold |q| < 0.1, sharpening q^2
        Q_THRESH = 0.10
        q_abs    = np.abs(charges)
        q_sharp  = np.where(q_abs >= Q_THRESH, q_abs ** 2, 0.0).astype(np.float32)
        ch[:, CH_NUCLEOPHILICITY]  = np.where(charges <  0.0, q_sharp, 0.0)
        ch[:, CH_ELECTROPHILICITY] = np.where(charges >= 0.0, q_sharp, 0.0)

        # CH_WATER_CONSERVED / CH_WATER_DYNAMIC: 0.0
        ch[:, CH_WATER_CONSERVED] = 0.0
        ch[:, CH_WATER_DYNAMIC]   = 0.0

        return ch

    def project_molecule(self, drug_engine, start_coords):
        """Projects APIs/Excipients into a drug grid and returns affinity score."""
        if self.mode == "C-Native":
            all_coords  = []
            all_ch_vals = []
            for mol in drug_engine.apis + drug_engine.excipients:
                ch = self._mol_to_channel_array(mol)
                all_coords.append(np.array(mol.coords, dtype=np.float32))
                all_ch_vals.append(ch)

            if not all_coords:
                return 0.0

            coords_np = np.vstack(all_coords).astype(np.float32)
            ch_np     = np.vstack(all_ch_vals).astype(np.float32)
            n_atoms   = len(coords_np)

            # Translate to grid-local coordinates
            coords_np[:, 0] -= start_coords[0]
            coords_np[:, 1] -= start_coords[1]
            coords_np[:, 2] -= start_coords[2]

            # Build drug grid projecting each atom with full channel vector
            drug_ptr = self._lib.nibble_create_grid(
                self.dim_x, self.dim_y, self.dim_z, self.resolution
            )
            ch_row = (ctypes.c_float * N_CHANNELS)()
            for i in range(n_atoms):
                for c in range(N_CHANNELS):
                    ch_row[c] = ch_np[i, c]
                self._lib.nibble_project_atom(
                    drug_ptr,
                    coords_np[i, 0], coords_np[i, 1], coords_np[i, 2],
                    1.5, ch_row
                )
            affinity = self._lib.nibble_compute_affinity(self.backend.grid_ptr, drug_ptr)
            self._lib.nibble_free_grid(drug_ptr)
            return float(affinity)

        else:
            # NumPy fallback -- full 16-channel projection
            drug_grid = NumPyNibbleEngine(self.dim_x, self.dim_y, self.dim_z, self.resolution)
            for mol in drug_engine.apis + drug_engine.excipients:
                ch = self._mol_to_channel_array(mol)
                for i in range(len(mol.coords)):
                    drug_grid.project_atom(
                        mol.coords[i][0] - start_coords[0],
                        mol.coords[i][1] - start_coords[1],
                        mol.coords[i][2] - start_coords[2],
                        1.5, ch[i].tolist()
                    )
            return self.backend.compute_affinity(drug_grid)

    def export_cube(self, filename: str, channel: int = 0, atoms=None, start_coords=(0.0, 0.0, 0.0)):
        if hasattr(self.backend, 'export_cube'):
            self.backend.export_cube(filename, channel, atoms, start_coords)
        else:
            print(" Warning: export_cube not implemented for NumPy-Fallback backend.\ ")

    def export_plotly_html(self, filename: str, channel: int = 0, isovalue: float = 0.1, protein_atoms=None, ligand_atoms=None, score=None, start_coords=(0.0, 0.0, 0.0)):
        if hasattr(self.backend, 'export_plotly_html'):
            self.backend.export_plotly_html(filename, channel, isovalue, protein_atoms, ligand_atoms, score, start_coords)
        else:
            print(" Warning: export_plotly_html not implemented for NumPy backend.\ ")
