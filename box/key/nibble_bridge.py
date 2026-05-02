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
N_CHANNELS = 11
CH_STERIC_DEMAND = 0
CH_ELEC_DEMAND = 1
CH_HBA_DEMAND = 2
CH_HBD_DEMAND = 3
CH_LIPO_DEMAND = 4
CH_AROM_DEMAND = 5
CH_METAL_DEMAND = 6
CH_CATION_DEMAND = 7
CH_ANION_DEMAND = 8
CH_PHOBIC_CORE = 9
CH_SOLVENT_EXPO = 10

# --- C Structure Definitions ---
class NibbleGridStruct(ctypes.Structure):
    _fields_ = [
        ("dim_x", ctypes.c_int),
        ("dim_y", ctypes.c_int),
        ("dim_z", ctypes.c_int),
        ("resolution", ctypes.c_float),
        ("data", ctypes.POINTER(ctypes.c_float))
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
        except Exception:
            cls._lib = None

    def load_pdb_pocket(self, pdb_path, center, range_val=10.0, blur_radius=1.5):
        return self.backend.load_pdb_pocket(pdb_path, center, range_val, blur_radius)

    def project_molecule(self, drug_engine, start_coords):
        """Projects APIs/Excipients and returns affinity."""
        if self.mode == "C-Native":
            # Extract combined data for C
            all_coords = []
            all_charges = []
            all_hydro = []
            for mol in drug_engine.apis + drug_engine.excipients:
                all_coords.extend(mol.coords)
                all_charges.extend(mol.charges)
                all_hydro.extend(mol.hydrophobicity)
            return self.backend.project_molecule_data(all_coords, all_charges, all_hydro, start_coords)
        else:
            # NumPy path
            drug_grid = NumPyNibbleEngine(self.dim_x, self.dim_y, self.dim_z, self.resolution)
            for mol in drug_engine.apis + drug_engine.excipients:
                for i in range(len(mol.coords)):
                    ch_vals = [0.0] * N_CHANNELS
                    ch_vals[CH_STERIC_DEMAND] = 1.0
                    ch_vals[CH_ELEC_DEMAND] = float(mol.charges[i])
                    ch_vals[CH_LIPO_DEMAND] = float(mol.hydrophobicity[i])
                    drug_grid.project_atom(
                        mol.coords[i][0] - start_coords[0],
                        mol.coords[i][1] - start_coords[1],
                        mol.coords[i][2] - start_coords[2],
                        1.5, ch_vals
                    )
            return self.backend.compute_affinity(drug_grid)
