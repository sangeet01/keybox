import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional
from .models import DosageForm, DosageFormConfig

class EnhancedVoxelGrid:
    """Advanced 3D voxel grid with multi-phase and temporal support"""
    
    def __init__(self, bounds: Tuple[float, float, float], resolution: float = 1.0, 
                 dosage_config: Optional[DosageFormConfig] = None):
        self.bounds = bounds
        self.resolution = resolution
        self.dosage_config = dosage_config or DosageFormConfig(DosageForm.ORAL_SOLID)
        self.shape = tuple(int(b / resolution) for b in bounds)
        
        # Enhanced field channels
        self.electrostatic = np.zeros(self.shape)
        self.hydrophobic = np.zeros(self.shape)
        self.steric = np.zeros(self.shape)
        self.h_donor = np.zeros(self.shape)
        self.h_acceptor = np.zeros(self.shape)
        self.polarizability = np.zeros(self.shape)
        self.solvent_accessible = np.zeros(self.shape)
        self.reactive_sites = np.zeros(self.shape)
        self.lipid_density = np.zeros(self.shape)
        self.viscosity = np.zeros(self.shape)
        self.ph_field = np.zeros(self.shape) 
        self.ionic_field = np.zeros(self.shape) 
        self.metal_ion = np.zeros(self.shape)
        self.peroxide_density = np.zeros(self.shape)
        self.acid_density = np.zeros(self.shape) 
        self.base_density = np.zeros(self.shape) 
        self.acid_density_api = np.zeros(self.shape)
        self.base_density_api = np.zeros(self.shape)
        self.acid_density_exc = np.zeros(self.shape)
        self.base_density_exc = np.zeros(self.shape)
        
        self.amine_density_api = np.zeros(self.shape)
        self.carbonyl_density_api = np.zeros(self.shape)
        self.amine_density_exc = np.zeros(self.shape)
        self.carbonyl_density_exc = np.zeros(self.shape)
        
        self.phase_map = np.zeros(self.shape, dtype=int) 
        self.diffusion_coeffs = np.ones(self.shape) 
        
        self.temporal_occupancy = []
        self.residence_time = np.zeros(self.shape)

        x = np.linspace(0, bounds[0], self.shape[0])
        y = np.linspace(0, bounds[1], self.shape[1])
        z = np.linspace(0, bounds[2], self.shape[2])
        gx, gy, gz = np.meshgrid(x, y, z, indexing='ij')
        self._voxel_coords_flat = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=1
        ).astype(np.float64)
        self._voxel_tree = cKDTree(self._voxel_coords_flat)
        
        self.use_sparse = np.prod(self.shape) > 1e6
        self.coords = (gx, gy, gz)
