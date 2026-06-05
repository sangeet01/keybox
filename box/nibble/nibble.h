#ifndef NIBBLE_H
#define NIBBLE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * nibble.h - The C-Core for Negative Space Matrix Multiplication
 * Part of the KeyBox / Khukuri ecosystem.
 *
 * V4: Tier 3 (Langevin trajectory), Tier 4 (Induced Fit dynamics),
 * and Gap Closure channels (covalent, water network).
 */

#define N_CHANNELS 16

typedef struct {
    int dim_x;
    int dim_y;
    int dim_z;
    float resolution;
    float *data; /* Flattened [dim_x * dim_y * dim_z * N_CHANNELS] */
} NibbleGrid;

/* -------------------------------------------------------------------
 * Voxel channel mappings (Complementary Demands)
 * ------------------------------------------------------------------- */
typedef enum {
    CH_STERIC_DEMAND    = 0,  /* 0 = wall, 1 = void (ideal for drug atom) */
    CH_ELEC_DEMAND      = 1,  /* Complementary to protein charge */
    CH_HBA_DEMAND       = 2,  /* High if protein has H-bond donor */
    CH_HBD_DEMAND       = 3,  /* High if protein has H-bond acceptor */
    CH_LIPO_DEMAND      = 4,  /* High if protein surface is hydrophobic */
    CH_AROM_DEMAND      = 5,  /* High if protein surface has pi-stacking */
    CH_METAL_DEMAND     = 6,  /* High if protein has metal coordinating site */
    CH_CATION_DEMAND    = 7,  /* Seeking cation-pi interaction */
    CH_ANION_DEMAND     = 8,  /* Seeking anionic interaction */
    CH_PHOBIC_CORE      = 9,  /* Buried hydrophobic demand */
    CH_SOLVENT_EXPO     = 10, /* Solvent accessibility */
    /* Gap Closure channels */
    CH_FLEXIBILITY      = 11, /* Per-voxel residue flexibility (AA type derived) */
    CH_NUCLEOPHILICITY  = 12, /* Fukui f- : susceptibility to electrophilic attack */
    CH_ELECTROPHILICITY = 13, /* Fukui f+ : susceptibility to nucleophilic attack */
    CH_WATER_CONSERVED  = 14, /* Crystallographic water bridges (B-factor < 30) */
    CH_WATER_DYNAMIC    = 15  /* Predicted water occupancy probability field */
} NibbleChannel;

/* -------------------------------------------------------------------
 * Rigid-body molecule state for Langevin trajectory (Tier 3)
 * 6 DOF: 3 translation + quaternion rotation
 * ------------------------------------------------------------------- */
typedef struct {
    float  cx, cy, cz;       /* Center of mass position */
    float  vx, vy, vz;       /* Center of mass velocity */
    float  qw, qx, qy, qz;  /* Orientation quaternion (cumulative) */
    float  wx, wy, wz;       /* Angular velocity */
    int    n_atoms;
    float *local_coords;     /* Atom coords in current pose [n_atoms * 3] */
    float *ref_coords;       /* Atom coords in initial (reference) pose [n_atoms * 3]
                              * Rotation is always applied to ref_coords -> local_coords
                              * to prevent quaternion drift accumulation. */
    float *ch_vals;          /* Channel values per atom [n_atoms * N_CHANNELS] */
} NibbleMol;

/* -------------------------------------------------------------------
 * Core grid management
 * ------------------------------------------------------------------- */
NibbleGrid*  nibble_create_grid(int dx, int dy, int dz, float res);
void         nibble_free_grid(NibbleGrid *grid);
void         nibble_reset_steric(NibbleGrid *grid);

/* -------------------------------------------------------------------
 * Atom projection and scoring
 * ------------------------------------------------------------------- */
void   nibble_project_atom(NibbleGrid *grid, float x, float y, float z,
                            float radius, const float *channel_values);
float  nibble_compute_affinity(const NibbleGrid *pocket, const NibbleGrid *drug);
void   nibble_update_local(NibbleGrid *grid, int cx, int cy, int cz,
                            int radius, const float *diff_data);

/* -------------------------------------------------------------------
 * PDB loader (original + extended for water and flexibility channels)
 * ------------------------------------------------------------------- */
int nibble_load_pdb_pocket(NibbleGrid *grid, const char *pdb_path,
                            float center_x, float center_y, float center_z,
                            float range, float blur_radius);

/* -------------------------------------------------------------------
 * Tier 3: Langevin Trajectory Engine
 * ------------------------------------------------------------------- */

/* Compute gradient of affinity field at position (x,y,z) via finite difference */
void  nibble_gradient(const NibbleGrid *pocket, float x, float y, float z,
                      float *gx, float *gy, float *gz);

/* Allocate/free a NibbleMol from atom arrays */
NibbleMol* nibble_mol_create(int n_atoms, const float *local_coords,
                              const float *ch_vals);
void       nibble_mol_free(NibbleMol *mol);

/* Project a NibbleMol at current pose into a pre-created drug grid */
void nibble_mol_project(const NibbleMol *mol, NibbleGrid *drug_grid);

/* Score the current pose of mol against pocket */
float nibble_mol_score(const NibbleMol *mol, const NibbleGrid *pocket,
                       NibbleGrid *scratch_drug);

/* Quaternion-based rotation utilities */
void   nibble_quat_normalize(float *qw, float *qx, float *qy, float *qz);
void   nibble_quat_rotate_point(float x, float y, float z,
                                 float qw, float qx, float qy, float qz,
                                 float *rx, float *ry, float *rz);
void   nibble_quat_integrate(float *qw, float *qx, float *qy, float *qz,
                              float wx, float wy, float wz, float dt);

/* Inertia tensor and torque computation */
void   nibble_compute_inertia_tensor(const NibbleMol *mol,
                                      float I[3][3]);
void   nibble_compute_body_torque(const NibbleMol *mol, const NibbleGrid *pocket,
                                   float *tau_x, float *tau_y, float *tau_z);

/* Apply quaternion rotation to all atoms in molecule */
void   nibble_rotate_mol_atoms(NibbleMol *mol);

/* Single Langevin integration step (translational + rotational) */
void nibble_langevin_step(NibbleMol *mol, const NibbleGrid *pocket,
                           NibbleGrid *scratch_drug,
                           float dt, float gamma, float kT,
                           unsigned int *rng_state);

/* Full Langevin trajectory: returns minimum score found and writes best pose */
float nibble_trajectory(NibbleMol *mol, const NibbleGrid *pocket,
                         NibbleGrid *scratch_drug,
                         int n_steps, float dt, float gamma, float kT,
                         float *best_cx_out, float *best_cy_out, float *best_cz_out);

/* -------------------------------------------------------------------
 * Tier 4: Induced Fit Dynamic Protein Grid
 * ------------------------------------------------------------------- */

/* Mechanism 1: Apply time-varying Gaussian thermal breathing to grid */
void nibble_thermal_breathe(NibbleGrid *grid, float sigma_delta);

/* Mechanism 2: Elastic deformation of protein grid from ligand contact pressure */
void nibble_elastic_deform(NibbleGrid *protein, const NibbleGrid *ligand,
                            float elasticity_scale);

/* Mechanism 3: Linear interpolation between two endpoint conformer grids */
void nibble_interpolate_grids(NibbleGrid *dest,
                               const NibbleGrid *grid_open,
                               const NibbleGrid *grid_closed,
                               float alpha);

/* Compute pocket occupancy scalar q from ligand grid overlap */
float nibble_pocket_occupancy(const NibbleGrid *protein, const NibbleGrid *ligand,
                               float threshold);

/* Full Tier 4 frame update: interpolate + breathe + elastic deform */
void nibble_tier4_frame_update(NibbleGrid *protein_working,
                                const NibbleGrid *grid_open,
                                const NibbleGrid *grid_closed,
                                const NibbleGrid *ligand,
                                float sigma_delta,
                                float elasticity_scale);

/* -------------------------------------------------------------------
 * Gap Closure: Fukui Reactivity Channels (Gap 1)
 * ------------------------------------------------------------------- */

/* Project Fukui nucleophilicity/electrophilicity channels from partial charges.
   ch_elec is the partial charge array [n_atoms], coords is [n_atoms*3].
   Positive charge -> electrophilic (CH_ELECTROPHILICITY).
   Negative charge -> nucleophilic  (CH_NUCLEOPHILICITY). */
void nibble_project_fukui(NibbleGrid *grid, int n_atoms,
                           const float *coords, const float *partial_charges,
                           float blur_radius);

/* Check for covalent event: returns voxel linear index of first triggered site
   or -1 if none. threshold is the minimum overlap to trigger bond formation. */
int nibble_check_covalent(const NibbleGrid *protein, const NibbleGrid *ligand,
                           float threshold);

/* -------------------------------------------------------------------
 * Gap Closure: Virtual Water Network (Gap 3)
 * ------------------------------------------------------------------- */

/* Compute dynamic water occupancy field from existing electrostatic/HBA/HBD
   channels and store result in CH_WATER_DYNAMIC.
   delta_G_bulk is the cost of desolvating one water (~2.0 kcal/mol).
   kT at 298K = 0.593 kcal/mol. */
void nibble_compute_water_network(NibbleGrid *grid, float delta_G_bulk, float kT);

/* Compute desolvation penalty: sum of CH_WATER_DYNAMIC * delta_G_bulk
   over voxels where ligand steric > threshold. */
float nibble_desolvation_penalty(const NibbleGrid *protein, const NibbleGrid *ligand,
                                  float delta_G_bulk, float steric_threshold);

/* Full affinity score with water network and desolvation penalty included */
float nibble_compute_affinity_full(const NibbleGrid *pocket, const NibbleGrid *drug,
                                    float delta_G_bulk);

#endif /* NIBBLE_H */
