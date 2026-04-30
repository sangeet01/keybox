#ifndef NIBBLE_H
#define NIBBLE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* 
 * nibble.h — The C-Core for Negative Space Matrix Multiplication
 * Part of the KeyBox / Khukuri ecosystem.
 */

#define N_CHANNELS 11

typedef struct {
    int dim_x;
    int dim_y;
    int dim_z;
    float resolution;
    float *data; // Flattened array of size [dim_x * dim_y * dim_z * N_CHANNELS]
} NibbleGrid;

/* Voxel channel mappings (Complementary Demands) */
typedef enum {
    CH_STERIC_DEMAND = 0,    // 0 = wall, 1 = empty (ideal for drug atom)
    CH_ELEC_DEMAND   = 1,    // Complementary to protein charge
    CH_HBA_DEMAND    = 2,    // High if protein has H-bond donor
    CH_HBD_DEMAND    = 3,    // High if protein has H-bond acceptor
    CH_LIPO_DEMAND   = 4,    // High if protein surface is hydrophobic
    CH_AROM_DEMAND   = 5,    // High if protein surface has pi-stacking potential
    CH_METAL_DEMAND  = 6,    // High if protein has metal coordinating site
    CH_CATION_DEMAND = 7,    // Specifically seeking cation-pi interaction
    CH_ANION_DEMAND  = 8,    // Specifically seeking anionic interaction
    CH_PHOBIC_CORE   = 9,    // Buried hydrophobic demand
    CH_SOLVENT_EXPO  = 10    // Solvent accessibility (0 = buried, 1 = surface)
} NibbleChannel;

/* Core Engine Functions */
NibbleGrid* nibble_create_grid(int dx, int dy, int dz, float res);
void nibble_free_grid(NibbleGrid *grid);

/* Reset steric channel to 1.0 (void). Call before projecting protein atoms. */
void nibble_reset_steric(NibbleGrid *grid);

/* PDB Ingestion - Loads a pocket and inverts it into demand voxels.
   Reads B-factor for adaptive Gaussian blur (sniper mode thermal averaging). */
int nibble_load_pdb_pocket(NibbleGrid *grid, const char *pdb_path,
                            float center_x, float center_y, float center_z,
                            float range, float blur_radius);

/* Frobenius inner product affinity score. */
float nibble_compute_affinity(const NibbleGrid *pocket, const NibbleGrid *drug);

/* Local voxel patch update for O(r^3) mutation handling (Pincer). */
void nibble_update_local(NibbleGrid *grid, int cx, int cy, int cz,
                          int radius, const float *diff_data);

/* Gaussian atom projection - projects drug or protein atom fields into grid. */
void nibble_project_atom(NibbleGrid *grid, float x, float y, float z,
                          float radius, const float *channel_values);

#endif // NIBBLE_H
