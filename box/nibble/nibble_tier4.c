/*
 * nibble_tier4.c - Induced Fit Dynamic Protein Grid (Tier 4)
 * Implements thermal breathing, elastic deformation, and normal mode interpolation.
 */
#include "nibble.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Mechanism 1: Thermal breathing via scaled Gaussian re-blur.
   sigma_delta is the additional blur radius to apply this frame (+ve = expand, -ve = contract).
   Implemented as a 3x3x3 box convolution approximation for speed. */
void nibble_thermal_breathe(NibbleGrid *grid, float sigma_delta) {
    if (fabsf(sigma_delta) < 1e-6f) return;
    size_t n_vox = (size_t)grid->dim_x * grid->dim_y * grid->dim_z;
    float  *tmp  = (float*)malloc(n_vox * N_CHANNELS * sizeof(float));
    if (!tmp) return;
    memcpy(tmp, grid->data, n_vox * N_CHANNELS * sizeof(float));

    float w_center = 1.0f - fabsf(sigma_delta) * 0.15f;
    float w_face   = fabsf(sigma_delta) * 0.025f; /* 6 faces */

    int dx = grid->dim_x, dy = grid->dim_y, dz = grid->dim_z;
    int nc = N_CHANNELS;

    for (int x = 1; x < dx-1; ++x) for (int y = 1; y < dy-1; ++y) for (int z = 1; z < dz-1; ++z) {
        float *out = &grid->data[((size_t)(x*dy + y)*dz + z) * nc];
        const float *c  = &tmp[((size_t)(x*dy + y)*dz + z) * nc];
        const float *xp = &tmp[((size_t)((x+1)*dy + y)*dz + z) * nc];
        const float *xn = &tmp[((size_t)((x-1)*dy + y)*dz + z) * nc];
        const float *yp = &tmp[((size_t)(x*dy + (y+1))*dz + z) * nc];
        const float *yn = &tmp[((size_t)(x*dy + (y-1))*dz + z) * nc];
        const float *zp = &tmp[((size_t)(x*dy + y)*dz + (z+1)) * nc];
        const float *zn = &tmp[((size_t)(x*dy + y)*dz + (z-1)) * nc];
        for (int c2 = 0; c2 < nc; ++c2)
            out[c2] = w_center * c[c2] + w_face * (xp[c2]+xn[c2]+yp[c2]+yn[c2]+zp[c2]+zn[c2]);
    }
    free(tmp);
}

/* Mechanism 2: Elastic deformation from ligand contact pressure.
   Where ligand steric overlaps protein wall, the protein field yields outward
   scaled by the per-voxel flexibility (CH_FLEXIBILITY channel). */
void nibble_elastic_deform(NibbleGrid *protein, const NibbleGrid *ligand,
                            float elasticity_scale) {
    size_t n_vox = (size_t)protein->dim_x * protein->dim_y * protein->dim_z;
    int dx = protein->dim_x, dy = protein->dim_y, dz = protein->dim_z;
    int nc = N_CHANNELS;

    for (size_t idx = 0; idx < n_vox; ++idx) {
        float *pp  = &protein->data[idx * nc];
        const float *dp = &ligand->data[idx * nc];

        float ps = pp[CH_STERIC_DEMAND]; /* protein steric */
        float ds = dp[CH_STERIC_DEMAND]; /* ligand steric */
        float flex = pp[CH_FLEXIBILITY]; /* residue flexibility, 0-1 */

        /* Clash: ligand in wall voxel */
        if (ds > 0.1f && ps < 0.1f) {
            float pressure = ds * elasticity_scale * (0.1f + flex);
            /* Soften the steric wall proportionally to flexibility */
            pp[CH_STERIC_DEMAND] += pressure;
            if (pp[CH_STERIC_DEMAND] > 1.0f) pp[CH_STERIC_DEMAND] = 1.0f;
        }
    }
}

/* Mechanism 3: Linear interpolation between open and closed conformer grids.
   alpha=0 gives grid_open, alpha=1 gives grid_closed. */
void nibble_interpolate_grids(NibbleGrid *dest,
                               const NibbleGrid *grid_open,
                               const NibbleGrid *grid_closed,
                               float alpha) {
    float beta = 1.0f - alpha;
    size_t n = (size_t)dest->dim_x * dest->dim_y * dest->dim_z * N_CHANNELS;
    for (size_t i = 0; i < n; ++i)
        dest->data[i] = beta * grid_open->data[i] + alpha * grid_closed->data[i];
}

/* Pocket occupancy: fraction of pocket voxels (steric void) occupied by ligand */
float nibble_pocket_occupancy(const NibbleGrid *protein, const NibbleGrid *ligand,
                               float threshold) {
    size_t n_vox = (size_t)protein->dim_x * protein->dim_y * protein->dim_z;
    int nc = N_CHANNELS;
    size_t pocket_count = 0, occupied = 0;

    for (size_t i = 0; i < n_vox; ++i) {
        float ps = protein->data[i * nc + CH_STERIC_DEMAND];
        if (ps > 0.5f) { /* pocket void voxel */
            ++pocket_count;
            if (ligand->data[i * nc + CH_STERIC_DEMAND] > threshold)
                ++occupied;
        }
    }
    if (pocket_count == 0) return 0.0f;
    return (float)occupied / (float)pocket_count;
}

/* Full Tier 4 frame update: interpolate then breathe then elastic deform */
void nibble_tier4_frame_update(NibbleGrid *protein_working,
                                const NibbleGrid *grid_open,
                                const NibbleGrid *grid_closed,
                                const NibbleGrid *ligand,
                                float sigma_delta,
                                float elasticity_scale) {
    /* Step 1: compute occupancy -> alpha for sigmoid interpolation */
    float q = nibble_pocket_occupancy(protein_working, ligand, 0.1f);
    /* Sigmoid: alpha = 1 / (1 + exp(-10*(q-0.5))) */
    float alpha = 1.0f / (1.0f + expf(-10.0f * (q - 0.5f)));

    /* Step 2: interpolate between conformers */
    nibble_interpolate_grids(protein_working, grid_open, grid_closed, alpha);

    /* Step 3: thermal breathing */
    nibble_thermal_breathe(protein_working, sigma_delta);

    /* Step 4: elastic deformation from ligand contact */
    nibble_elastic_deform(protein_working, ligand, elasticity_scale);
}
