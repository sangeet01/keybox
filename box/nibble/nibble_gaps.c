/*
 * nibble_gaps.c - Gap Closure: Fukui Reactivity Channels + Virtual Water Network
 * Implements covalent bond detection and water-mediated H-bond capture.
 */
#include "nibble.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Gap 1: Project Fukui reactivity channels from partial charges.
   Positive charge -> electrophilic site (CH_ELECTROPHILICITY).
   Negative charge -> nucleophilic site  (CH_NUCLEOPHILICITY). */
void nibble_project_fukui(NibbleGrid *grid, int n_atoms,
                           const float *coords, const float *partial_charges,
                           float blur_radius) {
    for (int i = 0; i < n_atoms; ++i) {
        float q = partial_charges[i];
        float ch[N_CHANNELS] = {0};
        if (q > 0.0f)
            ch[CH_ELECTROPHILICITY] = q;
        else
            ch[CH_NUCLEOPHILICITY]  = -q; /* store as positive magnitude */
        nibble_project_atom(grid,
                            coords[i*3+0], coords[i*3+1], coords[i*3+2],
                            blur_radius, ch);
    }
}

/* Check for covalent event: returns linear voxel index of first triggered site,
   or -1 if no covalent event detected. */
int nibble_check_covalent(const NibbleGrid *protein, const NibbleGrid *ligand,
                           float threshold) {
    size_t n_vox = (size_t)protein->dim_x * protein->dim_y * protein->dim_z;
    int nc = N_CHANNELS;
    for (size_t i = 0; i < n_vox; ++i) {
        const float *pp = &protein->data[i * nc];
        const float *dp = &ligand->data[i * nc];
        /* Covalent event: protein nucleophilicity overlaps ligand electrophilicity */
        float overlap = pp[CH_NUCLEOPHILICITY] * dp[CH_ELECTROPHILICITY];
        if (overlap > threshold) return (int)i;
        /* Also check reverse: protein electrophilic, ligand nucleophilic */
        overlap = pp[CH_ELECTROPHILICITY] * dp[CH_NUCLEOPHILICITY];
        if (overlap > threshold) return (int)i;
    }
    return -1;
}

/* Gap 3: Compute dynamic water occupancy field from local electrostatic + HB channels.
   P_water(v) = exp(-delta_G_water(v) / kT), stored in CH_WATER_DYNAMIC. */
void nibble_compute_water_network(NibbleGrid *grid, float delta_G_bulk, float kT) {
    size_t n_vox = (size_t)grid->dim_x * grid->dim_y * grid->dim_z;
    int nc = N_CHANNELS;
    float inv_kT = 1.0f / kT;
    float Z = 0.0f; /* partition function accumulator */

    /* First pass: compute raw occupancy probabilities */
    float *raw = (float*)malloc(n_vox * sizeof(float));
    if (!raw) return;

    for (size_t i = 0; i < n_vox; ++i) {
        float *vp = &grid->data[i * nc];
        /* Only place water in void voxels (near protein surface but not inside) */
        float ps = vp[CH_STERIC_DEMAND];
        if (ps < 0.3f) { raw[i] = 0.0f; continue; } /* inside protein wall */

        /* delta_G_water approximation from field channels */
        float dG = delta_G_bulk
                   - 1.5f * fabsf(vp[CH_ELEC_DEMAND])
                   - 1.2f * vp[CH_HBA_DEMAND]
                   - 1.2f * vp[CH_HBD_DEMAND];

        raw[i] = expf(-dG * inv_kT);
        Z += raw[i];
    }

    /* Second pass: normalize and store in CH_WATER_DYNAMIC */
    float inv_Z = (Z > 1e-9f) ? (1.0f / Z) : 0.0f;
    for (size_t i = 0; i < n_vox; ++i)
        grid->data[i * nc + CH_WATER_DYNAMIC] = raw[i] * inv_Z * n_vox;

    free(raw);
}

/* Desolvation penalty: cost of displacing water from voxels where ligand inserts */
float nibble_desolvation_penalty(const NibbleGrid *protein, const NibbleGrid *ligand,
                                  float delta_G_bulk, float steric_threshold) {
    size_t n_vox = (size_t)protein->dim_x * protein->dim_y * protein->dim_z;
    int nc = N_CHANNELS;
    float penalty = 0.0f;
    for (size_t i = 0; i < n_vox; ++i) {
        float ds = ligand->data[i * nc + CH_STERIC_DEMAND];
        if (ds > steric_threshold) {
            float p_water = protein->data[i * nc + CH_WATER_DYNAMIC];
            penalty += p_water * delta_G_bulk;
        }
    }
    return penalty;
}

/* Full affinity with water network bonus and desolvation penalty */
float nibble_compute_affinity_full(const NibbleGrid *pocket, const NibbleGrid *drug,
                                    float delta_G_bulk) {
    float base = nibble_compute_affinity(pocket, drug);

    size_t n_vox = (size_t)pocket->dim_x * pocket->dim_y * pocket->dim_z;
    int nc = N_CHANNELS;
    float water_bonus = 0.0f;
    float desolv      = 0.0f;

    for (size_t i = 0; i < n_vox; ++i) {
        const float *pp = &pocket->data[i * nc];
        const float *dp = &drug->data[i * nc];
        float p_conserved = pp[CH_WATER_CONSERVED];
        float p_dynamic   = pp[CH_WATER_DYNAMIC];
        float ds          = dp[CH_STERIC_DEMAND];

        /* Water-mediated H-bond bonus: ligand HBA/HBD overlapping a water voxel */
        water_bonus += p_conserved * (dp[CH_HBA_DEMAND] + dp[CH_HBD_DEMAND]) * 0.5f;
        water_bonus += p_dynamic   * (dp[CH_HBA_DEMAND] + dp[CH_HBD_DEMAND]) * 0.3f;

        /* Desolvation: penalize displacing occupied water voxels */
        if (ds > 0.1f)
            desolv += (p_conserved + p_dynamic) * delta_G_bulk;
    }

    return base + water_bonus - desolv;
}
