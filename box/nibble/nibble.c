/*
 * nibble.c (V3 - Production)
 * Negative Space Matrix Multiplication Docking Engine
 *
 * Architecture (theory.txt):
 *   Phase I  Coarse  1.0 A : Sterics + electrostatics, ~10 us per molecule
 *   Phase II Sniper  0.25A : Full 11-channel + B-factor Gaussian blur, ~5 ms per molecule
 *
 * Demand tensor convention:
 *   Steric channel initialised to +1.0 (all void).
 *   Protein atoms project -1.0 Gaussian, driving wall voxels toward 0.
 *   Drug atoms project +1.0 steric.
 *   Hadamard sum: positive in void (good fit), negative in wall (clash).
 */

#include "nibble.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define CLIP_MAX 200.0f

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ------------------------------------------------------------------ */
/* Grid management                                                      */
/* ------------------------------------------------------------------ */

NibbleGrid* nibble_create_grid(int dx, int dy, int dz, float res) {
    NibbleGrid *g = (NibbleGrid*)malloc(sizeof(NibbleGrid));
    if (!g) return NULL;
    g->dim_x = dx; g->dim_y = dy; g->dim_z = dz; g->resolution = res;
    size_t n = (size_t)dx * dy * dz * N_CHANNELS;
    g->data = (float*)calloc(n, sizeof(float));
    if (!g->data) { free(g); return NULL; }
    return g;
}

void nibble_free_grid(NibbleGrid *g) {
    if (g) { if (g->data) free(g->data); free(g); }
}

/* Set steric channel to 1.0 everywhere (pure void before loading protein) */
void nibble_reset_steric(NibbleGrid *g) {
    size_t n_vox = (size_t)g->dim_x * g->dim_y * g->dim_z;
    for (size_t i = 0; i < n_vox; ++i)
        g->data[i * N_CHANNELS + CH_STERIC_DEMAND] = 1.0f;
}

/* ------------------------------------------------------------------ */
/* Gaussian atom projection                                             */
/* ------------------------------------------------------------------ */

void nibble_project_atom(NibbleGrid *g, float x, float y, float z,
                          float radius, const float *ch) {
    float inv_res = 1.0f / g->resolution;
    int ix = (int)(x * inv_res);
    int iy = (int)(y * inv_res);
    int iz = (int)(z * inv_res);
    int r_vox    = (int)(radius * inv_res) + 1;
    float r2inv  = 1.0f / (2.0f * radius * radius);
    float lim_sq = 4.0f * radius * radius;

    int x0 = ix - r_vox; if (x0 < 0) x0 = 0;
    int x1 = ix + r_vox; if (x1 >= g->dim_x) x1 = g->dim_x - 1;
    int y0 = iy - r_vox; if (y0 < 0) y0 = 0;
    int y1 = iy + r_vox; if (y1 >= g->dim_y) y1 = g->dim_y - 1;
    int z0 = iz - r_vox; if (z0 < 0) z0 = 0;
    int z1 = iz + r_vox; if (z1 >= g->dim_z) z1 = g->dim_z - 1;

    for (int nx = x0; nx <= x1; ++nx) {
        float vx  = nx * g->resolution;
        float dxs = (x - vx) * (x - vx);
        if (dxs > lim_sq) continue;
        for (int ny = y0; ny <= y1; ++ny) {
            float vy   = ny * g->resolution;
            float dxys = dxs + (y - vy) * (y - vy);
            if (dxys > lim_sq) continue;
            for (int nz = z0; nz <= z1; ++nz) {
                float vz = nz * g->resolution;
                float dz = z - vz;
                float d2 = dxys + dz * dz;
                if (d2 >= lim_sq) continue;
                float w  = expf(-d2 * r2inv);
                float *vp = &g->data[
                    (size_t)(nx * g->dim_y * g->dim_z + ny * g->dim_z + nz)
                    * N_CHANNELS];
                for (int c = 0; c < N_CHANNELS; ++c)
                    vp[c] = clampf(vp[c] + ch[c] * w, -CLIP_MAX, CLIP_MAX);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Affinity score: Frobenius inner product of pocket and drug tensors  */
/* ------------------------------------------------------------------ */

float nibble_compute_affinity(const NibbleGrid *pocket, const NibbleGrid *drug) {
    size_t n_vox = (size_t)pocket->dim_x * pocket->dim_y * pocket->dim_z;
    float score = 0.0f;
    const float *pp = pocket->data;
    const float *dp = drug->data;

    for (size_t i = 0; i < n_vox; ++i) {
        float ps = pp[0];   /* pocket steric (1 = void, 0/neg = wall) */
        float ds = dp[0];   /* drug   steric (1 = drug atom present)  */

        /* Hard clash: drug occupies wall space */
        if (ds > 0.1f && ps < 0.05f) {
            score -= 500.0f * ds;
        } else {
            /* Unrolled Hadamard sum over all 11 channels */
            score += ps      * ds;       /* steric complement        */
            score += pp[1]  * dp[1];    /* electrostatics           */
            score += pp[2]  * dp[2];    /* HBA demand               */
            score += pp[3]  * dp[3];    /* HBD demand               */
            score += pp[4]  * dp[4];    /* lipophilicity            */
            score += pp[5]  * dp[5];    /* aromaticity / pi-stack   */
            score += pp[6]  * dp[6];    /* metal coordination       */
            score += pp[7]  * dp[7];    /* cation-pi                */
            score += pp[8]  * dp[8];    /* anionic                  */
            score += pp[9]  * dp[9];    /* buried hydrophobic core  */
            score += pp[10] * dp[10];   /* solvent exposure         */
        }
        pp += N_CHANNELS;
        dp += N_CHANNELS;
    }
    return score;
}

/* ------------------------------------------------------------------ */
/* Local voxel update: O(r^3) mutation patch                           */
/* Called by Pincer when bacteria mutates one amino acid.               */
/* diff_data is an N_CHANNELS array added to every voxel in the sphere. */
/* ------------------------------------------------------------------ */

void nibble_update_local(NibbleGrid *g, int cx, int cy, int cz,
                          int radius, const float *diff_data) {
    int r2 = radius * radius;
    int x0 = cx - radius; if (x0 < 0) x0 = 0;
    int x1 = cx + radius; if (x1 >= g->dim_x) x1 = g->dim_x - 1;
    int y0 = cy - radius; if (y0 < 0) y0 = 0;
    int y1 = cy + radius; if (y1 >= g->dim_y) y1 = g->dim_y - 1;
    int z0 = cz - radius; if (z0 < 0) z0 = 0;
    int z1 = cz + radius; if (z1 >= g->dim_z) z1 = g->dim_z - 1;

    for (int nx = x0; nx <= x1; ++nx) {
        int ddx = (nx - cx) * (nx - cx);
        for (int ny = y0; ny <= y1; ++ny) {
            int ddy = ddx + (ny - cy) * (ny - cy);
            for (int nz = z0; nz <= z1; ++nz) {
                if (ddy + (nz - cz) * (nz - cz) > r2) continue;
                float *vp = &g->data[
                    (size_t)(nx * g->dim_y * g->dim_z + ny * g->dim_z + nz)
                    * N_CHANNELS];
                for (int c = 0; c < N_CHANNELS; ++c)
                    vp[c] = clampf(vp[c] + diff_data[c], -CLIP_MAX, CLIP_MAX);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* PDB pocket loader                                                    */
/* Reads ATOM/HETATM records, extracts coordinates, element, B-factor. */
/* Projects complementary demand fields into the negative-space tensor. */
/* ------------------------------------------------------------------ */

int nibble_load_pdb_pocket(NibbleGrid *g, const char *pdb_path,
                            float cx, float cy, float cz,
                            float range, float blur_radius) {
    FILE *fp = fopen(pdb_path, "r");
    if (!fp) return -1;

    /* Pre-fill steric channel to 1.0 (all void). Protein atoms will
       project -1.0 Gaussian onto steric, driving walls toward 0. */
    nibble_reset_steric(g);

    char  line[128];
    int   n_atoms = 0;
    float sx      = cx - range;
    float sy      = cy - range;
    float sz      = cz - range;
    float range2  = range * range;

    while (fgets(line, sizeof(line), fp)) {
        int is_atom   = (strncmp(line, "ATOM  ", 6) == 0);
        int is_hetatm = (strncmp(line, "HETATM", 6) == 0);
        if (!is_atom && !is_hetatm) continue;

        int len = (int)strlen(line);
        if (len < 54) continue;   /* need at least XYZ columns */

        /* Coordinates: PDB fixed columns 31-38, 39-46, 47-54 (0-indexed 30,38,46) */
        char buf[9];
        strncpy(buf, line + 30, 8); buf[8] = '\0'; float ax = (float)atof(buf);
        strncpy(buf, line + 38, 8); buf[8] = '\0'; float ay = (float)atof(buf);
        strncpy(buf, line + 46, 8); buf[8] = '\0'; float az = (float)atof(buf);

        float ddx = ax - cx, ddy = ay - cy, ddz = az - cz;
        if (ddx*ddx + ddy*ddy + ddz*ddz > range2) continue;

        /*
         * B-factor (columns 61-66, 0-indexed 60-65) drives adaptive blur.
         * Theory.txt: atoms with high B-factor (flexible loops) get a wider
         * Gaussian — this approximates 100ns MD thermal averaging in O(1).
         * sigma(B) = sqrt(B / 8*pi^2), scaled against the base blur_radius.
         */
        float atom_blur = blur_radius;
        if (len > 66) {
            char bfbuf[7];
            strncpy(bfbuf, line + 60, 6); bfbuf[6] = '\0';
            float bval = (float)atof(bfbuf);
            if (bval > 0.0f) {
                float b_sigma = sqrtf(bval / 78.957f); /* 8*pi^2 */
                b_sigma = b_sigma > 2.0f ? 2.0f : b_sigma;
                atom_blur = blur_radius * (1.0f + b_sigma);
            }
        }

        /* Element symbol: columns 77-78 (0-indexed 76-77), or atom name col 13 */
        char e = ' ';
        if (len > 77 && line[76] != ' ') {
            e = line[76];
        } else if (len > 77 && line[77] != ' ') {
            e = line[77];
        } else {
            /* Fallback: first non-space character of atom name (col 13-14) */
            e = (line[12] != ' ') ? line[12] : line[13];
        }

        /* Build the complementary demand vector.
         * Steric: -1.0 marks protein wall (projection drives void to 0).
         * Other channels encode what the drug must supply to bind here. */
        float d[N_CHANNELS] = {0.0f};
        d[CH_STERIC_DEMAND] = -1.0f;   /* always: protein atom = wall */

        switch (e) {
            case 'O':
                /* Oxygen donor -> drug needs H-bond acceptor */
                d[CH_HBA_DEMAND]   =  1.0f;
                d[CH_ELEC_DEMAND]  = -0.6f;   /* electronegative wall */
                d[CH_SOLVENT_EXPO] =  0.3f;
                break;
            case 'N':
                /* Nitrogen acceptor -> drug needs H-bond donor */
                d[CH_HBD_DEMAND]   =  1.0f;
                d[CH_ELEC_DEMAND]  =  0.4f;
                break;
            case 'C':
                d[CH_LIPO_DEMAND]  =  0.7f;
                d[CH_PHOBIC_CORE]  =  0.5f;
                break;
            case 'S':
                d[CH_HBA_DEMAND]   =  0.5f;
                d[CH_METAL_DEMAND] =  0.3f;
                d[CH_LIPO_DEMAND]  =  0.4f;
                break;
            case 'P':
                d[CH_ANION_DEMAND] =  0.8f;
                d[CH_HBA_DEMAND]   =  0.6f;
                break;
            case 'F':
            case 'l':   /* Cl: second char */
                d[CH_LIPO_DEMAND]  =  0.5f;
                d[CH_ELEC_DEMAND]  = -0.5f;
                break;
            case 'H':
                d[CH_HBD_DEMAND]   =  0.2f;
                break;
            default:
                /* Unknown element: only steric wall */
                break;
        }

        nibble_project_atom(g, ax - sx, ay - sy, az - sz, atom_blur, d);
        n_atoms++;
    }

    fclose(fp);
    return n_atoms;
}
