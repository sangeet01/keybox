/*
 * nibble_peptide.c
 * Peptide-Protein Docking Extension for the Nibble Voxel Engine.
 *
 * Implements:
 *   nibble_peptide_build()             — sequence → PEP_Peptide
 *   nibble_peptide_add_constraint()    — staple/disulfide/lactam/cyclic
 *   nibble_peptide_scan_grooves()      — binding groove detection
 *   nibble_peptide_ss_bias()           — secondary structure bonus field
 *   nibble_peptide_project()           — segment projection into grid
 *   nibble_peptide_trajectory()        — multi-segment Langevin
 *   nibble_peptide_hydrophobic_moment()— Eisenberg amphipathic moment
 *   nibble_peptide_dock()              — full pipeline
 *   nibble_peptide_print_result()      — diagnostic output
 *   nibble_peptide_print()             — peptide summary
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#include "nibble_peptide.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Amino acid property table
 * Hydrophobicity: Eisenberg consensus scale (-1 polar, +1 hydrophobic)
 * ----------------------------------------------------------------------- */
typedef struct {
    char one;
    char three[4];
    float hydrophobicity;
    float charge;       /* formal charge at pH 7 */
    int   hbd;          /* H-bond donors on side chain */
    int   hba;          /* H-bond acceptors on side chain */
    float mass;         /* Da */
} _AA_Props;

static const _AA_Props _AA_TABLE[] = {
    {'A', "ALA",  0.62f,  0.0f, 0, 0, 89.0f},
    {'R', "ARG", -2.53f,  1.0f, 5, 0, 174.0f},
    {'N', "ASN", -0.78f,  0.0f, 2, 2, 132.0f},
    {'D', "ASP", -0.90f, -1.0f, 0, 2, 133.0f},
    {'C', "CYS",  0.29f,  0.0f, 1, 0, 121.0f},
    {'Q', "GLN", -0.85f,  0.0f, 2, 2, 146.0f},
    {'E', "GLU", -0.74f, -1.0f, 0, 2, 147.0f},
    {'G', "GLY",  0.48f,  0.0f, 0, 0, 75.0f},
    {'H', "HIS", -0.40f,  0.1f, 2, 1, 155.0f},
    {'I', "ILE",  1.38f,  0.0f, 0, 0, 131.0f},
    {'L', "LEU",  1.06f,  0.0f, 0, 0, 131.0f},
    {'K', "LYS", -1.50f,  1.0f, 3, 0, 146.0f},
    {'M', "MET",  0.64f,  0.0f, 0, 1, 149.0f},
    {'F', "PHE",  1.19f,  0.0f, 0, 0, 165.0f},
    {'P', "PRO",  0.12f,  0.0f, 0, 0, 115.0f},
    {'S', "SER", -0.18f,  0.0f, 1, 1, 105.0f},
    {'T', "THR", -0.05f,  0.0f, 1, 1, 119.0f},
    {'W', "TRP", -0.90f,  0.0f, 2, 0, 204.0f},
    {'Y', "TYR",  0.26f,  0.0f, 2, 1, 181.0f},
    {'V', "VAL",  1.08f,  0.0f, 0, 0, 117.0f},
    {'\0', "",    0.0f,   0.0f, 0, 0,   0.0f},
};

static const _AA_Props *_lookup_aa(char one) {
    for (int i = 0; _AA_TABLE[i].one != '\0'; i++) {
        if (_AA_TABLE[i].one == one) return &_AA_TABLE[i];
    }
    return &_AA_TABLE[0]; /* default to ALA */
}

/* Equilibrium distances for constraints */
static float _constraint_eq_dist(PEP_ConstraintType type) {
    switch (type) {
        case PEP_CONSTRAINT_STAPLE:    return 8.0f;   /* i to i+4 Cα Cα ~8Å in helix */
        case PEP_CONSTRAINT_CYCLIC:    return 3.8f;   /* N to C Cα distance */
        case PEP_CONSTRAINT_DISULFIDE: return 4.5f;   /* Cβ-Cβ distance for S-S */
        case PEP_CONSTRAINT_LACTAM:    return 5.5f;   /* side-chain to side-chain */
        default:                       return 6.0f;
    }
}

/* Internal: clamp float */
static inline float _clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Internal: squared distance */
static inline float _dist2(
    float ax, float ay, float az,
    float bx, float by, float bz)
{
    float dx = ax-bx, dy = ay-by, dz = az-bz;
    return dx*dx + dy*dy + dz*dz;
}

/* Internal: xorshift RNG */
static float _rng_float(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return ((float)(*s & 0x7FFFFFFF)) / (float)0x7FFFFFFF;
}
static float _rng_normal(unsigned int *s) {
    float u1 = _rng_float(s) + 1e-8f, u2 = _rng_float(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530f * u2);
}

/* Internal: quaternion rotate vector */
static void _quat_rotate(
    float qw, float qx, float qy, float qz,
    float vx, float vy, float vz,
    float *ox, float *oy, float *oz)
{
    /* v' = q * v * q^-1 */
    float tx = 2.0f*(qy*vz - qz*vy);
    float ty = 2.0f*(qz*vx - qx*vz);
    float tz = 2.0f*(qx*vy - qy*vx);
    *ox = vx + qw*tx + qy*tz - qz*ty;
    *oy = vy + qw*ty + qz*tx - qx*tz;
    *oz = vz + qw*tz + qx*ty - qy*tx;
}

/* -----------------------------------------------------------------------
 * nibble_peptide_build
 * ----------------------------------------------------------------------- */
int nibble_peptide_build(
    const char  *sequence,
    const char  *ss_string,
    int          is_cyclic,
    PEP_Peptide *out)
{
    if (!sequence || !out) return -1;
    int n = (int)strlen(sequence);
    if (n <= 0 || n > PEP_MAX_RESIDUES) return -1;

    memset(out, 0, sizeof(PEP_Peptide));
    out->n_residues = n;
    out->is_cyclic  = is_cyclic;

    float net_charge = 0.0f;
    float total_mass = 0.0f;

    /* Count helix/strand residues for dominant SS */
    int n_helix = 0, n_strand = 0;

    /* Build residues in extended conformation (phi=-120, psi=120) */
    /* Cα placement: along z-axis initially, extended */
    float z = 0.0f;

    for (int i = 0; i < n; i++) {
        PEP_Residue *res = &out->residues[i];
        char one = sequence[i];
        const _AA_Props *aa = _lookup_aa(one);

        res->one_letter   = one;
        memcpy(res->resname, aa->three, 4);
        res->hydrophobicity = aa->hydrophobicity;
        res->charge         = aa->charge;
        res->hbd_count      = aa->hbd;
        res->hba_count      = aa->hba;
        res->sasa           = 150.0f;  /* approximate extended conformation */
        res->is_constrained = 0;

        /* Secondary structure */
        if (ss_string && ss_string[i] != '\0') {
            char ss = ss_string[i];
            if (ss == 'H' || ss == 'h') {
                res->ss = PEP_SS_HELIX;
                res->phi = -1.047f;  /* -60 degrees */
                res->psi = -0.698f;  /* -40 degrees */
                z += PEP_RISE_PER_RES_HELIX;
                n_helix++;
            } else if (ss == 'E' || ss == 'e') {
                res->ss = PEP_SS_STRAND;
                res->phi = -2.182f;  /* -125 degrees */
                res->psi =  2.356f;  /* 135 degrees */
                z += PEP_RISE_PER_RES_BETA;
                n_strand++;
            } else {
                res->ss = PEP_SS_COIL;
                res->phi = -1.222f;  /* extended coil */
                res->psi =  1.047f;
                z += PEP_RISE_PER_RES_BETA * 0.8f;
            }
        } else {
            res->ss  = PEP_SS_COIL;
            res->phi = -1.222f;
            res->psi =  1.047f;
            z       += PEP_RISE_PER_RES_BETA * 0.8f;
        }

        /* Cα position: extended along z */
        res->ca_x = 0.0f;
        res->ca_y = 0.0f;
        res->ca_z = z;

        /* Channel demands for this residue */
        memset(res->d, 0, sizeof(res->d));
        res->d[CH_STERIC_DEMAND] = -1.0f;  /* occupies space */
        res->d[CH_LIPO_DEMAND]   = _clampf(aa->hydrophobicity, 0.0f, 1.0f);
        res->d[CH_ELEC_DEMAND]   = aa->charge * 0.5f;
        res->d[CH_HBD_DEMAND]    = aa->hbd > 0 ? 0.7f : 0.0f;
        res->d[CH_HBA_DEMAND]    = aa->hba > 0 ? 0.7f : 0.0f;
        if (one == 'F' || one == 'Y' || one == 'W' || one == 'H')
            res->d[CH_AROM_DEMAND] = 0.8f;

        net_charge += aa->charge;
        total_mass  += aa->mass;
    }

    out->net_charge = net_charge;
    out->length_angstrom = z;

    /* Centre of mass: Cα centroid */
    float com_x = 0, com_y = 0, com_z = 0;
    for (int i = 0; i < n; i++) {
        com_x += out->residues[i].ca_x;
        com_y += out->residues[i].ca_y;
        com_z += out->residues[i].ca_z;
    }
    float inv_n = 1.0f / n;
    out->com_x = com_x * inv_n;
    out->com_y = com_y * inv_n;
    out->com_z = com_z * inv_n;

    /* Dominant secondary structure */
    if (n_helix > n/2)        out->dominant_ss = PEP_SS_HELIX;
    else if (n_strand > n/3)  out->dominant_ss = PEP_SS_STRAND;
    else                      out->dominant_ss = PEP_SS_COIL;

    /* Build flexible segments: 4 residues per segment */
    int seg_size = 4;
    int n_segs = (n + seg_size - 1) / seg_size;
    if (n_segs > PEP_MAX_SEGMENTS) n_segs = PEP_MAX_SEGMENTS;
    out->n_segments = n_segs;

    for (int s = 0; s < n_segs; s++) {
        PEP_Segment *seg = &out->segments[s];
        seg->res_start = s * seg_size;
        seg->res_end   = (s + 1) * seg_size - 1;
        if (seg->res_end >= n) seg->res_end = n - 1;

        /* Segment COM */
        float sx = 0, sy = 0, sz = 0;
        float seg_mass = 0;
        for (int r = seg->res_start; r <= seg->res_end; r++) {
            const _AA_Props *aa = _lookup_aa(out->residues[r].one_letter);
            sx += out->residues[r].ca_x * aa->mass;
            sy += out->residues[r].ca_y * aa->mass;
            sz += out->residues[r].ca_z * aa->mass;
            seg_mass += aa->mass;
        }
        seg->mass = seg_mass;
        float inv_m = seg_mass > 0 ? 1.0f / seg_mass : 1.0f;
        seg->cx = sx * inv_m;
        seg->cy = sy * inv_m;
        seg->cz = sz * inv_m;

        /* Identity quaternion */
        seg->qw = 1.0f; seg->qx = 0.0f; seg->qy = 0.0f; seg->qz = 0.0f;

        /* Approximate inertia (sphere) */
        float r_sphere = cbrtf(3.0f * seg_mass / (4.0f * 3.14159f * 1.35f));
        seg->inertia_xx = seg->inertia_yy = seg->inertia_zz =
            0.4f * seg_mass * r_sphere * r_sphere;

        /* Aggregate channel demands */
        memset(seg->d, 0, sizeof(seg->d));
        for (int r = seg->res_start; r <= seg->res_end; r++) {
            for (int c = 0; c < N_CHANNELS; c++) {
                seg->d[c] += out->residues[r].d[c];
            }
        }
        /* Normalise by residue count */
        int n_res_seg = seg->res_end - seg->res_start + 1;
        for (int c = 0; c < N_CHANNELS; c++) {
            seg->d[c] /= n_res_seg;
        }
    }

    /* Hydrophobic moment */
    out->hydrophobic_moment = nibble_peptide_hydrophobic_moment(
        out, out->dominant_ss
    );

    return n;
}

/* -----------------------------------------------------------------------
 * nibble_peptide_add_constraint
 * ----------------------------------------------------------------------- */
int nibble_peptide_add_constraint(
    PEP_Peptide       *pep,
    PEP_ConstraintType type,
    int                res_i,
    int                res_j,
    float              spring_k)
{
    if (!pep || pep->n_constraints >= PEP_MAX_CONSTRAINTS) return -1;
    if (res_i < 0 || res_j >= pep->n_residues || res_i >= res_j) return -1;

    int idx = pep->n_constraints++;
    PEP_Constraint *c = &pep->constraints[idx];
    c->type       = type;
    c->res_i      = res_i;
    c->res_j      = res_j;
    c->eq_distance = spring_k > 0 ? _constraint_eq_dist(type) : _constraint_eq_dist(type);
    c->spring_k   = spring_k > 0 ? spring_k : 10.0f;

    pep->residues[res_i].is_constrained = 1;
    pep->residues[res_j].is_constrained = 1;

    if (type == PEP_CONSTRAINT_CYCLIC) pep->is_cyclic   = 1;
    if (type == PEP_CONSTRAINT_STAPLE) pep->is_stapled  = 1;

    return idx;
}

/* -----------------------------------------------------------------------
 * nibble_peptide_scan_grooves
 * ----------------------------------------------------------------------- */
int nibble_peptide_scan_grooves(
    const NibbleGrid *grid,
    PEP_Groove       *grooves,
    float             min_len)
{
    if (!grid || !grooves) return 0;
    if (min_len <= 0) min_len = PEP_GROOVE_MIN_LEN;

    float res = grid->resolution;
    int dx = grid->dim_x, dy = grid->dim_y, dz = grid->dim_z;
    int nc = N_CHANNELS;
    int n_grooves = 0;

    /*
     * Groove detection algorithm:
     *   A groove is a region where:
     *   (a) steric field transitions from wall to void (surface boundary)
     *   (b) the steric void is elongated in one direction (ridge shape)
     *   (c) flanking voxels have higher channel values than the groove axis
     *
     * Implementation: scan each axis (x, y, z) for elongated void regions.
     * For each candidate groove, compute:
     *   - axis direction
     *   - length, width, depth
     *   - integrated affinity (sum of non-steric channels along groove)
     */

    /* Search along all three principal axes */
    int axes[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    int dims[3]    = {dx, dy, dz};

    for (int axis = 0; axis < 3 && n_grooves < PEP_MAX_GROOVES; axis++) {
        int a0 = axis;
        int a1 = (axis + 1) % 3;
        int a2 = (axis + 2) % 3;
        int d0 = dims[a0], d1 = dims[a1], d2 = dims[a2];

        /* Scan cross-sections perpendicular to this axis */
        for (int i1 = 2; i1 < d1-2 && n_grooves < PEP_MAX_GROOVES; i1++) {
            for (int i2 = 2; i2 < d2-2; i2++) {
                /* Look for void run along a0 */
                int groove_start = -1;
                float groove_sum_aff = 0.0f;
                int   groove_len_vox = 0;
                int   n_hbond = 0, n_hydro = 0;

                for (int i0 = 0; i0 < d0; i0++) {
                    /* Map axis indices to grid x,y,z */
                    int ix, iy, iz;
                    if (axis == 0) { ix = i0; iy = i1; iz = i2; }
                    else if (axis == 1) { ix = i2; iy = i0; iz = i1; }
                    else { ix = i1; iy = i2; iz = i0; }

                    size_t idx_g = (size_t)(ix * dy * dz + iy * dz + iz) * nc;
                    float steric = grid->data[idx_g + CH_STERIC_DEMAND];

                    /* Void voxel in surface region */
                    if (steric > 0.4f && steric < 0.9f) {
                        if (groove_start < 0) groove_start = i0;
                        /* Affinity: sum of field channels */
                        float aff = 0.0f;
                        for (int c = 1; c < nc; c++) aff += grid->data[idx_g + c];
                        groove_sum_aff += aff;
                        groove_len_vox++;
                        /* Count H-bond sites */
                        if (grid->data[idx_g + CH_HBA_DEMAND] > 0.3f ||
                            grid->data[idx_g + CH_HBD_DEMAND] > 0.3f) n_hbond++;
                        /* Count hydrophobic patches */
                        if (grid->data[idx_g + CH_LIPO_DEMAND] > 0.5f) n_hydro++;
                    } else if (groove_start >= 0) {
                        /* End of void run — check if it qualifies as groove */
                        float length_A = groove_len_vox * res;
                        if (length_A >= min_len && n_grooves < PEP_MAX_GROOVES) {
                            PEP_Groove *g = &grooves[n_grooves++];
                            /* Entry point: start of groove */
                            int ex, ey, ez;
                            if (axis == 0) { ex = groove_start; ey = i1; ez = i2; }
                            else if (axis == 1) { ex = i2; ey = groove_start; ez = i1; }
                            else { ex = i1; ey = i2; ez = groove_start; }
                            g->entry_x = ex * res;
                            g->entry_y = ey * res;
                            g->entry_z = ez * res;
                            g->axis_x = (float)axes[axis][0];
                            g->axis_y = (float)axes[axis][1];
                            g->axis_z = (float)axes[axis][2];
                            g->length  = length_A;
                            g->width   = PEP_GROOVE_WIDTH;  /* approximate */
                            g->depth   = PEP_GROOVE_DEPTH;
                            g->affinity_score = groove_sum_aff / (float)groove_len_vox;
                            g->n_hbond_sites  = n_hbond;
                            g->n_hydrophobic_patches = n_hydro;
                        }
                        /* Reset */
                        groove_start = -1;
                        groove_sum_aff = 0.0f;
                        groove_len_vox = 0;
                        n_hbond = 0; n_hydro = 0;
                    }
                }
            }
        }
    }

    /* Sort grooves by affinity score descending (simple insertion sort) */
    for (int i = 1; i < n_grooves; i++) {
        PEP_Groove tmp = grooves[i];
        int j = i - 1;
        while (j >= 0 && grooves[j].affinity_score < tmp.affinity_score) {
            grooves[j+1] = grooves[j];
            j--;
        }
        grooves[j+1] = tmp;
    }

    return n_grooves;
}

/* -----------------------------------------------------------------------
 * nibble_peptide_hydrophobic_moment
 * ----------------------------------------------------------------------- */
float nibble_peptide_hydrophobic_moment(
    const PEP_Peptide *pep,
    PEP_SecondaryStructure ss)
{
    if (!pep || pep->n_residues == 0) return 0.0f;

    /* Angle per residue (radians) */
    float angle_per_res;
    switch (ss) {
        case PEP_SS_HELIX:  angle_per_res = 2.0f * 3.14159f / 3.6f; break;
        case PEP_SS_STRAND: angle_per_res = 2.0f * 3.14159f / 2.0f; break;
        default:            angle_per_res = 2.0f * 3.14159f / 3.6f; break;
    }

    float sum_x = 0.0f, sum_y = 0.0f;
    for (int i = 0; i < pep->n_residues; i++) {
        float h = pep->residues[i].hydrophobicity;
        float a = angle_per_res * i;
        sum_x += h * cosf(a);
        sum_y += h * sinf(a);
    }

    return sqrtf(sum_x*sum_x + sum_y*sum_y) / pep->n_residues;
}

/* -----------------------------------------------------------------------
 * nibble_peptide_ss_bias
 * ----------------------------------------------------------------------- */
void nibble_peptide_ss_bias(
    NibbleGrid        *grid,
    const PEP_Peptide *pep,
    const PEP_Groove  *groove,
    float              weight)
{
    if (!grid || !pep || weight <= 0) return;

    float res = grid->resolution;
    int dx = grid->dim_x, dy = grid->dim_y, dz = grid->dim_z;
    int nc = N_CHANNELS;

    /* Entry point and axis */
    float ex = groove ? groove->entry_x : (dx * res * 0.5f);
    float ey = groove ? groove->entry_y : (dy * res * 0.5f);
    float ez = groove ? groove->entry_z : (dz * res * 0.5f);
    float ax = groove ? groove->axis_x  : 0.0f;
    float ay = groove ? groove->axis_y  : 0.0f;
    float az = groove ? groove->axis_z  : 1.0f;

    /* Normalise axis */
    float anorm = sqrtf(ax*ax + ay*ay + az*az);
    if (anorm > 1e-6f) { ax /= anorm; ay /= anorm; az /= anorm; }

    float rise = (pep->dominant_ss == PEP_SS_HELIX)
                 ? PEP_RISE_PER_RES_HELIX : PEP_RISE_PER_RES_BETA;

    /* For each residue, add its demand at its expected position along groove */
    for (int i = 0; i < pep->n_residues; i++) {
        float t = i * rise;  /* distance along groove axis */
        float px = ex + ax * t;
        float py = ey + ay * t;
        float pz = ez + az * t;

        /* Grid indices */
        int ix = (int)(px / res);
        int iy = (int)(py / res);
        int iz = (int)(pz / res);
        if (ix < 0 || ix >= dx || iy < 0 || iy >= dy || iz < 0 || iz >= dz)
            continue;

        const PEP_Residue *r = &pep->residues[i];

        /* Helix: every ~3.6 residues, a hydrophobic face appears */
        float helix_phase = 0.0f;
        if (pep->dominant_ss == PEP_SS_HELIX) {
            float angle = 2.0f * 3.14159f * i / PEP_HELIX_PERIOD;
            helix_phase = (cosf(angle) + 1.0f) * 0.5f;  /* 0–1 */
        }

        /* Apply SS bias to a small neighbourhood */
        int radius_vox = 2;
        for (int di = -radius_vox; di <= radius_vox; di++) {
            for (int dj = -radius_vox; dj <= radius_vox; dj++) {
                for (int dk = -radius_vox; dk <= radius_vox; dk++) {
                    int xi = ix+di, yj = iy+dj, zk = iz+dk;
                    if (xi<0||xi>=dx||yj<0||yj>=dy||zk<0||zk>=dz) continue;
                    float d2 = (di*di + dj*dj + dk*dk) * res * res;
                    float gauss = expf(-d2 / (2.0f * res * res * 4));
                    size_t vidx = (size_t)(xi*dy*dz + yj*dz + zk) * nc;
                    /* Bias hydrophobic channel at hydrophobic residues */
                    if (r->hydrophobicity > 0.3f) {
                        grid->data[vidx + CH_LIPO_DEMAND] +=
                            weight * r->hydrophobicity * gauss * helix_phase;
                    }
                    /* Bias H-bond channels */
                    if (r->hbd_count > 0) {
                        grid->data[vidx + CH_HBD_DEMAND] +=
                            weight * 0.5f * gauss;
                    }
                    if (r->hba_count > 0) {
                        grid->data[vidx + CH_HBA_DEMAND] +=
                            weight * 0.5f * gauss;
                    }
                    /* Bias electrostatic channel */
                    grid->data[vidx + CH_ELEC_DEMAND] +=
                        weight * r->charge * 0.3f * gauss;
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * nibble_peptide_project
 * ----------------------------------------------------------------------- */
void nibble_peptide_project(
    NibbleGrid        *grid,
    const PEP_Peptide *pep,
    int                seg_idx,
    float              blur)
{
    if (!grid || !pep) return;
    if (blur <= 0) blur = 1.5f;

    float res = grid->resolution;
    float gx0 = pep->com_x - pep->length_angstrom * 0.5f;
    float gy0 = pep->com_y - pep->length_angstrom * 0.5f;
    float gz0 = pep->com_z - pep->length_angstrom * 0.5f;

    int s_start = (seg_idx >= 0) ? seg_idx : 0;
    int s_end   = (seg_idx >= 0) ? seg_idx : pep->n_segments - 1;

    for (int s = s_start; s <= s_end; s++) {
        const PEP_Segment *seg = &pep->segments[s];
        for (int r = seg->res_start; r <= seg->res_end; r++) {
            const PEP_Residue *res_r = &pep->residues[r];
            nibble_project_atom(
                grid,
                res_r->ca_x - gx0,
                res_r->ca_y - gy0,
                res_r->ca_z - gz0,
                blur,
                (float*)res_r->d
            );
        }
    }
}

/* -----------------------------------------------------------------------
 * nibble_peptide_trajectory
 * ----------------------------------------------------------------------- */
float nibble_peptide_trajectory(
    const NibbleGrid  *grid,
    PEP_Peptide       *pep,
    const PEP_Groove  *groove,
    int                n_steps,
    float              dt,
    float              gamma,
    float              kT,
    PEP_DockingResult *out)
{
    if (!grid || !pep || !out) return 0.0f;
    if (n_steps <= 0) n_steps = 500;
    if (dt    <= 0)   dt      = 0.002f;
    if (gamma <= 0)   gamma   = 0.5f;
    if (kT    <= 0)   kT      = 0.593f;

    unsigned int rng = 98765u;

    /* Place peptide along groove if given */
    if (groove) {
        float rise = (pep->dominant_ss == PEP_SS_HELIX)
                     ? PEP_RISE_PER_RES_HELIX : PEP_RISE_PER_RES_BETA;
        for (int i = 0; i < pep->n_residues; i++) {
            pep->residues[i].ca_x = groove->entry_x + groove->axis_x * rise * i;
            pep->residues[i].ca_y = groove->entry_y + groove->axis_y * rise * i;
            pep->residues[i].ca_z = groove->entry_z + groove->axis_z * rise * i;
        }
        /* Re-compute segment COMs */
        for (int s = 0; s < pep->n_segments; s++) {
            PEP_Segment *seg = &pep->segments[s];
            float sx = 0, sy = 0, sz = 0;
            int n_r = seg->res_end - seg->res_start + 1;
            for (int r = seg->res_start; r <= seg->res_end; r++) {
                sx += pep->residues[r].ca_x;
                sy += pep->residues[r].ca_y;
                sz += pep->residues[r].ca_z;
            }
            seg->cx = sx / n_r;
            seg->cy = sy / n_r;
            seg->cz = sz / n_r;
        }
        pep->com_x = groove->entry_x + groove->axis_x * pep->length_angstrom * 0.5f;
        pep->com_y = groove->entry_y + groove->axis_y * pep->length_angstrom * 0.5f;
        pep->com_z = groove->entry_z + groove->axis_z * pep->length_angstrom * 0.5f;
    }

    /* Scratch grid for scoring */
    NibbleGrid *scratch = nibble_create_grid(
        grid->dim_x, grid->dim_y, grid->dim_z, grid->resolution
    );
    if (!scratch) return 0.0f;

    /* Initial score */
    nibble_reset_steric(scratch);
    nibble_peptide_project(scratch, pep, -1, 1.5f);
    float best_score = nibble_compute_affinity_full(grid, scratch, 2.0f);
    best_score /= 2000.0f;
    best_score  = _clampf(best_score, 0.0f, 1.0f);

    /* Save best Cα positions */
    float best_ca[PEP_MAX_RESIDUES * 3];
    for (int i = 0; i < pep->n_residues; i++) {
        best_ca[i*3+0] = pep->residues[i].ca_x;
        best_ca[i*3+1] = pep->residues[i].ca_y;
        best_ca[i*3+2] = pep->residues[i].ca_z;
    }

    float exp_gdt   = expf(-gamma * dt);
    float noise_sc  = sqrtf(2.0f * gamma * kT * dt);
    float res_a     = grid->resolution;
    /* Backbone spring constant between adjacent segments */
    float K_bb      = 15.0f;  /* kcal/mol/Å² */

    for (int step = 0; step < n_steps; step++) {
        /* Update each segment */
        for (int s = 0; s < pep->n_segments; s++) {
            PEP_Segment *seg = &pep->segments[s];

            /* Gradient from protein field at segment COM */
            float gx, gy, gz;
            nibble_gradient(grid, seg->cx, seg->cy, seg->cz, &gx, &gy, &gz);

            /* Backbone spring forces from adjacent segments */
            float fx = gx, fy = gy, fz = gz;

            if (s > 0) {
                PEP_Segment *prev = &pep->segments[s-1];
                float dx = seg->cx - prev->cx;
                float dy = seg->cy - prev->cy;
                float dz = seg->cz - prev->cz;
                float d  = sqrtf(dx*dx + dy*dy + dz*dz);
                float L_eq = (seg->res_start - prev->res_start) *
                             (pep->dominant_ss == PEP_SS_HELIX ?
                              PEP_RISE_PER_RES_HELIX : PEP_RISE_PER_RES_BETA);
                if (d > 1e-6f) {
                    float f_spring = K_bb * (d - L_eq) / d;
                    fx -= f_spring * dx;
                    fy -= f_spring * dy;
                    fz -= f_spring * dz;
                }
            }
            if (s < pep->n_segments - 1) {
                PEP_Segment *next = &pep->segments[s+1];
                float dx = next->cx - seg->cx;
                float dy = next->cy - seg->cy;
                float dz = next->cz - seg->cz;
                float d  = sqrtf(dx*dx + dy*dy + dz*dz);
                float L_eq = (next->res_start - seg->res_start) *
                             (pep->dominant_ss == PEP_SS_HELIX ?
                              PEP_RISE_PER_RES_HELIX : PEP_RISE_PER_RES_BETA);
                if (d > 1e-6f) {
                    float f_spring = K_bb * (d - L_eq) / d;
                    fx += f_spring * dx;
                    fy += f_spring * dy;
                    fz += f_spring * dz;
                }
            }

            /* Constraint forces */
            for (int c = 0; c < pep->n_constraints; c++) {
                PEP_Constraint *con = &pep->constraints[c];
                /* Find which segment contains res_i and res_j */
                int is_i = (con->res_i >= seg->res_start && con->res_i <= seg->res_end);
                int is_j = (con->res_j >= seg->res_start && con->res_j <= seg->res_end);
                if (!is_i && !is_j) continue;

                PEP_Residue *ri = &pep->residues[con->res_i];
                PEP_Residue *rj = &pep->residues[con->res_j];
                float cdx = rj->ca_x - ri->ca_x;
                float cdy = rj->ca_y - ri->ca_y;
                float cdz = rj->ca_z - ri->ca_z;
                float cd  = sqrtf(cdx*cdx + cdy*cdy + cdz*cdz);
                if (cd > 1e-6f) {
                    float cf = con->spring_k * (cd - con->eq_distance) / cd;
                    if (is_i) { fx += cf * cdx; fy += cf * cdy; fz += cf * cdz; }
                    if (is_j) { fx -= cf * cdx; fy -= cf * cdy; fz -= cf * cdz; }
                }
            }

            /* Langevin integration */
            seg->vx = seg->vx * exp_gdt + fx * dt + noise_sc * _rng_normal(&rng);
            seg->vy = seg->vy * exp_gdt + fy * dt + noise_sc * _rng_normal(&rng);
            seg->vz = seg->vz * exp_gdt + fz * dt + noise_sc * _rng_normal(&rng);

            float new_cx = seg->cx + seg->vx * dt;
            float new_cy = seg->cy + seg->vy * dt;
            float new_cz = seg->cz + seg->vz * dt;

            /* Boundary clamp */
            float mx = (grid->dim_x - 2) * res_a;
            float my = (grid->dim_y - 2) * res_a;
            float mz = (grid->dim_z - 2) * res_a;
            seg->cx = _clampf(new_cx, 0, mx);
            seg->cy = _clampf(new_cy, 0, my);
            seg->cz = _clampf(new_cz, 0, mz);

            /* Move residues with segment */
            float ddx = seg->cx - (seg->cx - seg->vx * dt);
            float ddy = seg->cy - (seg->cy - seg->vy * dt);
            float ddz = seg->cz - (seg->cz - seg->vz * dt);
            for (int r = seg->res_start; r <= seg->res_end; r++) {
                pep->residues[r].ca_x += ddx;
                pep->residues[r].ca_y += ddy;
                pep->residues[r].ca_z += ddz;
            }
        }

        /* Score current pose */
        if (step % 5 == 0) {
            nibble_reset_steric(scratch);
            nibble_peptide_project(scratch, pep, -1, 1.5f);
            float raw = nibble_compute_affinity_full(grid, scratch, 2.0f);
            float sc  = _clampf(raw / 2000.0f, 0.0f, 1.0f);
            if (sc > best_score) {
                best_score = sc;
                for (int i = 0; i < pep->n_residues; i++) {
                    best_ca[i*3+0] = pep->residues[i].ca_x;
                    best_ca[i*3+1] = pep->residues[i].ca_y;
                    best_ca[i*3+2] = pep->residues[i].ca_z;
                }
            }
        }
    }

    nibble_free_grid(scratch);

    /* Restore best pose */
    for (int i = 0; i < pep->n_residues; i++) {
        pep->residues[i].ca_x = best_ca[i*3+0];
        pep->residues[i].ca_y = best_ca[i*3+1];
        pep->residues[i].ca_z = best_ca[i*3+2];
    }

    /* Fill result */
    memset(out, 0, sizeof(PEP_DockingResult));
    out->composite_score = best_score;
    out->affinity_score  = best_score;
    out->groove_index    = groove ? 0 : -1;
    for (int i = 0; i < pep->n_residues; i++) {
        out->ca_coords[i*3+0] = pep->residues[i].ca_x;
        out->ca_coords[i*3+1] = pep->residues[i].ca_y;
        out->ca_coords[i*3+2] = pep->residues[i].ca_z;
    }

    /* Estimate H-bonds: count residues near H-bond sites on protein */
    int n_hb = 0;
    for (int i = 0; i < pep->n_residues; i++) {
        if (pep->residues[i].hbd_count > 0 || pep->residues[i].hba_count > 0)
            n_hb++;
    }
    out->n_hbonds_formed = (float)n_hb * 0.4f;

    /* Amphipathic score */
    out->amphipathic_score = _clampf(pep->hydrophobic_moment * 2.0f, 0.0f, 1.0f);

    return best_score;
}

/* -----------------------------------------------------------------------
 * nibble_peptide_dock — full pipeline
 * ----------------------------------------------------------------------- */
float nibble_peptide_dock(
    const NibbleGrid *protein_grid,
    PEP_Peptide      *pep,
    float             delta_G_bulk,
    float             kT,
    int               n_steps,
    PEP_DockingResult *out)
{
    if (!protein_grid || !pep || !out) return 0.0f;
    if (kT <= 0) kT = 0.593f;
    if (delta_G_bulk <= 0) delta_G_bulk = 2.0f;
    if (n_steps <= 0) n_steps = 500;

    /* Step 1: Detect grooves */
    PEP_Groove grooves[PEP_MAX_GROOVES];
    int n_grooves = nibble_peptide_scan_grooves(
        protein_grid, grooves, PEP_GROOVE_MIN_LEN
    );

    /* Step 2: Apply SS bias to grid */
    NibbleGrid *biased_grid = nibble_create_grid(
        protein_grid->dim_x, protein_grid->dim_y, protein_grid->dim_z,
        protein_grid->resolution
    );
    if (!biased_grid) return 0.0f;

    /* Copy grid */
    size_t grid_size = (size_t)protein_grid->dim_x * protein_grid->dim_y *
                       protein_grid->dim_z * N_CHANNELS * sizeof(float);
    memcpy(biased_grid->data, protein_grid->data, grid_size);

    /* Apply water network then SS bias */
    nibble_compute_water_network(biased_grid, delta_G_bulk, kT);
    nibble_peptide_ss_bias(
        biased_grid, pep,
        (n_grooves > 0) ? &grooves[0] : NULL,
        0.2f
    );

    /* Step 3: Trial docking in each groove (up to 3) */
    PEP_DockingResult best_result;
    memset(&best_result, 0, sizeof(best_result));
    float best_score = -1.0f;
    int n_trials = n_grooves < 3 ? n_grooves : 3;
    if (n_trials == 0) n_trials = 1;  /* at least one trial even with no groove */

    for (int t = 0; t < n_trials; t++) {
        PEP_DockingResult trial_result;
        PEP_Groove *g = (n_grooves > 0) ? &grooves[t] : NULL;

        float sc = nibble_peptide_trajectory(
            biased_grid, pep, g, n_steps,
            0.002f, 0.5f, kT, &trial_result
        );
        trial_result.groove_index = t;

        /* SS bias score: how well the groove matches peptide SS */
        if (g) {
            float ss_match = (pep->dominant_ss == PEP_SS_HELIX)
                ? g->n_hydrophobic_patches * 0.1f
                : g->n_hbond_sites * 0.08f;
            trial_result.ss_bias_score    = _clampf(ss_match, 0.0f, 1.0f);
            trial_result.groove_match_score =
                _clampf(g->affinity_score / 5.0f, 0.0f, 1.0f);
        }

        /* Constraint penalty */
        float cpen = 0.0f;
        for (int c = 0; c < pep->n_constraints; c++) {
            PEP_Constraint *con = &pep->constraints[c];
            PEP_Residue *ri = &pep->residues[con->res_i];
            PEP_Residue *rj = &pep->residues[con->res_j];
            float d = sqrtf(_dist2(ri->ca_x, ri->ca_y, ri->ca_z,
                                   rj->ca_x, rj->ca_y, rj->ca_z));
            float dev = fabsf(d - con->eq_distance);
            cpen += dev / con->eq_distance;
        }
        cpen = _clampf(cpen / (pep->n_constraints + 1), 0.0f, 1.0f);
        trial_result.constraint_penalty = cpen;

        /* Composite */
        trial_result.composite_score = _clampf(
            0.50f * trial_result.affinity_score     +
            0.15f * trial_result.ss_bias_score      +
            0.15f * trial_result.groove_match_score +
            0.10f * trial_result.amphipathic_score  -
            0.10f * cpen,
            0.0f, 1.0f
        );

        if (trial_result.composite_score > best_score) {
            best_score  = trial_result.composite_score;
            best_result = trial_result;
        }
    }

    nibble_free_grid(biased_grid);
    *out = best_result;
    return best_score;
}

/* -----------------------------------------------------------------------
 * Utility printers
 * ----------------------------------------------------------------------- */
void nibble_peptide_print_result(const PEP_DockingResult *r) {
    if (!r) return;
    printf("[nibble_peptide] Docking Result\n");
    printf("  Composite score:      %.4f\n", r->composite_score);
    printf("  Affinity score:       %.4f\n", r->affinity_score);
    printf("  SS bias score:        %.4f\n", r->ss_bias_score);
    printf("  Groove match:         %.4f  (groove %d)\n",
           r->groove_match_score, r->groove_index);
    printf("  Amphipathic score:    %.4f\n", r->amphipathic_score);
    printf("  Constraint penalty:   %.4f\n", r->constraint_penalty);
    printf("  Estimated H-bonds:    %.1f\n", r->n_hbonds_formed);
}

void nibble_peptide_print(const PEP_Peptide *p) {
    if (!p) return;
    const char *ss_names[] = {"Coil", "Helix", "Strand", "Turn"};
    printf("[nibble_peptide] Peptide (%d residues)\n", p->n_residues);
    printf("  Dominant SS:   %s\n", ss_names[p->dominant_ss]);
    printf("  Net charge:    %.1f\n", p->net_charge);
    printf("  Length:        %.1f Å\n", p->length_angstrom);
    printf("  Hydrophobic moment: %.4f %s\n",
           p->hydrophobic_moment,
           p->hydrophobic_moment > 0.4f ? "(amphipathic)" : "");
    printf("  Cyclic:        %s\n", p->is_cyclic  ? "yes" : "no");
    printf("  Stapled:       %s\n", p->is_stapled ? "yes" : "no");
    printf("  Constraints:   %d\n", p->n_constraints);
    printf("  Segments:      %d\n", p->n_segments);
}
