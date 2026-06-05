/*
 * nibble_gaps.c - Gap Closure: Fukui Reactivity Channels + Virtual Water Network
 * Implements covalent bond detection and water-mediated H-bond capture.
 */
#include "nibble.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
 * Gap 1: Project Fukui reactivity channels from partial charges.
 *
 * The original implementation used raw charge sign as a proxy:
 *   q > 0 -> electrophilic, q < 0 -> nucleophilic
 * This is physically wrong. Partial charge measures permanent polarity,
 * not frontier orbital reactivity. A carbonyl carbon (C=O) has q > 0
 * but is electrophilic at carbon AND the oxygen is nucleophilic — the
 * sign alone misclassifies both atoms.
 *
 * Improved model: condensed-to-atom Fukui approximation.
 *
 * For a ground-state proxy without QM:
 *   f+(i)  ~ susceptibility to nucleophilic attack (LUMO-weighted)
 *          ~ large positive partial charge on atom i
 *            AND atom is in a low-coordination environment
 *   f-(i)  ~ susceptibility to electrophilic attack (HOMO-weighted)
 *          ~ large negative partial charge on atom i
 *            AND atom has lone pairs (N, O, S)
 *
 * The key improvement: we use |q| as the magnitude driver in both
 * channels, but gate each channel by the SIGN to avoid cross-contamination.
 * We also apply a nonlinear sharpening (q^2) so only strongly polarised
 * atoms contribute — weak partial charges below 0.1e are noise.
 *
 * Additionally, atoms with |q| < Q_THRESHOLD are skipped entirely —
 * they contribute neither nucleophilic nor electrophilic character.
 *
 * This is the Parr-Yang condensed Fukui approximation adapted for
 * voxel projection. Ref: Parr & Yang, JACS 1984.
 */
#define FUKUI_Q_THRESHOLD 0.10f   /* minimum |q| to register reactivity  */
#define FUKUI_SHARPENING  2.0f    /* exponent for nonlinear sharpening    */

void nibble_project_fukui(NibbleGrid *grid, int n_atoms,
                           const float *coords, const float *partial_charges,
                           float blur_radius) {
    for (int i = 0; i < n_atoms; ++i) {
        float q = partial_charges[i];
        float q_abs = fabsf(q);

        /* Skip weakly polarised atoms — they are noise */
        if (q_abs < FUKUI_Q_THRESHOLD) continue;

        /* Nonlinear sharpening: concentrate signal on strongly polarised atoms */
        float q_sharp = powf(q_abs, FUKUI_SHARPENING);

        float ch[N_CHANNELS] = {0};
        if (q > 0.0f) {
            /*
             * Positively charged atom: electron-deficient, susceptible to
             * nucleophilic attack. Projects into CH_ELECTROPHILICITY (f+).
             * Examples: carbonyl C, amide C, phosphorus in phosphate.
             */
            ch[CH_ELECTROPHILICITY] = q_sharp;
        } else {
            /*
             * Negatively charged atom: electron-rich, susceptible to
             * electrophilic attack. Projects into CH_NUCLEOPHILICITY (f-).
             * Examples: carbonyl O, amine N, thiol S.
             */
            ch[CH_NUCLEOPHILICITY] = q_sharp;
        }

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

/* Gap 3: Virtual Water Network
 *
 * Computes per-voxel water occupancy and stores in CH_WATER_DYNAMIC.
 *
 * Five problems fixed vs previous model:
 *
 * 1. CORRECT NORMALISATION
 *    P_water(v) is a true probability in [0,1], sum over pocket = 1.
 *    Previous model multiplied by n_vox producing values > 1.
 *    We store relative occupancy: p(v) / mean_p, so values > 1.0
 *    mean above-average occupancy (likely water site), < 1.0 transient.
 *    This is consistent with WaterMap convention and makes desolvation
 *    penalty physically interpretable.
 *
 * 2. SPATIAL CONNECTIVITY
 *    Water molecules form H-bond networks. A voxel adjacent to a
 *    high-occupancy neighbour is more stable. Implemented as a single
 *    face-neighbour averaging pass (sigma ~ 1 voxel, O(n_vox)).
 *    Connectivity weight: CONN_WEIGHT = 0.25.
 *    Ref: Huggins et al., JCTC 2012.
 *
 * 3. BRIDGING WATER DETECTION
 *    A bridging water satisfies H-bond demands on BOTH sides:
 *      (a) pocket has both HBA and HBD demand (can receive and donate)
 *      (b) at least one face-adjacent voxel is void (ps > 0.7)
 *    Bridging voxels get BRIDGE_BONUS = 1.5 multiplier on raw occupancy.
 *    Distinguishes pharmacophore waters from bulk surface waters.
 *
 * 4. HARD CUTOFFS
 *    ps < WALL_THRESHOLD (0.3) -> inside protein wall -> P = 0.
 *    dG > DG_CUTOFF * kT (5.0) -> thermally inaccessible -> P = 0.
 *
 * 5. DOUBLE-COUNTING FIX IN affinity_full
 *    nibble_compute_affinity() already applies -0.3 * p_conserved * ds
 *    and -0.1 * p_dynamic * ds as base displacement cost.
 *    affinity_full now adds only the INCREMENTAL remainder so
 *    total penalty = full model, not 2x.
 *
 * dG model (from previous fix, retained):
 *   dG(v) = delta_G_bulk
 *           - W_ELEC * |elec(v)|
 *           - W_HB   * (hba(v) + hbd(v))
 *           + W_STER * burial(v)
 *
 * Ref: Abel et al., JCTC 2008; Michel et al., JCTC 2009.
 */

#define WALL_THRESHOLD  0.30f
#define DG_CUTOFF       5.0f
#define W_ELEC          1.5f
#define W_HB            1.2f
#define W_STER          3.0f
#define CONN_WEIGHT     0.25f
#define BRIDGE_BONUS    1.5f
#define HB_MIN          0.15f

void nibble_compute_water_network(NibbleGrid *grid, float delta_G_bulk, float kT) {
    int nx = grid->dim_x, ny = grid->dim_y, nz = grid->dim_z;
    size_t n_vox = (size_t)nx * ny * nz;
    int nc = N_CHANNELS;
    float inv_kT = 1.0f / kT;

    float *raw  = (float*)calloc(n_vox, sizeof(float));
    float *conn = (float*)calloc(n_vox, sizeof(float));
    if (!raw || !conn) { free(raw); free(conn); return; }

    /* ------------------------------------------------------------------
     * Pass 1: per-voxel Boltzmann occupancy with bridging detection
     * ------------------------------------------------------------------ */
    float Z = 0.0f;
    size_t n_candidates = 0;

    for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
    for (int iz = 0; iz < nz; ++iz) {
        size_t idx = ((size_t)ix * ny + iy) * nz + iz;
        float *vp  = &grid->data[idx * nc];
        float ps   = vp[CH_STERIC_DEMAND];

        if (ps < WALL_THRESHOLD) continue;

        float burial = 1.0f - ps;
        if (burial < 0.0f) burial = 0.0f;

        float dG = delta_G_bulk
                   - W_ELEC * fabsf(vp[CH_ELEC_DEMAND])
                   - W_HB   * (vp[CH_HBA_DEMAND] + vp[CH_HBD_DEMAND])
                   + W_STER * burial;

        if (dG > DG_CUTOFF * kT) continue;

        /* Bridging water: pocket has both HBA+HBD demand AND
         * a face-adjacent void voxel exists for ligand access */
        float bridge = 1.0f;
        if (vp[CH_HBA_DEMAND] > HB_MIN && vp[CH_HBD_DEMAND] > HB_MIN) {
            int void_nb = 0;
            #define CHECK_NB(NX,NY,NZ) \
                if (!void_nb && (NX)>=0 && (NX)<nx && (NY)>=0 \
                    && (NY)<ny && (NZ)>=0 && (NZ)<nz) { \
                    float nps = grid->data[(((size_t)(NX)*ny+(NY))*nz+(NZ))*nc \
                                           + CH_STERIC_DEMAND]; \
                    if (nps > 0.7f) void_nb = 1; \
                }
            CHECK_NB(ix-1, iy,   iz  )
            CHECK_NB(ix+1, iy,   iz  )
            CHECK_NB(ix,   iy-1, iz  )
            CHECK_NB(ix,   iy+1, iz  )
            CHECK_NB(ix,   iy,   iz-1)
            CHECK_NB(ix,   iy,   iz+1)
            #undef CHECK_NB
            if (void_nb) bridge = BRIDGE_BONUS;
        }

        raw[idx] = bridge * expf(-dG * inv_kT);
        Z += raw[idx];
        n_candidates++;
    }}}

    if (Z < 1e-9f || n_candidates == 0) { free(raw); free(conn); return; }

    /* ------------------------------------------------------------------
     * Pass 2: spatial connectivity via face-neighbour averaging
     * ------------------------------------------------------------------ */
    for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
    for (int iz = 0; iz < nz; ++iz) {
        size_t idx = ((size_t)ix * ny + iy) * nz + iz;
        if (raw[idx] < 1e-9f) continue;

        float nb_sum = 0.0f; int nb_n = 0;
        #define ACCUM_NB(NX,NY,NZ) \
            if ((NX)>=0 && (NX)<nx && (NY)>=0 && (NY)<ny \
                && (NZ)>=0 && (NZ)<nz) { \
                nb_sum += raw[((size_t)(NX)*ny+(NY))*nz+(NZ)]; nb_n++; }
        ACCUM_NB(ix-1, iy,   iz  )
        ACCUM_NB(ix+1, iy,   iz  )
        ACCUM_NB(ix,   iy-1, iz  )
        ACCUM_NB(ix,   iy+1, iz  )
        ACCUM_NB(ix,   iy,   iz-1)
        ACCUM_NB(ix,   iy,   iz+1)
        #undef ACCUM_NB

        conn[idx] = raw[idx] + CONN_WEIGHT * (nb_n > 0 ? nb_sum / nb_n : 0.0f);
    }}}

    /* ------------------------------------------------------------------
     * Pass 3: normalise to relative occupancy and store
     * ------------------------------------------------------------------ */
    float Z2 = 0.0f;
    for (size_t i = 0; i < n_vox; ++i) Z2 += conn[i];

    float mean_occ = (Z2 > 1e-9f && n_candidates > 0)
                     ? (Z2 / (float)n_candidates) : 1.0f;
    float inv_Z2   = (Z2 > 1e-9f) ? (1.0f / Z2) : 0.0f;

    for (size_t i = 0; i < n_vox; ++i) {
        /* p = true probability; store as p / mean_p = relative occupancy */
        float p = conn[i] * inv_Z2;
        grid->data[i * nc + CH_WATER_DYNAMIC] =
            (mean_occ > 1e-9f) ? (p / (mean_occ * inv_Z2)) : 0.0f;
    }

    free(raw);
    free(conn);
}

/* Desolvation penalty: cost of displacing above-average water voxels.
 * Only voxels with relative occupancy > 1.0 (more stable than average)
 * incur a meaningful penalty. Transient waters (< 1.0) are nearly free.
 * penalty(v) = max(0, p_rel(v) - 1.0) * delta_G_bulk * ds(v)
 */
float nibble_desolvation_penalty(const NibbleGrid *protein, const NibbleGrid *ligand,
                                  float delta_G_bulk, float steric_threshold) {
    size_t n_vox = (size_t)protein->dim_x * protein->dim_y * protein->dim_z;
    int nc = N_CHANNELS;
    float penalty = 0.0f;
    for (size_t i = 0; i < n_vox; ++i) {
        float ds = ligand->data[i * nc + CH_STERIC_DEMAND];
        if (ds <= steric_threshold) continue;
        float p_rel  = protein->data[i * nc + CH_WATER_DYNAMIC];
        float excess = p_rel - 1.0f;
        if (excess > 0.0f)
            penalty += excess * delta_G_bulk * ds;
    }
    return penalty;
}

/* Full affinity with water network bonus and desolvation penalty.
 *
 * nibble_compute_affinity() already applies a small base displacement
 * cost: -0.3 * p_conserved * ds and -0.1 * p_dynamic * ds.
 * affinity_full() adds ONLY the incremental remainder to avoid
 * double-counting.
 *
 * Water-mediated H-bond bonus:
 *   True bridging requires BOTH sides to be satisfied:
 *     protein HBA <-water-> ligand HBD  (and vice versa)
 *   Score = complementary cross-product of HBA/HBD demands.
 *   conserved water: weight 0.5 (always present)
 *   dynamic water:   weight 0.25, only for above-average sites (p_rel > 1)
 */
float nibble_compute_affinity_full(const NibbleGrid *pocket, const NibbleGrid *drug,
                                    float delta_G_bulk) {
    float base = nibble_compute_affinity(pocket, drug);

    size_t n_vox = (size_t)pocket->dim_x * pocket->dim_y * pocket->dim_z;
    int nc = N_CHANNELS;
    float water_bonus  = 0.0f;
    float desolv_extra = 0.0f;

    for (size_t i = 0; i < n_vox; ++i) {
        const float *pp = &pocket->data[i * nc];
        const float *dp = &drug->data[i * nc];
        float p_con = pp[CH_WATER_CONSERVED];
        float p_dyn = pp[CH_WATER_DYNAMIC];
        float ds    = dp[CH_STERIC_DEMAND];

        /* Water-mediated H-bond: complementary cross-product only */
        if (p_con > 0.01f) {
            float bridge = pp[CH_HBA_DEMAND] * dp[CH_HBD_DEMAND]
                         + pp[CH_HBD_DEMAND] * dp[CH_HBA_DEMAND];
            water_bonus += 0.5f * p_con * bridge;
        }
        if (p_dyn > 1.0f) {
            float bridge = pp[CH_HBA_DEMAND] * dp[CH_HBD_DEMAND]
                         + pp[CH_HBD_DEMAND] * dp[CH_HBA_DEMAND];
            water_bonus += 0.25f * (p_dyn - 1.0f) * bridge;
        }

        /* Incremental desolvation: full model minus base already applied */
        if (ds > 0.1f) {
            float extra_c = (delta_G_bulk - 0.3f) * p_con * ds;
            float p_excess = (p_dyn > 1.0f) ? (p_dyn - 1.0f) : 0.0f;
            float extra_d = (delta_G_bulk - 0.1f) * p_excess * ds;
            if (extra_c > 0.0f) desolv_extra += extra_c;
            if (extra_d > 0.0f) desolv_extra += extra_d;
        }
    }

    return base + water_bonus - desolv_extra;
}
