/*
 * nibble_protac.c
 * PROTAC Ternary Complex Extension for the Nibble Voxel Engine.
 *
 * Implements:
 *   nibble_protac_create()        — bifunctional molecule decomposition
 *   nibble_protac_free()          — cleanup
 *   nibble_protac_linker_penalty()— Gaussian chain geometry feasibility
 *   nibble_protac_cooperativity() — Hook et al. alpha parameter
 *   nibble_protac_neo_ppi()       — induced ternary interface scoring
 *   nibble_protac_score()         — full ternary composite score
 *   nibble_protac_trajectory()    — dumbbell Langevin optimisation
 *   nibble_protac_infer_linker()  — linker property inference
 *   nibble_protac_print_score()   — diagnostic output
 *
 * Zero modifications to any existing Nibble file.
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#include "nibble_protac.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Internal constants
 * ----------------------------------------------------------------------- */

/* Composite score component weights (sum = 1.0) */
#define _WP_A          0.30f   /* warhead A affinity                        */
#define _WP_B          0.30f   /* warhead B affinity                        */
#define _WP_LINKER     0.15f   /* linker geometry feasibility               */
#define _WP_COOP       0.15f   /* cooperativity                             */
#define _WP_NEO        0.10f   /* neo-PPI surface                           */

/* Hook effect penalty */
#define _HOOK_PENALTY  0.30f   /* subtract from composite if Hook predicted */

/* Degradation efficiency weights */
#define _WD_TERNARY    0.50f   /* ternary score contribution                */
#define _WD_ALPHA      0.30f   /* cooperativity contribution                */
#define _WD_NEO        0.20f   /* neo-PPI contribution                      */

/* Thermal noise seed */
#define _PROTAC_RNG_SEED 42317u

/* -----------------------------------------------------------------------
 * Internal: xorshift RNG (same as nibble_tier3.c — standalone copy)
 * ----------------------------------------------------------------------- */
static float _rng_float(unsigned int *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return ((float)(*s & 0x7FFFFFFF)) / (float)0x7FFFFFFF;
}
static float _rng_normal(unsigned int *s) {
    float u1 = _rng_float(s) + 1e-8f;
    float u2 = _rng_float(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530f * u2);
}

static inline float _clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* -----------------------------------------------------------------------
 * nibble_protac_create
 * ----------------------------------------------------------------------- */
PROTAC_Molecule *nibble_protac_create(
    int          n_atoms_total,
    const float *coords,
    const float *ch_vals,
    int          split_a,
    int          split_b)
{
    if (!coords || !ch_vals) return NULL;
    if (split_a <= 0 || split_b <= split_a || split_b >= n_atoms_total)
        return NULL;

    PROTAC_Molecule *p = (PROTAC_Molecule*)calloc(1, sizeof(PROTAC_Molecule));
    if (!p) return NULL;

    int n_a      = split_a;
    int n_linker = split_b - split_a;
    int n_b      = n_atoms_total - split_b;

    /* ---- Warhead A ---- */
    p->warhead_a = nibble_mol_create(
        n_a,
        coords,
        ch_vals
    );
    if (!p->warhead_a) { free(p); return NULL; }

    /* ---- Linker ---- */
    int linker_atoms = n_linker < PROTAC_MAX_LINKER_ATOMS
                       ? n_linker : PROTAC_MAX_LINKER_ATOMS;
    p->linker.n_atoms = linker_atoms;
    memcpy(p->linker.coords,
           coords + split_a * 3,
           linker_atoms * 3 * sizeof(float));
    memcpy(p->linker.ch_vals,
           ch_vals + split_a * N_CHANNELS,
           linker_atoms * N_CHANNELS * sizeof(float));
    nibble_protac_infer_linker(&p->linker, linker_atoms, NULL);

    /* ---- Warhead B ---- */
    p->warhead_b = nibble_mol_create(
        n_b,
        coords + split_b * 3,
        ch_vals + split_b * N_CHANNELS
    );
    if (!p->warhead_b) {
        nibble_mol_free(p->warhead_a);
        free(p);
        return NULL;
    }

    /* Initial separation: from warhead_a COM to warhead_b COM */
    /* Both are in local frame initially — separation starts at linker length */
    p->sep_x = p->linker.contour_length;
    p->sep_y = 0.0f;
    p->sep_z = 0.0f;
    p->separation = p->linker.contour_length;

    return p;
}

/* -----------------------------------------------------------------------
 * nibble_protac_free
 * ----------------------------------------------------------------------- */
void nibble_protac_free(PROTAC_Molecule *p) {
    if (!p) return;
    if (p->warhead_a) nibble_mol_free(p->warhead_a);
    if (p->warhead_b) nibble_mol_free(p->warhead_b);
    free(p);
}

/* -----------------------------------------------------------------------
 * nibble_protac_infer_linker
 * ----------------------------------------------------------------------- */
void nibble_protac_infer_linker(
    PROTAC_Linker *linker,
    int            n_atoms,
    const char    *smiles)
{
    if (!linker) return;
    linker->n_atoms = n_atoms;

    /* Contour length: n_atoms × average C-C bond (1.54Å) + some for angles */
    linker->contour_length = n_atoms * 1.54f * 1.1f;  /* 10% stretch factor */

    /* Persistence length and flexibility from simple heuristics */
    if (smiles) {
        /* Count aromatic indicators for rigidity */
        int n_arom = 0, n_alkyl = 0;
        for (const char *c = smiles; *c; c++) {
            if (*c == 'c' || *c == 'n' || *c == 'o') n_arom++;
            if (*c == 'C' || *c == 'N' || *c == 'O') n_alkyl++;
        }
        float frac_rigid = (float)n_arom / (float)(n_arom + n_alkyl + 1);
        linker->flexibility       = 1.0f - frac_rigid;
        linker->persistence_length = 3.8f + frac_rigid * 20.0f; /* 4–24Å */
    } else {
        /* Default: assume PEG-like flexible linker */
        linker->flexibility        = 0.8f;
        linker->persistence_length = 4.0f;  /* Å */
    }
}

/* -----------------------------------------------------------------------
 * nibble_protac_linker_penalty
 *
 * Gaussian chain model (FJC):
 *   <r²> = n * b²   where n = effective Kuhn segments, b = Kuhn length
 *   P(r) ∝ exp(-3r²/(2<r²>))
 *
 * For a worm-like chain (WLC) with persistence length Lp:
 *   <r²> = 2*Lp*L * (1 - Lp/L * (1 - exp(-L/Lp)))
 *
 * Penalty = 1 - P(required_ete) / P(0) normalised to [0,1]
 * ----------------------------------------------------------------------- */
float nibble_protac_linker_penalty(
    const PROTAC_Linker *linker,
    float                required_ete,
    float               *linker_ete_out)
{
    if (!linker) return 1.0f;

    float L  = linker->contour_length;
    float Lp = linker->persistence_length;
    float b  = PROTAC_KUHN_LENGTH;

    if (L < 1e-6f) return 1.0f;

    /* WLC mean-square end-to-end distance */
    float r2_wlc;
    if (Lp < 1e-6f) {
        r2_wlc = L * b;   /* fully flexible fallback */
    } else {
        float x = L / Lp;
        r2_wlc = 2.0f * Lp * L * (1.0f - (Lp / L) * (1.0f - expf(-x)));
    }

    float r_rms = sqrtf(r2_wlc);
    if (linker_ete_out) *linker_ete_out = r_rms;

    /* Maximum physically reachable end-to-end distance */
    float r_max = L * linker->flexibility + L * (1.0f - linker->flexibility) * 0.5f;

    /* If required distance exceeds maximum: impossible */
    if (required_ete > r_max) return 1.0f;

    /* Gaussian chain probability at required_ete */
    if (r2_wlc < 1e-6f) return 0.5f;
    float exponent = -3.0f * required_ete * required_ete / (2.0f * r2_wlc);
    float log_p = exponent;  /* ln(P(r)) ignoring normalisation constant */

    /* P at r=0 (mode for Gaussian) vs P at r=required_ete */
    /* penalty = 1 - exp(log_p - log_p_max) where log_p_max = 0 at r→0 */
    float penalty = 1.0f - expf(log_p);
    return _clampf(penalty, 0.0f, 1.0f);
}

/* -----------------------------------------------------------------------
 * nibble_protac_cooperativity
 *
 * Hook et al. (2015) model:
 *   Kd_ternary_A = Kd_binary_A / alpha
 *   Kd_ternary_B = Kd_binary_B / alpha
 *
 * We approximate alpha from the sum of induced binding energy components.
 * delta_G_induced ≈ -kT * ln(score_a * score_b * (1 + neo_ppi))
 * alpha = exp(-delta_G_induced / kT)
 *
 * Hook effect: when PROTAC concentration is high, both warheads compete
 * for separate copies of target and E3, reducing ternary complex.
 * We model this as negative cooperativity when individual warhead scores
 * are both very high (> 0.85) — indicating concentration-dependent hook.
 * ----------------------------------------------------------------------- */
float nibble_protac_cooperativity(
    float score_a,
    float score_b,
    float neo_ppi,
    float kT)
{
    if (kT < 1e-6f) kT = 0.593f;

    score_a = _clampf(score_a, 1e-6f, 1.0f);
    score_b = _clampf(score_b, 1e-6f, 1.0f);
    neo_ppi = _clampf(neo_ppi, 0.0f,  1.0f);

    /* Induced binding energy from ternary complex formation */
    /* Higher individual scores + neo-PPI = more cooperative */
    float delta_G_induced = -kT * logf(score_a * score_b * (1.0f + neo_ppi));
    float alpha = expf(-delta_G_induced / kT);

    /* Hook effect: both warheads very high affinity → anticooperative */
    if (score_a > 0.85f && score_b > 0.85f) {
        /* Binary-to-ternary competition: reduce alpha */
        float hook_factor = (score_a - 0.85f) * (score_b - 0.85f) * 10.0f;
        alpha *= expf(-hook_factor);
    }

    return _clampf(alpha, 0.01f, PROTAC_ALPHA_MAX);
}

/* -----------------------------------------------------------------------
 * nibble_protac_neo_ppi
 *
 * The PROTAC's linker creates a proximity mask between the two protein
 * surfaces. Where the linker atoms are, the two proteins are forced into
 * contact. We score this induced interface using the existing PPI machinery.
 *
 * Algorithm:
 *   1. Find grid voxels near the linker centroid (within contour_length/2)
 *   2. Create a proximity mask: 1 inside linker region, 0 outside
 *   3. Run nibble_ppi_affinity() on the masked grid region
 *   4. Return the masked PPI score as neo-PPI contribution
 * ----------------------------------------------------------------------- */
float nibble_protac_neo_ppi(
    const NibbleGrid      *grid_target,
    const NibbleGrid      *grid_e3,
    const PROTAC_Molecule *protac,
    PPI_Interface         *iface,
    float                  kT)
{
    if (!grid_target || !grid_e3 || !protac) return 0.0f;

    /*
     * Simplified approach: the neo-PPI score is estimated from the
     * overlap integral of the two surface grids within the linker volume.
     *
     * Linker centroid: midpoint between warhead_a and warhead_b COMs
     */
    float lcx = (protac->warhead_a->cx + protac->warhead_b->cx) * 0.5f;
    float lcy = (protac->warhead_a->cy + protac->warhead_b->cy) * 0.5f;
    float lcz = (protac->warhead_a->cz + protac->warhead_b->cz) * 0.5f;
    float mask_r = protac->linker.contour_length * 0.6f;  /* generous radius */
    float mask_r2 = mask_r * mask_r;

    float res = grid_target->resolution;
    int dx = grid_target->dim_x;
    int dy = grid_target->dim_y;
    int dz = grid_target->dim_z;
    int nc = N_CHANNELS;

    double neo_sum = 0.0;
    int    neo_count = 0;

    for (int x = 0; x < dx; x++) {
        float vx = x * res;
        float d2x = (vx - lcx) * (vx - lcx);
        if (d2x > mask_r2) continue;
        for (int y = 0; y < dy; y++) {
            float vy = y * res;
            float d2xy = d2x + (vy - lcy) * (vy - lcy);
            if (d2xy > mask_r2) continue;
            for (int z = 0; z < dz; z++) {
                float vz = z * res;
                float d2 = d2xy + (vz - lcz) * (vz - lcz);
                if (d2 > mask_r2) continue;

                /* Check bounds for grid_e3 (may have different dims) */
                if (x >= grid_e3->dim_x || y >= grid_e3->dim_y || z >= grid_e3->dim_z)
                    continue;

                size_t idx_t = (size_t)(x * dy * dz + y * dz + z) * nc;
                size_t idx_e = (size_t)(x * grid_e3->dim_y * grid_e3->dim_z +
                                         y * grid_e3->dim_z + z) * nc;

                /* Proximity-gated inner product of field channels */
                float proximity = expf(-d2 / (mask_r2 * 0.5f));
                double ch_sum = 0.0;
                for (int c = 1; c < nc; c++) {  /* skip steric ch 0 */
                    ch_sum += (double)(grid_target->data[idx_t + c] *
                                       grid_e3->data[idx_e + c]);
                }
                neo_sum   += proximity * ch_sum;
                neo_count++;
            }
        }
    }

    if (neo_count == 0) return 0.0f;

    float neo_raw = (float)(neo_sum / neo_count);
    /* Normalise: typical inner product per voxel ~1–10, scale to [0,1] */
    float neo_score = _clampf(neo_raw / 5.0f, 0.0f, 1.0f);

    /* Store in iface if provided */
    if (iface) {
        iface->sc_score          = neo_score;
        iface->mean_hotspot_weight = neo_score;
    }

    return neo_score;
}

/* -----------------------------------------------------------------------
 * nibble_protac_score — full ternary complex scoring
 * ----------------------------------------------------------------------- */
float nibble_protac_score(
    const NibbleGrid       *grid_target,
    const NibbleGrid       *grid_e3,
    PROTAC_Molecule        *protac,
    PROTAC_TernaryGeometry *geometry,
    float                   delta_G_bulk,
    float                   kT,
    PROTAC_TernaryScore    *out)
{
    if (!grid_target || !grid_e3 || !protac || !geometry || !out) return 0.0f;
    if (kT < 1e-6f) kT = 0.593f;
    if (delta_G_bulk < 0.1f) delta_G_bulk = 2.0f;

    memset(out, 0, sizeof(PROTAC_TernaryScore));
    out->w_a      = _WP_A;
    out->w_b      = _WP_B;
    out->w_linker = _WP_LINKER;
    out->w_coop   = _WP_COOP;
    out->w_neo_ppi= _WP_NEO;

    /* --- 1. Warhead A vs target grid --- */
    /*
     * Project warhead_a into a scratch drug grid and score against
     * grid_target using nibble_compute_affinity_full (existing function).
     */
    NibbleGrid *scratch_a = nibble_create_grid(
        grid_target->dim_x, grid_target->dim_y, grid_target->dim_z,
        grid_target->resolution
    );
    if (!scratch_a) return 0.0f;
    nibble_mol_project(protac->warhead_a, scratch_a);
    nibble_compute_water_network(scratch_a, delta_G_bulk, kT);
    float raw_a = nibble_compute_affinity_full(grid_target, scratch_a, delta_G_bulk);
    /* Normalise: typical raw affinity range 0–2000; clamp to [0,1] */
    out->score_warhead_a = _clampf(raw_a / 2000.0f, 0.0f, 1.0f);
    nibble_free_grid(scratch_a);

    /* --- 2. Warhead B vs E3 grid --- */
    NibbleGrid *scratch_b = nibble_create_grid(
        grid_e3->dim_x, grid_e3->dim_y, grid_e3->dim_z,
        grid_e3->resolution
    );
    if (!scratch_b) return 0.0f;
    nibble_mol_project(protac->warhead_b, scratch_b);
    nibble_compute_water_network(scratch_b, delta_G_bulk, kT);
    float raw_b = nibble_compute_affinity_full(grid_e3, scratch_b, delta_G_bulk);
    out->score_warhead_b = _clampf(raw_b / 2000.0f, 0.0f, 1.0f);
    nibble_free_grid(scratch_b);

    /* --- 3. Linker geometry --- */
    float linker_ete = 0.0f;
    out->linker_penalty = nibble_protac_linker_penalty(
        &protac->linker,
        geometry->required_linker_ete,
        &linker_ete
    );
    out->linker_end_to_end = linker_ete;
    out->linker_feasibility = 1.0f - out->linker_penalty;

    /* --- 4. Neo-PPI score --- */
    PPI_Interface neo_iface;
    memset(&neo_iface, 0, sizeof(neo_iface));
    out->neo_ppi_score = nibble_protac_neo_ppi(
        grid_target, grid_e3, protac, &neo_iface, kT
    );

    /* --- 5. Cooperativity --- */
    out->alpha = nibble_protac_cooperativity(
        out->score_warhead_a,
        out->score_warhead_b,
        out->neo_ppi_score,
        kT
    );
    out->hook_effect = (out->alpha < PROTAC_ALPHA_NEUTRAL * 0.5f) ? 1 : 0;

    /* Normalised cooperativity contribution (0–1) */
    float norm_coop = _clampf(
        logf(out->alpha + 1e-6f) / logf(PROTAC_ALPHA_MAX),
        0.0f, 1.0f
    );

    /* --- 6. Composite ternary score --- */
    float composite =
        _WP_A      * out->score_warhead_a  +
        _WP_B      * out->score_warhead_b  +
        _WP_LINKER * out->linker_feasibility +
        _WP_COOP   * norm_coop             +
        _WP_NEO    * out->neo_ppi_score;

    /* Hook effect penalty */
    if (out->hook_effect) {
        composite -= _HOOK_PENALTY;
    }

    out->composite_score = _clampf(composite, 0.0f, 1.0f);

    /* --- 7. Degradation efficiency proxy --- */
    /*
     * Degradation requires:
     *   (a) Stable ternary complex  → composite score
     *   (b) Positive cooperativity  → alpha contribution
     *   (c) Strong neo-PPI          → induced interface
     *
     * deg_efficiency also depends on Kd_ubiquitination (not modelled here)
     * and proteasome accessibility (not modelled here) — so this is a proxy.
     */
    float alpha_contrib = _clampf(
        (out->alpha - 1.0f) / (PROTAC_ALPHA_MAX - 1.0f),
        0.0f, 1.0f
    );
    out->deg_efficiency = _clampf(
        _WD_TERNARY * out->composite_score +
        _WD_ALPHA   * alpha_contrib         +
        _WD_NEO     * out->neo_ppi_score,
        0.0f, 1.0f
    );

    return out->composite_score;
}

/* -----------------------------------------------------------------------
 * nibble_protac_trajectory — dumbbell Langevin optimisation
 * ----------------------------------------------------------------------- */
float nibble_protac_trajectory(
    const NibbleGrid       *grid_target,
    const NibbleGrid       *grid_e3,
    PROTAC_Molecule        *protac,
    PROTAC_TernaryGeometry *geometry,
    int                     n_steps,
    float                   dt,
    float                   gamma,
    float                   kT,
    PROTAC_TernaryScore    *out)
{
    if (!grid_target || !grid_e3 || !protac || !geometry || !out) return 0.0f;
    if (n_steps <= 0) n_steps = PROTAC_MAX_TRAJ_STEPS;
    if (dt    <= 0.0f) dt     = 0.002f;
    if (gamma <= 0.0f) gamma  = 1.0f;
    if (kT    <= 0.0f) kT     = 0.593f;

    unsigned int rng = _PROTAC_RNG_SEED;

    /* Spring equilibrium: linker contour length */
    float L_eq = protac->linker.contour_length;
    float K    = PROTAC_SPRING_K;

    /* Initial score */
    PROTAC_TernaryScore best_score;
    nibble_protac_score(grid_target, grid_e3, protac, geometry,
                        2.0f, kT, &best_score);
    float best_composite = best_score.composite_score;

    /* Save best warhead positions */
    float best_ax = protac->warhead_a->cx;
    float best_ay = protac->warhead_a->cy;
    float best_az = protac->warhead_a->cz;
    float best_bx = protac->warhead_b->cx;
    float best_by = protac->warhead_b->cy;
    float best_bz = protac->warhead_b->cz;

    /* Pre-allocate scratch grids for per-step scoring */
    NibbleGrid *scratch_a = nibble_create_grid(
        grid_target->dim_x, grid_target->dim_y, grid_target->dim_z,
        grid_target->resolution
    );
    NibbleGrid *scratch_b = nibble_create_grid(
        grid_e3->dim_x, grid_e3->dim_y, grid_e3->dim_z,
        grid_e3->resolution
    );
    if (!scratch_a || !scratch_b) {
        if (scratch_a) nibble_free_grid(scratch_a);
        if (scratch_b) nibble_free_grid(scratch_b);
        *out = best_score;
        return best_composite;
    }

    float exp_gdt   = expf(-gamma * dt);
    float noise_sc  = sqrtf(2.0f * gamma * kT * dt);

    /* Warhead velocities (start at zero) */
    float vax = 0.0f, vay = 0.0f, vaz = 0.0f;
    float vbx = 0.0f, vby = 0.0f, vbz = 0.0f;

    for (int step = 0; step < n_steps; step++) {

        /* ---- Gradient forces on each warhead from their pocket grids ---- */
        float gax, gay, gaz, gbx, gby, gbz;
        nibble_gradient(grid_target,
                        protac->warhead_a->cx,
                        protac->warhead_a->cy,
                        protac->warhead_a->cz,
                        &gax, &gay, &gaz);
        nibble_gradient(grid_e3,
                        protac->warhead_b->cx,
                        protac->warhead_b->cy,
                        protac->warhead_b->cz,
                        &gbx, &gby, &gbz);

        /* ---- Spring (linker) force ---- */
        /*
         * Current separation vector: warhead_b - warhead_a
         * Spring force on warhead_a: +K*(|sep|-L_eq) * sep_hat
         * Spring force on warhead_b: -K*(|sep|-L_eq) * sep_hat
         */
        float sepx = protac->warhead_b->cx - protac->warhead_a->cx;
        float sepy = protac->warhead_b->cy - protac->warhead_a->cy;
        float sepz = protac->warhead_b->cz - protac->warhead_a->cz;
        float sep  = sqrtf(sepx*sepx + sepy*sepy + sepz*sepz);

        float spring_fx = 0.0f, spring_fy = 0.0f, spring_fz = 0.0f;
        if (sep > 1e-6f) {
            float inv_sep = 1.0f / sep;
            float stretch = sep - L_eq;
            spring_fx = K * stretch * sepx * inv_sep;
            spring_fy = K * stretch * sepy * inv_sep;
            spring_fz = K * stretch * sepz * inv_sep;
        }

        /* Update separation in PROTAC struct */
        protac->sep_x      = sepx;
        protac->sep_y      = sepy;
        protac->sep_z      = sepz;
        protac->separation = sep;

        /* ---- Langevin update for warhead A ---- */
        vax = vax * exp_gdt + (gax + spring_fx) * dt
              + noise_sc * _rng_normal(&rng);
        vay = vay * exp_gdt + (gay + spring_fy) * dt
              + noise_sc * _rng_normal(&rng);
        vaz = vaz * exp_gdt + (gaz + spring_fz) * dt
              + noise_sc * _rng_normal(&rng);
        protac->warhead_a->cx += vax * dt;
        protac->warhead_a->cy += vay * dt;
        protac->warhead_a->cz += vaz * dt;

        /* ---- Langevin update for warhead B ---- */
        vbx = vbx * exp_gdt + (gbx - spring_fx) * dt
              + noise_sc * _rng_normal(&rng);
        vby = vby * exp_gdt + (gby - spring_fy) * dt
              + noise_sc * _rng_normal(&rng);
        vbz = vbz * exp_gdt + (gbz - spring_fz) * dt
              + noise_sc * _rng_normal(&rng);
        protac->warhead_b->cx += vbx * dt;
        protac->warhead_b->cy += vby * dt;
        protac->warhead_b->cz += vbz * dt;

        /* ---- Boundary conditions ---- */
        #define CLAMP_MOL(mol, grid) \
            do { \
                float _mx = (grid->dim_x - 2) * grid->resolution; \
                float _my = (grid->dim_y - 2) * grid->resolution; \
                float _mz = (grid->dim_z - 2) * grid->resolution; \
                if (mol->cx < 0) mol->cx = 0; if (mol->cx > _mx) mol->cx = _mx; \
                if (mol->cy < 0) mol->cy = 0; if (mol->cy > _my) mol->cy = _my; \
                if (mol->cz < 0) mol->cz = 0; if (mol->cz > _mz) mol->cz = _mz; \
            } while(0)
        CLAMP_MOL(protac->warhead_a, grid_target);
        CLAMP_MOL(protac->warhead_b, grid_e3);
        #undef CLAMP_MOL

        /* ---- Score current pose ---- */
        /*
         * Lightweight per-step scoring: project both warheads and
         * compute affinity directly (skip neo-PPI and cooperativity
         * for speed — recompute those at end for best pose).
         */
        nibble_reset_steric(scratch_a);
        nibble_mol_project(protac->warhead_a, scratch_a);
        float sc_a = nibble_compute_affinity_full(grid_target, scratch_a, 2.0f);

        nibble_reset_steric(scratch_b);
        nibble_mol_project(protac->warhead_b, scratch_b);
        float sc_b = nibble_compute_affinity_full(grid_e3, scratch_b, 2.0f);

        /* Quick linker penalty */
        float ete_now;
        float lp = nibble_protac_linker_penalty(&protac->linker, sep, &ete_now);

        float quick_composite = 0.4f * _clampf(sc_a / 2000.0f, 0.0f, 1.0f)
                               + 0.4f * _clampf(sc_b / 2000.0f, 0.0f, 1.0f)
                               + 0.2f * (1.0f - lp);

        if (quick_composite > best_composite) {
            best_composite = quick_composite;
            best_ax = protac->warhead_a->cx;
            best_ay = protac->warhead_a->cy;
            best_az = protac->warhead_a->cz;
            best_bx = protac->warhead_b->cx;
            best_by = protac->warhead_b->cy;
            best_bz = protac->warhead_b->cz;
        }
    }

    nibble_free_grid(scratch_a);
    nibble_free_grid(scratch_b);

    /* Restore best pose */
    protac->warhead_a->cx = best_ax;
    protac->warhead_a->cy = best_ay;
    protac->warhead_a->cz = best_az;
    protac->warhead_b->cx = best_bx;
    protac->warhead_b->cy = best_by;
    protac->warhead_b->cz = best_bz;

    /* Full score at best pose */
    nibble_protac_score(grid_target, grid_e3, protac, geometry, 2.0f, kT, out);

    return out->composite_score;
}

/* -----------------------------------------------------------------------
 * nibble_protac_print_score
 * ----------------------------------------------------------------------- */
void nibble_protac_print_score(const PROTAC_TernaryScore *s) {
    if (!s) return;
    printf("[nibble_protac] PROTAC Ternary Score Summary\n");
    printf("  Composite score:        %.4f\n", s->composite_score);
    printf("  Degradation efficiency: %.4f\n", s->deg_efficiency);
    printf("  ─── Components ───────────────────────────\n");
    printf("  Warhead A (target):     %.4f\n", s->score_warhead_a);
    printf("  Warhead B (E3 ligase):  %.4f\n", s->score_warhead_b);
    printf("  Linker feasibility:     %.4f  (penalty=%.4f, ete=%.1f Å)\n",
           s->linker_feasibility, s->linker_penalty, s->linker_end_to_end);
    printf("  Cooperativity alpha:    %.4f  %s\n",
           s->alpha, s->hook_effect ? "[Hook effect predicted!]" : "");
    printf("  Neo-PPI score:          %.4f\n", s->neo_ppi_score);
}
