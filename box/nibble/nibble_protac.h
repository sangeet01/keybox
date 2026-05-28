/*
 * nibble_protac.h
 * PROTAC Ternary Complex Extension for the Nibble Voxel Engine.
 *
 * PROTACs (PROteolysis TArgeting Chimeras) are bifunctional molecules:
 *
 *   [ Warhead A ] ─── [ Linker ] ─── [ Warhead B ]
 *        │                                  │
 *   binds Target                      binds E3 Ligase
 *
 * The drug does not inhibit the target directly. It recruits an E3 ubiquitin
 * ligase into proximity with the target, causing the target's ubiquitination
 * and subsequent proteasomal degradation. The key metric is not binary
 * binding but ternary complex formation — both warheads must be simultaneously
 * bound and the resulting protein-protein interface (Target:PROTAC:E3) must
 * be geometrically and energetically favourable.
 *
 * This extension adds four capabilities to Nibble:
 *
 *   1. Ternary grid superposition
 *      Both pocket grids (target + E3) are loaded simultaneously.
 *      The PROTAC NibbleMol is split into warhead_A, linker, warhead_B
 *      sub-molecules. Each warhead is evaluated against its pocket.
 *
 *   2. Linker geometry feasibility
 *      The linker must span the physical distance between the two pockets
 *      (target_centre → E3_centre) with the right length and flexibility.
 *      nibble_protac_linker_penalty() computes a penalty for impossible
 *      linker geometries using a Gaussian chain end-to-end distance model.
 *
 *   3. Cooperativity term (alpha)
 *      The ternary complex has a cooperative free energy contribution:
 *      binding one warhead changes the effective affinity of the other.
 *      This is the PROTAC cooperativity parameter α from the Hook et al.
 *      (2015) thermodynamic model. α > 1 = positive cooperativity,
 *      α < 1 = negative (Hook effect — too much PROTAC kills degradation).
 *
 *   4. Ternary Langevin trajectory
 *      nibble_protac_trajectory() runs simultaneous Langevin dynamics
 *      for both warheads, constrained by the linker geometry. The PROTAC
 *      is treated as a rigid dumbbell: warhead_A and warhead_B are coupled
 *      through a harmonic spring potential representing the linker.
 *
 * Output: PROTAC_TernaryScore — composite ternary complex score,
 * individual warhead scores, linker penalty, cooperativity factor,
 * and predicted degradation efficiency proxy.
 *
 * Architecture:
 *   Zero changes to nibble.c / nibble_tier3.c / nibble_tier4.c /
 *   nibble_gaps.c / nibble_ppi.c. Pure additive extension.
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#ifndef NIBBLE_PROTAC_H
#define NIBBLE_PROTAC_H

#include "nibble.h"
#include "nibble_ppi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * PROTAC constants
 * ----------------------------------------------------------------------- */

/* Linker model constants */
#define PROTAC_KUHN_LENGTH        3.8f   /* Å — C-C bond equivalent Kuhn segment */
#define PROTAC_MAX_LINKER_ATOMS   80     /* maximum linker heavy atoms            */
#define PROTAC_MIN_LINKER_ATOMS   3      /* minimum viable linker                 */

/* Cooperativity model */
#define PROTAC_ALPHA_NEUTRAL      1.0f   /* no cooperativity                      */
#define PROTAC_ALPHA_MAX          100.0f /* strong positive cooperativity         */
#define PROTAC_HOOK_THRESHOLD     0.3f   /* composite score below = Hook effect   */

/* Ternary Langevin */
#define PROTAC_SPRING_K           5.0f   /* kcal/mol/Å² — linker spring constant  */
#define PROTAC_MAX_TRAJ_STEPS     500    /* default trajectory steps              */

/* -----------------------------------------------------------------------
 * Linker representation
 * ----------------------------------------------------------------------- */
typedef struct {
    int    n_atoms;            /* number of linker heavy atoms              */
    float  contour_length;     /* Å — fully extended linker length          */
    float  persistence_length; /* Å — worm-like chain persistence length    */
    float  flexibility;        /* 0=rigid rod, 1=fully flexible coil        */
    /* Atom coordinates (if loaded from PDB/SMILES) */
    float  coords[PROTAC_MAX_LINKER_ATOMS * 3];
    /* Per-atom channel values for scoring (N_CHANNELS each) */
    float  ch_vals[PROTAC_MAX_LINKER_ATOMS * 16];
} PROTAC_Linker;

/* -----------------------------------------------------------------------
 * PROTAC molecule — bifunctional decomposition
 * ----------------------------------------------------------------------- */
typedef struct {
    NibbleMol *warhead_a;      /* warhead binding target protein            */
    NibbleMol *warhead_b;      /* warhead binding E3 ligase                 */
    PROTAC_Linker linker;      /* linker connecting the two warheads        */

    /* Separation vector (warhead_a COM → warhead_b COM) in grid space */
    float  sep_x, sep_y, sep_z;
    float  separation;         /* Å — current inter-warhead distance        */
} PROTAC_Molecule;

/* -----------------------------------------------------------------------
 * Ternary complex geometry
 * ----------------------------------------------------------------------- */
typedef struct {
    /* Target protein pocket centre (in target grid coordinates) */
    float  target_cx, target_cy, target_cz;

    /* E3 ligase pocket centre (in E3 grid coordinates) */
    float  e3_cx, e3_cy, e3_cz;

    /* Vector from target pocket to E3 pocket in global space */
    float  pocket_sep_x, pocket_sep_y, pocket_sep_z;
    float  pocket_separation;  /* Å — distance between pocket centres       */

    /* Required linker end-to-end distance (derived from pocket separation
       minus warhead radii, approximately) */
    float  required_linker_ete; /* Å */

    /* Induced PPI surface at the ternary interface */
    /* (Target surface induced by PROTAC presence + E3 surface) */
    int    has_neo_ppi;         /* 1 if a neo-PPI surface was detected      */
    float  neo_ppi_score;       /* score of the induced ternary PPI surface */
} PROTAC_TernaryGeometry;

/* -----------------------------------------------------------------------
 * Full ternary complex score
 * ----------------------------------------------------------------------- */
typedef struct {
    /* Individual warhead scores */
    float  score_warhead_a;        /* warhead A vs target pocket             */
    float  score_warhead_b;        /* warhead B vs E3 pocket                 */

    /* Linker geometry */
    float  linker_penalty;         /* 0=ideal, 1=impossible geometry         */
    float  linker_end_to_end;      /* Å — mean end-to-end distance           */
    float  linker_feasibility;     /* 1 - linker_penalty                     */

    /* Cooperativity */
    float  alpha;                  /* cooperativity parameter (Hook et al.)  */
    int    hook_effect;            /* 1 = Hook effect predicted              */

    /* Neo-PPI (induced protein-protein interface in ternary complex) */
    float  neo_ppi_score;          /* 0 if no neo-PPI detected               */

    /* Degradation efficiency proxy */
    float  deg_efficiency;         /* 0–1; composite degradation prediction  */

    /* Composite ternary score */
    float  composite_score;        /* 0–1; primary ranking metric            */

    /* Component weights used */
    float  w_a, w_b, w_linker, w_coop, w_neo_ppi;
} PROTAC_TernaryScore;

/* -----------------------------------------------------------------------
 * Function declarations
 * ----------------------------------------------------------------------- */

/*
 * Create a PROTAC_Molecule from atom data.
 *
 * The PROTAC is split into warhead_a, linker, warhead_b at the
 * split_a and split_b atom indices.
 *
 * Args:
 *   n_atoms_total:  total heavy atom count of the PROTAC
 *   coords:         flat [n_atoms * 3] float array (Å, local frame)
 *   ch_vals:        flat [n_atoms * N_CHANNELS] float array
 *   split_a:        last atom index of warhead A (exclusive) [0, split_b)
 *   split_b:        first atom index of warhead B [split_a, n_atoms)
 *                   atoms in [split_a, split_b) are the linker
 *
 * Returns: heap-allocated PROTAC_Molecule, or NULL on failure.
 */
PROTAC_Molecule *nibble_protac_create(
    int          n_atoms_total,
    const float *coords,
    const float *ch_vals,
    int          split_a,
    int          split_b
);

/*
 * Free a PROTAC_Molecule.
 */
void nibble_protac_free(PROTAC_Molecule *protac);

/*
 * Compute linker geometry feasibility using Gaussian chain model.
 *
 * Models the linker as a freely jointed chain (FJC) and computes
 * the probability that a chain of n_atoms Kuhn segments can span
 * the required end-to-end distance.
 *
 * P(r) = (3/(2*pi*<r²>))^(3/2) * exp(-3*r² / (2*<r²>))
 * <r²> = n_atoms * b² where b = PROTAC_KUHN_LENGTH
 *
 * Returns:
 *   penalty in [0, 1]: 0 = ideal span, 1 = geometrically impossible.
 *   Also fills linker_ete_out with the most probable end-to-end distance.
 */
float nibble_protac_linker_penalty(
    const PROTAC_Linker *linker,
    float                required_ete,
    float               *linker_ete_out
);

/*
 * Compute cooperativity parameter alpha.
 *
 * Based on Hook et al. (2015) thermodynamic model for ternary complex.
 * alpha = exp(-delta_G_ternary / kT)
 * where delta_G_ternary is estimated from the induced PPI surface energy.
 *
 * alpha > 1: both warheads bound cooperatively (productive PROTAC)
 * alpha < 1: anticooperative (Hook effect likely)
 * alpha ~ 1: neutral
 *
 * Args:
 *   score_a:   warhead A binding score (0–1)
 *   score_b:   warhead B binding score (0–1)
 *   neo_ppi:   neo-PPI surface score at ternary interface (0–1)
 *   kT:        thermal energy (kcal/mol)
 *
 * Returns: alpha (float, > 0)
 */
float nibble_protac_cooperativity(
    float score_a,
    float score_b,
    float neo_ppi,
    float kT
);

/*
 * Detect the neo-PPI surface induced at the Target:PROTAC:E3 interface.
 *
 * When a PROTAC bridges two proteins, it creates a novel protein-protein
 * interface that does not exist in the absence of the PROTAC. This
 * neo-PPI is critical for ternary complex stability — strong neo-PPI
 * = more stable ternary = better degradation.
 *
 * Uses nibble_ppi_affinity() on the target and E3 grids directly,
 * modulated by the PROTAC's linker position as a proximity mask.
 *
 * Args:
 *   grid_target:  NibbleGrid for target protein surface
 *   grid_e3:      NibbleGrid for E3 ligase surface
 *   protac:       PROTAC_Molecule (for linker proximity mask)
 *   iface:        PPI_Interface to fill (reuses PPI detection machinery)
 *   kT:           thermal energy
 *
 * Returns: neo-PPI composite score (0–1)
 */
float nibble_protac_neo_ppi(
    const NibbleGrid    *grid_target,
    const NibbleGrid    *grid_e3,
    const PROTAC_Molecule *protac,
    PPI_Interface        *iface,
    float                 kT
);

/*
 * Full PROTAC ternary complex scoring.
 *
 * Orchestrates all scoring components:
 *   1. Warhead A vs target grid (nibble_compute_affinity_full)
 *   2. Warhead B vs E3 grid (nibble_compute_affinity_full)
 *   3. Linker geometry penalty (nibble_protac_linker_penalty)
 *   4. Cooperativity alpha (nibble_protac_cooperativity)
 *   5. Neo-PPI score (nibble_protac_neo_ppi)
 *   6. Degradation efficiency proxy
 *
 * Composite:
 *   composite = w_a * norm(score_a) + w_b * norm(score_b)
 *             + w_linker * linker_feasibility
 *             + w_coop * log(alpha) / log(alpha_max)
 *             + w_neo_ppi * neo_ppi_score
 *             - hook_penalty (if Hook effect predicted)
 *
 * Args:
 *   grid_target:   NibbleGrid for target protein pocket
 *   grid_e3:       NibbleGrid for E3 ligase pocket
 *   protac:        PROTAC_Molecule
 *   geometry:      PROTAC_TernaryGeometry (pocket separation, etc.)
 *   delta_G_bulk:  desolvation free energy (kcal/mol)
 *   kT:            thermal energy (kcal/mol)
 *   out:           PROTAC_TernaryScore to fill
 *
 * Returns: composite ternary score (0–1)
 */
float nibble_protac_score(
    const NibbleGrid       *grid_target,
    const NibbleGrid       *grid_e3,
    PROTAC_Molecule        *protac,
    PROTAC_TernaryGeometry *geometry,
    float                   delta_G_bulk,
    float                   kT,
    PROTAC_TernaryScore    *out
);

/*
 * Ternary Langevin trajectory optimisation.
 *
 * Simultaneously optimises both warhead positions subject to the
 * linker harmonic spring constraint. The PROTAC is modelled as a
 * rigid dumbbell: warhead_a and warhead_b are coupled by a spring
 * with equilibrium length = linker contour length.
 *
 * Each Langevin step:
 *   1. Compute gradient for warhead_a in grid_target
 *   2. Compute gradient for warhead_b in grid_e3
 *   3. Add spring force: F_spring = -k * (|sep| - L_eq) * sep_hat
 *   4. Integrate both warhead positions with Langevin dynamics
 *   5. Update separation vector
 *
 * Args:
 *   grid_target:   NibbleGrid for target pocket
 *   grid_e3:       NibbleGrid for E3 pocket
 *   protac:        PROTAC_Molecule (modified in place)
 *   geometry:      PROTAC_TernaryGeometry
 *   n_steps:       number of trajectory steps
 *   dt:            time step (0.002 default)
 *   gamma:         friction coefficient
 *   kT:            thermal energy
 *   out:           PROTAC_TernaryScore for best pose found
 *
 * Returns: best composite ternary score found during trajectory.
 */
float nibble_protac_trajectory(
    const NibbleGrid       *grid_target,
    const NibbleGrid       *grid_e3,
    PROTAC_Molecule        *protac,
    PROTAC_TernaryGeometry *geometry,
    int                     n_steps,
    float                   dt,
    float                   gamma,
    float                   kT,
    PROTAC_TernaryScore    *out
);

/*
 * Utility: infer linker properties from atom count and SMILES.
 * Fills PROTAC_Linker.contour_length, .flexibility, .persistence_length.
 * If smiles is NULL, uses n_atoms only.
 */
void nibble_protac_infer_linker(
    PROTAC_Linker *linker,
    int            n_atoms,
    const char    *smiles
);

/*
 * Utility: print PROTAC_TernaryScore summary.
 */
void nibble_protac_print_score(const PROTAC_TernaryScore *score);

#endif /* NIBBLE_PROTAC_H */
