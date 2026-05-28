/*
 * nibble_antibody.h
 * Antibody CDR-Antigen Docking Extension for the Nibble Voxel Engine.
 *
 * Antibody-antigen docking is among the hardest problems in structural
 * biology. The reasons:
 *
 *   1. Six CDR loops of highly variable length and sequence, each with
 *      different canonical or non-canonical structure.
 *   2. CDR-H3 — the heaviest contributor to binding — has essentially
 *      no canonical structure. It folds differently in every antibody.
 *   3. The paratope (antigen-binding surface) is not a pocket — it is a
 *      concave patch formed by the six loops together. Shape is formed
 *      by the assembly, not by any single loop.
 *   4. The antigen epitope is also a surface patch — flat, concave, or
 *      convex depending on the antigen. Not a groove. Not a pocket.
 *   5. The VH/VL framework must remain rigid while only the CDR tips flex.
 *   6. IsoDDE (Feb 2026) achieves 2.3× AlphaFold3 accuracy on antibody-
 *      antigen prediction specifically on CDR-H3. That is the benchmark.
 *
 * This extension adds five capabilities:
 *
 *   1. CDR loop representation
 *      Six CDR loops (L1, L2, L3, H1, H2, H3) are parsed from the
 *      antibody PDB using Kabat/Chothia numbering conventions.
 *      Each loop is a PEP_Peptide sub-chain (reusing nibble_peptide)
 *      with its own segment-based Langevin dynamics.
 *      The framework Fv region (VH + VL) is treated as a rigid body.
 *
 *   2. Paratope demand field
 *      The six CDR loops together form a composite demand field —
 *      the paratope. nibble_antibody_build_paratope() projects all
 *      six loops into a single NibbleGrid representing what the
 *      antigen surface must present to be bound.
 *
 *   3. Epitope surface detection
 *      nibble_antibody_scan_epitope() detects surface-exposed patches
 *      on the antigen that are complementary to the paratope field.
 *      Uses a patch complementarity scan: slide the paratope grid over
 *      the antigen surface, compute Frobenius inner product at each
 *      position, find the maximum.
 *
 *   4. CDR-H3 flexible docking
 *      nibble_antibody_h3_trajectory() runs dedicated Langevin
 *      dynamics for CDR-H3 alone (the other CDRs held fixed at their
 *      canonical positions) against the antigen surface. This is the
 *      most expensive step but the most important for accuracy.
 *
 *   5. Full paratope-epitope scoring
 *      nibble_antibody_score() assembles the composite binding score:
 *      paratope-epitope shape complementarity (Sc, reusing nibble_ppi),
 *      CDR-H3 affinity, canonical CDR contributions, VH/VL orientation
 *      penalty, and buried surface area.
 *
 * Architecture: zero changes to any existing Nibble file.
 * Reuses nibble_peptide for CDR loop representation.
 * Reuses nibble_ppi for shape complementarity scoring.
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#ifndef NIBBLE_ANTIBODY_H
#define NIBBLE_ANTIBODY_H

#include "nibble.h"
#include "nibble_ppi.h"
#include "nibble_peptide.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Constants
 * ----------------------------------------------------------------------- */

#define AB_N_CDR_LOOPS      6      /* L1, L2, L3, H1, H2, H3              */
#define AB_MAX_CDR_LEN      30     /* max CDR loop residues                */
#define AB_MAX_EPITOPE_PATCHES 20  /* max epitope patches on antigen       */
#define AB_FV_MAX_RESIDUES  250    /* max Fv framework residues            */

/* CDR loop indices */
#define AB_CDR_L1    0
#define AB_CDR_L2    1
#define AB_CDR_L3    2
#define AB_CDR_H1    3
#define AB_CDR_H2    4
#define AB_CDR_H3    5

/* CDR contribution weights to paratope score
 * H3 dominates; H1,H2,L3 are next; L1,L2 contribute least.
 * From Raghunathan et al. structural analysis of CDR contributions. */
static const float AB_CDR_WEIGHTS[AB_N_CDR_LOOPS] = {
    0.08f,  /* L1 */
    0.06f,  /* L2 */
    0.14f,  /* L3 */
    0.14f,  /* H1 */
    0.14f,  /* H2 */
    0.44f,  /* H3 — dominant */
};

/* CDR name strings */
static const char *AB_CDR_NAMES[AB_N_CDR_LOOPS] = {
    "L1", "L2", "L3", "H1", "H2", "H3"
};

/* -----------------------------------------------------------------------
 * CDR loop descriptor
 * ----------------------------------------------------------------------- */
typedef struct {
    int   cdr_idx;                 /* AB_CDR_L1 ... AB_CDR_H3              */
    char  sequence[AB_MAX_CDR_LEN + 1];
    int   n_residues;
    /* Cα positions in Fv frame */
    float ca_x[AB_MAX_CDR_LEN];
    float ca_y[AB_MAX_CDR_LEN];
    float ca_z[AB_MAX_CDR_LEN];
    /* Anchor residues: N-terminal and C-terminal anchors to framework */
    float anchor_n_x, anchor_n_y, anchor_n_z;
    float anchor_c_x, anchor_c_y, anchor_c_z;
    /* Canonical structure class (Chothia canonical, 0 = non-canonical) */
    int   canonical_class;
    /* Whether this CDR uses the flexible H3 trajectory */
    int   is_h3;
    /* Reuse PEP_Peptide for channel demands and segment dynamics */
    PEP_Peptide pep;
} AB_CDRLoop;

/* -----------------------------------------------------------------------
 * Fv framework (rigid body)
 * ----------------------------------------------------------------------- */
typedef struct {
    /* VH domain centre */
    float vh_cx, vh_cy, vh_cz;
    /* VL domain centre */
    float vl_cx, vl_cy, vl_cz;
    /* VH-VL elbow angle (degrees) — typically 140–180 */
    float elbow_angle;
    /* Relative VH/VL orientation quaternion */
    float qw, qx, qy, qz;
    /* Rigid body translational and rotational velocity */
    float vx, vy, vz;
    float wx, wy, wz;
    /* Framework mass */
    float mass;
} AB_FvFramework;

/* -----------------------------------------------------------------------
 * Epitope patch on antigen surface
 * ----------------------------------------------------------------------- */
typedef struct {
    float cx, cy, cz;          /* patch centre                            */
    float normal_x, normal_y, normal_z; /* outward surface normal         */
    float area;                /* Å² — patch area                         */
    float complementarity;     /* Frobenius inner product with paratope   */
    float sc_score;            /* Lawrence-Coleman Sc at this patch        */
    float buried_area;         /* estimated buried area upon binding       */
    int   n_hbond_sites;       /* H-bond sites on the epitope             */
    int   n_hydrophobic;       /* hydrophobic contacts                    */
    int   n_charged;           /* charged residues on epitope             */
    int   is_linear_epitope;   /* 1 if contiguous sequence segment        */
    int   is_conformational;   /* 1 if formed by non-contiguous residues  */
} AB_EpitopePatch;

/* -----------------------------------------------------------------------
 * Full antibody docking result
 * ----------------------------------------------------------------------- */
typedef struct {
    float composite_score;         /* overall binding score 0–1           */
    float paratope_epitope_sc;     /* shape complementarity (Sc) 0–1      */
    float h3_affinity;             /* CDR-H3 affinity contribution 0–1    */
    float canonical_cdr_score;     /* L1-L2-L3-H1-H2 combined 0–1        */
    float buried_surface_area;     /* Å² total buried on both sides        */
    float vhvl_orientation_penalty;/* VH/VL geometry distortion 0–1       */
    /* Per-CDR scores */
    float cdr_scores[AB_N_CDR_LOOPS];
    /* Best epitope patch */
    int   best_epitope_idx;
    float best_epitope_complementarity;
    /* Number of epitope patches detected */
    int   n_epitope_patches;
    /* Estimated Kd class */
    int   kd_class;    /* 0=nM range, 1=uM range, 2=weak/non-binder      */
} AB_DockingResult;

/* -----------------------------------------------------------------------
 * Full antibody structure
 * ----------------------------------------------------------------------- */
typedef struct {
    AB_CDRLoop   cdrs[AB_N_CDR_LOOPS];
    AB_FvFramework framework;
    /* Paratope demand grid (built from all 6 CDRs) */
    /* Allocated/freed by nibble_antibody_build_paratope() */
    NibbleGrid  *paratope_grid;
    int          paratope_built;
    /* Antibody identifier */
    char         name[32];
} AB_Antibody;

/* -----------------------------------------------------------------------
 * Function declarations
 * ----------------------------------------------------------------------- */

/*
 * Parse an antibody Fv from a PDB file.
 * Identifies CDR loops using Kabat/Chothia positional numbering.
 * Builds AB_CDRLoop structs from the PDB ATOM records.
 * Populates AB_FvFramework from framework residues.
 *
 * Args:
 *   pdb_path:  path to antibody PDB (Fv or Fab format)
 *   vh_chain:  chain ID for VH domain
 *   vl_chain:  chain ID for VL domain
 *   out:       AB_Antibody to fill
 *
 * Returns: total CDR residues parsed, -1 on error.
 */
int nibble_antibody_parse(
    const char  *pdb_path,
    char         vh_chain,
    char         vl_chain,
    AB_Antibody *out
);

/*
 * Build CDR loops from explicit sequences (no PDB required).
 * Places loops in canonical or extended conformation.
 * Useful for de novo antibody design.
 *
 * Args:
 *   sequences: array of 6 CDR sequences (NULL entries use empty loop)
 *   out:       AB_Antibody to fill
 *
 * Returns: total CDR residues built, -1 on error.
 */
int nibble_antibody_build_from_sequences(
    const char  *sequences[AB_N_CDR_LOOPS],
    AB_Antibody *out
);

/*
 * Build the paratope demand grid from all six CDR loops.
 *
 * Projects all CDR channel demands into a single NibbleGrid.
 * CDR-H3 contribution is weighted by AB_CDR_WEIGHTS[H3]=0.44.
 * The resulting grid represents the "shape negative" of the ideal antigen.
 *
 * Args:
 *   ab:        AB_Antibody (CDRs must be built)
 *   grid_dim:  voxel grid dimension (80 recommended)
 *   grid_res:  resolution in Å (0.5 recommended)
 *
 * Returns: pointer to allocated NibbleGrid (stored in ab->paratope_grid),
 *          or NULL on failure. Caller must call nibble_free_grid() when done.
 */
NibbleGrid *nibble_antibody_build_paratope(
    AB_Antibody *ab,
    int          grid_dim,
    float        grid_res
);

/*
 * Scan the antigen surface for complementary epitope patches.
 *
 * Slides the paratope demand field over the antigen surface grid,
 * computing Frobenius inner product at each position.
 * Returns patches sorted by complementarity score.
 *
 * Algorithm:
 *   1. For each surface voxel on antigen grid (transitional steric):
 *   2. Sample paratope grid centred at that voxel
 *   3. Compute Frobenius inner product (sum of channel products)
 *   4. If above threshold: record as candidate epitope patch
 *   5. Cluster nearby candidates into patches
 *   6. Sort patches by score
 *
 * Args:
 *   antigen_grid:   NibbleGrid for antigen surface
 *   paratope_grid:  NibbleGrid from nibble_antibody_build_paratope()
 *   patches:        array of AB_EpitopePatch to fill (size AB_MAX_EPITOPE_PATCHES)
 *   threshold:      minimum inner product to consider (0.1 default)
 *
 * Returns: number of patches detected.
 */
int nibble_antibody_scan_epitope(
    const NibbleGrid   *antigen_grid,
    const NibbleGrid   *paratope_grid,
    AB_EpitopePatch    *patches,
    float               threshold
);

/*
 * Run CDR-H3 flexible Langevin trajectory against the antigen surface.
 *
 * CDR-H3 is the most variable CDR and dominates binding.
 * This function runs the PEP trajectory for H3 alone,
 * with the other five CDRs held at their current positions
 * as rigid context (contributing to gradient but not moving).
 *
 * Uses the multi-segment Langevin from nibble_peptide_trajectory()
 * internally, with additional constraints from the N/C anchor
 * residues connecting H3 to the framework.
 *
 * Args:
 *   antigen_grid:  NibbleGrid for antigen surface at best epitope
 *   ab:            AB_Antibody (H3 modified in place)
 *   epitope:       best epitope patch (sets initial H3 placement)
 *   n_steps:       trajectory steps (300 default)
 *   kT:            thermal energy
 *   score_out:     H3 affinity score returned here
 *
 * Returns: best CDR-H3 affinity score.
 */
float nibble_antibody_h3_trajectory(
    const NibbleGrid    *antigen_grid,
    AB_Antibody         *ab,
    const AB_EpitopePatch *epitope,
    int                  n_steps,
    float                kT,
    float               *score_out
);

/*
 * Full antibody-antigen docking score.
 *
 * Assembles composite score from all components:
 *   1. Paratope-epitope Sc (shape complementarity via nibble_ppi)
 *   2. CDR-H3 affinity (nibble_antibody_h3_trajectory)
 *   3. Canonical CDR contributions (L1,L2,L3,H1,H2 channel products)
 *   4. Buried surface area estimate
 *   5. VH/VL orientation penalty
 *
 * Composite:
 *   composite = w_sc * sc_score
 *             + w_h3 * h3_affinity
 *             + w_cdr * canonical_cdr_score
 *             + w_bsa * norm(buried_area)
 *             - w_vhvl * vhvl_penalty
 *
 * Args:
 *   antigen_grid:  NibbleGrid for antigen surface
 *   ab:            AB_Antibody (paratope_grid must be built)
 *   n_steps:       CDR-H3 trajectory steps
 *   delta_G:       water desolvation energy
 *   kT:            thermal energy
 *   out:           AB_DockingResult to fill
 *
 * Returns: composite binding score (0–1).
 */
float nibble_antibody_score(
    const NibbleGrid *antigen_grid,
    AB_Antibody      *ab,
    int               n_steps,
    float             delta_G,
    float             kT,
    AB_DockingResult *out
);

/*
 * Free allocated resources in an AB_Antibody.
 * Frees paratope_grid if built.
 */
void nibble_antibody_free(AB_Antibody *ab);

/*
 * Utility: print AB_DockingResult summary.
 */
void nibble_antibody_print_result(const AB_DockingResult *result);

/*
 * Utility: print AB_Antibody CDR summary.
 */
void nibble_antibody_print(const AB_Antibody *ab);

#endif /* NIBBLE_ANTIBODY_H */
