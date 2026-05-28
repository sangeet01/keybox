/*
 * nibble_peptide.h
 * Peptide-Protein Docking Extension for the Nibble Voxel Engine.
 *
 * Peptides are therapeutically important but poorly served by existing
 * docking tools. AutoDockCrankPep exists but is slow and ignores induced
 * fit. Glide SP handles peptides but requires an expensive licence.
 * No open-source engine handles stapled helices or cyclic peptides with
 * real backbone flexibility in a physics-grounded way.
 *
 * This extension adds four capabilities specific to peptide docking:
 *
 *   1. Backbone flexibility model
 *      Peptides are not rigid bodies. A 15-residue peptide has ~28
 *      rotatable backbone bonds. Nibble models this as a segmented
 *      dumbbell chain: the peptide is divided into N_SEGS rigid segments
 *      connected by flexible hinges. Each segment has its own position
 *      and orientation (quaternion). The Langevin integrator from
 *      nibble_tier3.c is extended to handle multi-segment chains.
 *
 *   2. Secondary structure bias
 *      Alpha-helices and beta-strands have distinct voxel field signatures.
 *      A helix presents a hydrophobic face every 3.6 residues — a
 *      periodic signal in the lipophilicity channel. A beta-strand
 *      presents alternating donor/acceptor H-bond patterns.
 *      nibble_peptide_ss_bias() adds a secondary-structure-derived
 *      bonus field to the grid that rewards the peptide adopting its
 *      preferred conformation at the interface.
 *
 *   3. Stapled helix and cyclic peptide support
 *      Stapled peptides (hydrocarbon-stapled alpha-helices) are
 *      constrained — two non-adjacent residues are covalently linked,
 *      locking the helical geometry. Cyclic peptides have their N and
 *      C termini connected, eliminating terminal flexibility.
 *      nibble_peptide_apply_constraint() enforces these geometric
 *      constraints as a penalty term in the Langevin trajectory.
 *
 *   4. Groove scanning
 *      Peptides don't bind spherical pockets — they bind grooves,
 *      channels, and extended flat surfaces. nibble_peptide_scan_groove()
 *      identifies linear binding grooves in the protein surface grid
 *      using ridge detection on the inverse steric field, producing
 *      a set of groove axis vectors and entry points that seed the
 *      Langevin trajectory with physically meaningful starting poses.
 *
 * Interaction classes covered after this extension:
 *   Small molecule → protein pocket       nibble.c
 *   Protein → protein interface           nibble_ppi.c
 *   PPI → small molecule inhibitor        nibble_ppi.c Phase VI
 *   PROTAC → ternary complex              nibble_protac.c
 *   Peptide → protein groove/surface      nibble_peptide.c  ← THIS FILE
 *
 * Architecture: zero changes to any existing Nibble file.
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#ifndef NIBBLE_PEPTIDE_H
#define NIBBLE_PEPTIDE_H

#include "nibble.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Peptide constants
 * ----------------------------------------------------------------------- */

#define PEP_MAX_RESIDUES      64    /* maximum peptide length supported     */
#define PEP_MAX_SEGMENTS      16    /* max flexible segments (chain beads)  */
#define PEP_MAX_CONSTRAINTS   32    /* max geometric constraints (staples)  */
#define PEP_MAX_GROOVES       20    /* max grooves detected on surface      */

/* Amino acid backbone geometry */
#define PEP_BOND_CA_C         1.52f /* Å — Cα-C bond                       */
#define PEP_BOND_C_N          1.33f /* Å — peptide C-N bond                 */
#define PEP_BOND_N_CA         1.46f /* Å — N-Cα bond                        */
#define PEP_RISE_PER_RES_HELIX 1.5f /* Å — axial rise per residue in helix  */
#define PEP_RISE_PER_RES_BETA  3.2f /* Å — axial rise per residue in strand */
#define PEP_HELIX_PERIOD       3.6f /* residues per helix turn              */

/* Groove detection parameters */
#define PEP_GROOVE_MIN_LEN    8.0f  /* Å — minimum groove length            */
#define PEP_GROOVE_WIDTH      6.0f  /* Å — typical peptide groove width     */
#define PEP_GROOVE_DEPTH      3.5f  /* Å — minimum groove depth             */

/* -----------------------------------------------------------------------
 * Secondary structure types
 * ----------------------------------------------------------------------- */
typedef enum {
    PEP_SS_COIL   = 0,
    PEP_SS_HELIX  = 1,
    PEP_SS_STRAND = 2,
    PEP_SS_TURN   = 3,
} PEP_SecondaryStructure;

/* -----------------------------------------------------------------------
 * Constraint types (staple / cyclisation)
 * ----------------------------------------------------------------------- */
typedef enum {
    PEP_CONSTRAINT_STAPLE    = 0,  /* hydrocarbon staple between i, i+4    */
    PEP_CONSTRAINT_CYCLIC    = 1,  /* N-C terminus cyclisation             */
    PEP_CONSTRAINT_DISULFIDE = 2,  /* Cys-Cys disulfide bridge             */
    PEP_CONSTRAINT_LACTAM    = 3,  /* side-chain lactam bridge             */
} PEP_ConstraintType;

/* -----------------------------------------------------------------------
 * Per-residue representation
 * ----------------------------------------------------------------------- */
typedef struct {
    char  resname[4];              /* 3-letter amino acid code             */
    char  one_letter;              /* single-letter code                   */
    float ca_x, ca_y, ca_z;       /* Cα position (grid frame)             */
    float phi, psi;                /* backbone dihedral angles (radians)   */
    PEP_SecondaryStructure ss;     /* secondary structure assignment       */
    float hydrophobicity;          /* Eisenberg scale -1 to +1             */
    float charge;                  /* formal charge at pH 7                */
    int   hbd_count;               /* H-bond donors on side chain          */
    int   hba_count;               /* H-bond acceptors on side chain       */
    float sasa;                    /* solvent accessible surface area (Å²) */
    int   is_constrained;          /* 1 if part of a staple/cyclic link    */
    float d[N_CHANNELS];           /* voxel channel demands for this res.  */
} PEP_Residue;

/* -----------------------------------------------------------------------
 * Geometric constraint (staple, disulfide, lactam, cyclisation)
 * ----------------------------------------------------------------------- */
typedef struct {
    PEP_ConstraintType type;
    int   res_i;                   /* first residue index                  */
    int   res_j;                   /* second residue index                 */
    float eq_distance;             /* Å — equilibrium distance             */
    float spring_k;                /* kcal/mol/Å² — constraint stiffness   */
} PEP_Constraint;

/* -----------------------------------------------------------------------
 * Flexible segment (chain bead for multi-segment Langevin)
 * ----------------------------------------------------------------------- */
typedef struct {
    int   res_start;               /* first residue in this segment        */
    int   res_end;                 /* last residue in this segment         */
    float cx, cy, cz;             /* segment centre of mass               */
    float qw, qx, qy, qz;        /* orientation quaternion               */
    float vx, vy, vz;            /* translational velocity               */
    float wx, wy, wz;            /* angular velocity                     */
    float mass;                   /* total mass (Da)                      */
    float inertia_xx, inertia_yy, inertia_zz; /* principal inertia       */
    /* Projection of this segment into channel values for scoring */
    float d[N_CHANNELS];
} PEP_Segment;

/* -----------------------------------------------------------------------
 * Groove descriptor
 * ----------------------------------------------------------------------- */
typedef struct {
    float entry_x, entry_y, entry_z;  /* groove entry point              */
    float axis_x, axis_y, axis_z;     /* unit vector along groove axis   */
    float length;                      /* Å — groove length               */
    float width;                       /* Å — mean groove width           */
    float depth;                       /* Å — mean groove depth           */
    float affinity_score;              /* integrated field score along groove */
    int   n_hbond_sites;               /* H-bond sites along groove        */
    int   n_hydrophobic_patches;       /* hydrophobic patches along groove */
} PEP_Groove;

/* -----------------------------------------------------------------------
 * Full peptide molecule for docking
 * ----------------------------------------------------------------------- */
typedef struct {
    PEP_Residue   residues[PEP_MAX_RESIDUES];
    int           n_residues;
    PEP_Segment   segments[PEP_MAX_SEGMENTS];
    int           n_segments;
    PEP_Constraint constraints[PEP_MAX_CONSTRAINTS];
    int           n_constraints;
    PEP_SecondaryStructure dominant_ss;  /* overall SS classification      */
    int           is_cyclic;             /* 1 if N-C cyclised               */
    int           is_stapled;            /* 1 if contains staple            */
    float         net_charge;            /* total formal charge at pH 7     */
    float         hydrophobic_moment;    /* amphipathic hydrophobic moment  */
    float         com_x, com_y, com_z;  /* centre of mass                  */
    float         length_angstrom;       /* end-to-end length in Å          */
} PEP_Peptide;

/* -----------------------------------------------------------------------
 * Peptide docking result
 * ----------------------------------------------------------------------- */
typedef struct {
    float composite_score;         /* overall docking score 0–1           */
    float affinity_score;          /* field inner product score 0–1       */
    float ss_bias_score;           /* secondary structure compatibility    */
    float groove_match_score;      /* peptide axis vs groove axis alignment*/
    float constraint_penalty;      /* staple/cyclic constraint violation   */
    float amphipathic_score;       /* hydrophobic face alignment           */
    float n_hbonds_formed;         /* estimated H-bonds at interface       */
    /* Best pose residue positions */
    float ca_coords[PEP_MAX_RESIDUES * 3]; /* Cα xyz of best pose         */
    /* Groove the peptide bound into (-1 = no groove) */
    int   groove_index;
    float rmsd_from_start;         /* Å — displacement from initial pose  */
} PEP_DockingResult;

/* -----------------------------------------------------------------------
 * Function declarations
 * ----------------------------------------------------------------------- */

/*
 * Build a PEP_Peptide from a sequence string and secondary structure
 * assignment. The peptide is placed in an extended conformation initially;
 * nibble_peptide_trajectory() will fold it into the binding site.
 *
 * Args:
 *   sequence:  one-letter amino acid sequence (e.g. "ACDEFGHIKLMNPQRSTVWY")
 *   ss_string: per-residue SS code: 'H'=helix, 'E'=strand, 'C'=coil
 *              (same length as sequence, or NULL for all-coil)
 *   is_cyclic: 1 if N-C cyclised
 *   out:       PEP_Peptide to fill
 *
 * Returns: number of residues parsed, -1 on error.
 */
int nibble_peptide_build(
    const char  *sequence,
    const char  *ss_string,
    int          is_cyclic,
    PEP_Peptide *out
);

/*
 * Add a geometric constraint (staple, disulfide, lactam).
 *
 * Args:
 *   pep:    PEP_Peptide to modify
 *   type:   constraint type
 *   res_i:  first residue index (0-based)
 *   res_j:  second residue index
 *   spring_k: constraint spring constant (kcal/mol/Å², 10.0 default)
 *
 * Returns: constraint index, -1 on error.
 */
int nibble_peptide_add_constraint(
    PEP_Peptide       *pep,
    PEP_ConstraintType type,
    int                res_i,
    int                res_j,
    float              spring_k
);

/*
 * Detect binding grooves on the protein surface grid.
 *
 * Uses ridge detection on the inverse steric field:
 * regions where steric is low but adjacent field channels are high.
 * Returns PEP_Groove array sorted by affinity_score descending.
 *
 * Args:
 *   grid:     NibbleGrid for the protein surface
 *   grooves:  array of PEP_Groove to fill (size PEP_MAX_GROOVES)
 *   min_len:  minimum groove length in Å (default PEP_GROOVE_MIN_LEN)
 *
 * Returns: number of grooves detected.
 */
int nibble_peptide_scan_grooves(
    const NibbleGrid *grid,
    PEP_Groove       *grooves,
    float             min_len
);

/*
 * Apply secondary structure bias to the protein surface grid.
 *
 * Adds a bonus field layer that rewards the peptide adopting its
 * preferred secondary structure at the interface. For helices,
 * this creates a periodic hydrophobic channel every 3.6 residues
 * along groove axes. For strands, it creates alternating H-bond
 * donor/acceptor bands.
 *
 * The bonus is added IN PLACE to grid channels (non-destructive:
 * the original grid can be restored via nibble_reset_steric).
 *
 * Args:
 *   grid:  NibbleGrid to modify
 *   pep:   PEP_Peptide (for dominant_ss and hydrophobic_moment)
 *   groove: PEP_Groove to bias along (NULL = apply globally)
 *   weight: strength of the bias (0.1–0.5 recommended)
 */
void nibble_peptide_ss_bias(
    NibbleGrid       *grid,
    const PEP_Peptide *pep,
    const PEP_Groove  *groove,
    float              weight
);

/*
 * Project a PEP_Peptide segment into a NibbleGrid for scoring.
 *
 * Unlike small molecules (single NibbleMol), peptides are projected
 * segment by segment, with each segment's channel demands summed
 * into the grid. This allows the multi-segment Langevin trajectory
 * to score each segment independently.
 *
 * Args:
 *   grid:    NibbleGrid to project into (reset steric first)
 *   pep:     PEP_Peptide
 *   seg_idx: segment index to project (-1 = all segments)
 *   blur:    Gaussian blur radius (1.5Å recommended)
 */
void nibble_peptide_project(
    NibbleGrid        *grid,
    const PEP_Peptide *pep,
    int                seg_idx,
    float              blur
);

/*
 * Multi-segment Langevin trajectory for peptide docking.
 *
 * Runs simultaneous Langevin dynamics for all peptide segments,
 * coupled by:
 *   1. Backbone harmonic springs between adjacent segment COMs
 *   2. Geometric constraints (staples, disulfides)
 *   3. Secondary structure dihedral bias (phi/psi preference wells)
 *
 * Each segment has independent translational and rotational dynamics.
 * The gradient field from the protein grid drives each segment
 * toward its optimal position independently, while the spring
 * coupling maintains chain connectivity.
 *
 * Args:
 *   grid:     NibbleGrid for the protein surface (with SS bias applied)
 *   pep:      PEP_Peptide (modified in place — best pose stored)
 *   groove:   seed groove for initial placement (NULL = centre of grid)
 *   n_steps:  trajectory steps (500 default)
 *   dt:       time step (0.002 default)
 *   gamma:    friction coefficient (0.5 for peptides — less damping)
 *   kT:       thermal energy (0.593 kcal/mol at 298K)
 *   out:      PEP_DockingResult to fill
 *
 * Returns: best composite score found.
 */
float nibble_peptide_trajectory(
    const NibbleGrid *grid,
    PEP_Peptide      *pep,
    const PEP_Groove *groove,
    int               n_steps,
    float             dt,
    float             gamma,
    float             kT,
    PEP_DockingResult *out
);

/*
 * Compute amphipathic hydrophobic moment of the peptide.
 * Used to detect amphipathic helices and orient them correctly
 * at hydrophobic/hydrophilic interface grooves.
 *
 * Eisenberg (1984) hydrophobic moment:
 *   µH = |Σ Hi * exp(i * 2π/3.6 * j)|
 * where Hi is the hydrophobicity of residue i and j is the position.
 *
 * Returns: hydrophobic moment (0 = non-amphipathic, >0.5 = amphipathic)
 */
float nibble_peptide_hydrophobic_moment(
    const PEP_Peptide *pep,
    PEP_SecondaryStructure ss
);

/*
 * Full peptide-protein docking pipeline.
 * Orchestrates groove scanning → SS bias → trajectory → scoring.
 *
 * Args:
 *   protein_grid:  NibbleGrid for protein surface
 *   pep:           PEP_Peptide (modified to best pose in place)
 *   delta_G_bulk:  desolvation energy (kcal/mol)
 *   kT:            thermal energy
 *   n_steps:       trajectory steps per groove trial
 *   out:           PEP_DockingResult for best pose
 *
 * Returns: best composite score across all groove trials.
 */
float nibble_peptide_dock(
    const NibbleGrid *protein_grid,
    PEP_Peptide      *pep,
    float             delta_G_bulk,
    float             kT,
    int               n_steps,
    PEP_DockingResult *out
);

/*
 * Utility: print PEP_DockingResult summary.
 */
void nibble_peptide_print_result(const PEP_DockingResult *result);

/*
 * Utility: print PEP_Peptide summary.
 */
void nibble_peptide_print(const PEP_Peptide *pep);

#endif /* NIBBLE_PEPTIDE_H */
