/*
 * nibble_ppi.h
 * Protein-Protein Interface (PPI) Extension for the Nibble Voxel Engine.
 *
 * Extends the existing 16-channel demand tensor framework to handle
 * protein-protein interactions without modifying any existing nibble files.
 *
 * Design principles:
 *   - Zero changes to nibble.c / nibble_tier3.c / nibble_tier4.c / nibble_gaps.c
 *   - All PPI logic is additive — new functions only
 *   - Both proteins load as NibbleGrid objects (existing type)
 *   - Protein B loads as NibbleMol for Langevin trajectory (existing type)
 *   - PPI scoring uses nibble_compute_affinity_full (existing function)
 *
 * Architecture:
 *
 *   Phase I  — Interface detection
 *              nibble_detect_interface() finds the contacting surface region
 *              between two PDB files using inter-residue distance cutoff.
 *              Returns PPI_Interface struct with hotspot mask and centre.
 *
 *   Phase II — Surface loading
 *              nibble_load_ppi_surface() loads a protein surface patch
 *              (not a pocket — larger, flatter, irregular) into a NibbleGrid.
 *              Adds a 17th virtual hotspot weight via CH_PHOBIC_CORE channel
 *              modulation (no schema change needed).
 *
 *   Phase III — Shape complementarity
 *               nibble_sc_score() computes Lawrence-Coleman shape
 *               complementarity (Sc) between two loaded surface grids.
 *               Sc = 1.0 = perfect fit, 0.0 = no complementarity.
 *               Standard threshold for known PPIs: Sc > 0.70.
 *
 *   Phase IV  — Hotspot scoring
 *               nibble_hotspot_score() weights interface residues by
 *               their contribution to binding energy using Alanine
 *               Scanning proxy: steric + electrostatic + H-bond channels.
 *
 *   Phase V   — Full PPI affinity
 *               nibble_ppi_affinity() integrates shape complementarity,
 *               hotspot scores, water network, and desolvation into a
 *               single PPI binding score comparable across complexes.
 *
 *   Phase VI  — Inhibitor mode
 *               nibble_load_ppi_inhibitor_site() identifies the minimal
 *               pharmacophore volume from the PPI interface that a small
 *               molecule inhibitor must occupy to disrupt binding.
 *               Output is a NibbleGrid pocket compatible with the existing
 *               Langevin docking workflow — no new code needed downstream.
 *
 * Channel usage (existing 16 channels, no schema change):
 *   CH_STERIC_DEMAND    — surface accessibility (1=exposed, 0=buried)
 *   CH_ELEC_DEMAND      — complementary charge demand
 *   CH_HBA_DEMAND       — H-bond acceptor demand at interface
 *   CH_HBD_DEMAND       — H-bond donor demand at interface
 *   CH_LIPO_DEMAND      — hydrophobic patch demand
 *   CH_AROM_DEMAND      — aromatic stacking demand
 *   CH_METAL_DEMAND     — metal coordination (bridge metals at PPI)
 *   CH_PHOBIC_CORE      — repurposed: hotspot weight (0–1)
 *   CH_FLEXIBILITY      — per-residue interface flexibility
 *   CH_NUCLEOPHILICITY  — Fukui f- for covalent crosslink detection
 *   CH_ELECTROPHILICITY — Fukui f+ for covalent crosslink detection
 *   CH_WATER_CONSERVED  — crystallographic bridging waters at interface
 *   CH_WATER_DYNAMIC    — dynamic water occupancy at interface
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#ifndef NIBBLE_PPI_H
#define NIBBLE_PPI_H

#include "nibble.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * PPI constants
 * ----------------------------------------------------------------------- */

#define PPI_MAX_RESIDUES       2000   /* max residues per protein surface    */
#define PPI_INTERFACE_CUTOFF   6.0f   /* Å — inter-atom contact distance     */
#define PPI_SURFACE_RANGE      40.0f  /* Å — default surface patch half-size */
#define PPI_HOTSPOT_ENERGY     2.0f   /* kcal/mol threshold for hotspot flag */
#define PPI_SC_NEIGHBOUR_R     7.5f   /* Å — shape complementarity dot window*/
#define PPI_MAX_ATOMS         50000   /* max atoms per PDB for PPI loading   */

/* -----------------------------------------------------------------------
 * Residue record (parsed from PDB for interface detection)
 * ----------------------------------------------------------------------- */
typedef struct {
    char  chain;
    int   resnum;
    char  resname[4];
    float cx, cy, cz;   /* residue centroid (mean of heavy atoms) */
    int   n_atoms;
    int   is_hotspot;    /* set by nibble_score_hotspots()          */
    float hotspot_weight; /* 0–1 energy contribution proxy          */
} PPI_Residue;

/* -----------------------------------------------------------------------
 * PPI interface descriptor
 * Returned by nibble_detect_interface().
 * ----------------------------------------------------------------------- */
typedef struct {
    /* Interface centre of geometry */
    float centre_x, centre_y, centre_z;

    /* Bounding box half-extents of interface region */
    float extent_x, extent_y, extent_z;

    /* Interface residues (protein A side) */
    PPI_Residue residues_a[PPI_MAX_RESIDUES];
    int         n_residues_a;

    /* Interface residues (protein B side) */
    PPI_Residue residues_b[PPI_MAX_RESIDUES];
    int         n_residues_b;

    /* Buried surface area estimate (Å²) */
    float buried_surface_area;

    /* Number of inter-protein H-bonds detected */
    int   n_hbonds;

    /* Number of hydrophobic contacts */
    int   n_hydrophobic;

    /* Bridging water count (if HETATM HOH within 3.5Å of both proteins) */
    int   n_bridging_waters;

    /* Shape complementarity score (0–1), filled by nibble_sc_score() */
    float sc_score;

    /* Mean hotspot weight across interface */
    float mean_hotspot_weight;

} PPI_Interface;

/* -----------------------------------------------------------------------
 * PPI scoring result
 * Returned by nibble_ppi_affinity().
 * ----------------------------------------------------------------------- */
typedef struct {
    float raw_affinity;        /* Frobenius inner product (field units²) */
    float sc_score;            /* Lawrence-Coleman shape complementarity  */
    float hotspot_score;       /* weighted hotspot contribution           */
    float water_bonus;         /* bridging water H-bond bonus             */
    float desolvation_penalty; /* cost of burying surface area            */
    float composite_score;     /* weighted combination of all above       */
    int   n_hotspots;          /* number of hotspot residues detected     */
    float buried_area_a;       /* buried Å² on protein A surface          */
    float buried_area_b;       /* buried Å² on protein B surface          */
} PPI_Score;

/* -----------------------------------------------------------------------
 * Function declarations
 * ----------------------------------------------------------------------- */

/*
 * Phase I: Interface detection.
 *
 * Loads two PDB files and identifies residues at the protein-protein
 * interface using an inter-atom distance cutoff (PPI_INTERFACE_CUTOFF).
 *
 * Args:
 *   pdb_a:   path to PDB file for protein A
 *   pdb_b:   path to PDB file for protein B (or second chain in same PDB)
 *   chain_a: chain identifier for protein A (e.g. 'A', 0 = any)
 *   chain_b: chain identifier for protein B (e.g. 'B', 0 = any)
 *   cutoff:  contact distance in Å (pass 0 for default PPI_INTERFACE_CUTOFF)
 *   out:     pointer to PPI_Interface to fill
 *
 * Returns:
 *   Number of interface residue pairs found, -1 on error.
 */
int nibble_detect_interface(
    const char *pdb_a,
    const char *pdb_b,
    char        chain_a,
    char        chain_b,
    float       cutoff,
    PPI_Interface *out
);

/*
 * Phase II: Surface loading.
 *
 * Loads the interface surface patch of one protein into a NibbleGrid.
 * Unlike nibble_load_pdb_pocket() (which loads a spherical pocket),
 * this function loads an irregular surface patch defined by the
 * PPI_Interface residue list.
 *
 * The CH_PHOBIC_CORE channel is repurposed as a hotspot weight map:
 * voxels near hotspot residues get higher CH_PHOBIC_CORE values,
 * making the affinity score hotspot-weighted without a schema change.
 *
 * Args:
 *   grid:      pre-allocated NibbleGrid (use nibble_create_grid)
 *   pdb_path:  path to PDB file
 *   iface:     PPI_Interface from nibble_detect_interface()
 *   use_side_a: 1 = load protein A residues, 0 = load protein B residues
 *   blur_radius: Gaussian blur radius in Å (1.5 recommended)
 *
 * Returns:
 *   Number of atoms projected, -1 on error.
 */
int nibble_load_ppi_surface(
    NibbleGrid    *grid,
    const char    *pdb_path,
    const PPI_Interface *iface,
    int            use_side_a,
    float          blur_radius
);

/*
 * Phase III: Shape complementarity (Lawrence-Coleman Sc).
 *
 * Computes the shape complementarity score between two loaded surface
 * grids. Sc measures how well the molecular surfaces of the two proteins
 * fit together, independent of chemical interactions.
 *
 * Algorithm:
 *   For each void voxel in grid_a (steric demand > 0.5):
 *     Find the nearest wall voxel in grid_b.
 *     Compute dot product of surface normals (estimated from gradient).
 *     Accumulate weighted dot product.
 *   Normalise to [0, 1].
 *
 * Sc > 0.70 indicates a well-complementary interface (antibody-antigen ~0.82,
 * designed PPI inhibitors ~0.74).
 *
 * Args:
 *   grid_a:  NibbleGrid for protein A surface
 *   grid_b:  NibbleGrid for protein B surface
 *
 * Returns:
 *   Sc score in [0, 1].
 */
float nibble_sc_score(
    const NibbleGrid *grid_a,
    const NibbleGrid *grid_b
);

/*
 * Phase IV: Hotspot scoring.
 *
 * Scores each interface residue for its likely contribution to binding
 * energy using an alanine-scanning proxy computed from the voxel fields.
 *
 * Proxy:
 *   hotspot_weight(residue) = w_steric * steric_burial
 *                           + w_elec   * |electrostatic_field|
 *                           + w_hb     * (hba + hbd)
 *                           + w_lipo   * hydrophobic_contact
 *
 * Residues with hotspot_weight > PPI_HOTSPOT_ENERGY are flagged.
 * Updates PPI_Interface.residues_a/b[i].is_hotspot and hotspot_weight.
 *
 * Args:
 *   grid_a:  NibbleGrid for protein A surface
 *   grid_b:  NibbleGrid for protein B surface
 *   iface:   PPI_Interface to update with hotspot flags
 *
 * Returns:
 *   Number of hotspot residues identified.
 */
int nibble_score_hotspots(
    const NibbleGrid *grid_a,
    const NibbleGrid *grid_b,
    PPI_Interface    *iface
);

/*
 * Phase V: Full PPI affinity score.
 *
 * Integrates all scoring components into a single PPI binding score.
 * Components:
 *   1. Raw affinity: nibble_compute_affinity_full() (existing function)
 *   2. Shape complementarity: nibble_sc_score()
 *   3. Hotspot weighting: residue-level energy contribution
 *   4. Water network: nibble_compute_water_network() (existing function)
 *   5. Desolvation: nibble_desolvation_penalty() (existing function)
 *
 * Composite score:
 *   composite = w_aff * norm(raw_affinity)
 *             + w_sc  * sc_score
 *             + w_hot * norm(hotspot_score)
 *             - w_des * norm(desolvation)
 *             + w_wat * norm(water_bonus)
 *
 * Args:
 *   grid_a:         NibbleGrid for protein A surface
 *   grid_b:         NibbleGrid for protein B surface
 *   iface:          PPI_Interface (must have hotspot scores)
 *   delta_G_bulk:   water desolvation cost (kcal/mol, ~2.0)
 *   kT:             thermal energy (0.593 kcal/mol at 298K)
 *   out:            PPI_Score struct to fill
 *
 * Returns:
 *   Composite PPI score (higher = stronger predicted binding).
 */
float nibble_ppi_affinity(
    const NibbleGrid *grid_a,
    const NibbleGrid *grid_b,
    PPI_Interface    *iface,
    float             delta_G_bulk,
    float             kT,
    PPI_Score        *out
);

/*
 * Phase VI: PPI inhibitor site extraction.
 *
 * Identifies the minimal pharmacophore volume from a PPI interface
 * that a small molecule inhibitor must occupy to disrupt binding.
 *
 * Algorithm:
 *   1. Find hotspot residues on protein A (the "druggable" side).
 *   2. Extract the sub-volume of the interface grid centred on hotspots.
 *   3. Apply inverse masking: regions where protein B occupies are
 *      converted to "demand" fields (what a drug must present).
 *   4. Output: a NibbleGrid pocket suitable for Langevin docking with
 *      the existing nibble_trajectory() function — no new code needed.
 *
 * This is the PPI → small molecule inhibitor design pipeline.
 * The output pocket is used exactly like any other Nibble pocket.
 *
 * Args:
 *   grid_a:       NibbleGrid for protein A (the target surface)
 *   grid_b:       NibbleGrid for protein B (the partner, to be displaced)
 *   iface:        PPI_Interface with hotspot scores
 *   inhibitor_grid: pre-allocated NibbleGrid to fill with pharmacophore
 *   min_hotspot_weight: only include hotspots above this weight (0.3 rec.)
 *
 * Returns:
 *   Number of hotspot voxels included in inhibitor pharmacophore.
 */
int nibble_load_ppi_inhibitor_site(
    const NibbleGrid *grid_a,
    const NibbleGrid *grid_b,
    const PPI_Interface *iface,
    NibbleGrid       *inhibitor_grid,
    float             min_hotspot_weight
);

/*
 * Utility: print PPI_Score summary to stdout.
 */
void nibble_ppi_print_score(const PPI_Score *score);

/*
 * Utility: print PPI_Interface summary to stdout.
 */
void nibble_ppi_print_interface(const PPI_Interface *iface);

#endif /* NIBBLE_PPI_H */
