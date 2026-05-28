/*
 * test_antibody.c
 * Smoke test for nibble_antibody.c
 *
 * Tests four scenarios:
 *   1. Build antibody from CDR sequences (design mode)
 *   2. Build paratope demand grid
 *   3. Scan antigen surface for epitope patches
 *   4. Full docking score with CDR-H3 trajectory
 *
 * Uses 1CRN.pdb as both antibody framework reference and antigen.
 *
 * Compile:
 *   gcc -O2 -I. test_antibody.c nibble.c nibble_tier3.c nibble_tier4.c \
 *       nibble_gaps.c nibble_ppi.c nibble_protac.c nibble_peptide.c \
 *       nibble_antibody.c -lm -o test_antibody
 *
 * Run:
 *   ./test_antibody path/to/1CRN.pdb
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nibble.h"
#include "nibble_ppi.h"
#include "nibble_peptide.h"
#include "nibble_antibody.h"

int main(int argc, char *argv[]) {
    const char *pdb = (argc > 1) ? argv[1] : "1CRN.pdb";
    printf("[test_antibody] Using PDB: %s\n\n", pdb);

    int pass = 1;

    /* ---------------------------------------------------------------
     * Test 1: Build antibody from CDR sequences
     * Using representative CDR sequences (trastuzumab-inspired)
     * --------------------------------------------------------------- */
    printf("[test_antibody] Test 1: Build from CDR sequences\n");

    const char *seqs[AB_N_CDR_LOOPS] = {
        "RASQDVNTAVA",   /* L1 */
        "SASFLYS",       /* L2 */
        "QQHYTTPPT",     /* L3 */
        "GFTFSSYW",      /* H1 */
        "IKDKSGAT",      /* H2 */
        "VFYGSKTDYFDY",  /* H3 — longest, most variable */
    };

    AB_Antibody ab;
    int n_cdr = nibble_antibody_build_from_sequences(seqs, &ab);
    printf("[test_antibody]   CDR residues built: %d\n", n_cdr);
    nibble_antibody_print(&ab);

    if (n_cdr <= 0) {
        fprintf(stderr, "[test_antibody] FAIL: build_from_sequences returned %d\n",
                n_cdr);
        pass = 0;
    }

    /* Verify all CDR loops built */
    for (int c = 0; c < AB_N_CDR_LOOPS; c++) {
        if (ab.cdrs[c].n_residues == 0) {
            printf("[test_antibody]   WARNING: CDR %s has 0 residues\n",
                   AB_CDR_NAMES[c]);
        } else {
            printf("[test_antibody]   CDR %s: %d residues  %s\n",
                   AB_CDR_NAMES[c], ab.cdrs[c].n_residues, ab.cdrs[c].sequence);
        }
    }
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 2: Build paratope demand grid
     * --------------------------------------------------------------- */
    printf("[test_antibody] Test 2: Build paratope grid\n");

    NibbleGrid *paratope = nibble_antibody_build_paratope(&ab, 60, 0.5f);
    if (!paratope) {
        fprintf(stderr, "[test_antibody] FAIL: build_paratope returned NULL\n");
        pass = 0;
    } else {
        printf("[test_antibody]   Paratope grid: %dx%dx%d @ 0.5Å  built=%d\n",
               paratope->dim_x, paratope->dim_y, paratope->dim_z,
               ab.paratope_built);
    }
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 3: Load antigen surface and scan for epitope patches
     * --------------------------------------------------------------- */
    printf("[test_antibody] Test 3: Epitope patch scanning\n");

    NibbleGrid *antigen_grid = nibble_create_grid(80, 80, 80, 0.5f);
    if (!antigen_grid) {
        fprintf(stderr, "[test_antibody] FAIL: create_grid returned NULL\n");
        nibble_antibody_free(&ab);
        return 1;
    }

    int n_atoms = nibble_load_pdb_pocket(antigen_grid, pdb,
                                          14.0f, 14.0f, 14.0f, 25.0f, 1.5f);
    printf("[test_antibody]   Antigen atoms loaded: %d\n", n_atoms);

    AB_EpitopePatch patches[AB_MAX_EPITOPE_PATCHES];
    int n_patches = nibble_antibody_scan_epitope(
        antigen_grid, paratope, patches, 0.02f
    );
    printf("[test_antibody]   Epitope patches found: %d\n", n_patches);

    for (int i = 0; i < n_patches && i < 3; i++) {
        printf("[test_antibody]     Patch %d: "
               "centre=(%.1f,%.1f,%.1f)  "
               "complementarity=%.4f  "
               "area=%.1fÅ²  "
               "hbonds=%d  hydro=%d  charged=%d\n",
               i,
               patches[i].cx, patches[i].cy, patches[i].cz,
               patches[i].complementarity,
               patches[i].area,
               patches[i].n_hbond_sites,
               patches[i].n_hydrophobic,
               patches[i].n_charged);
    }
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 4: CDR-H3 flexible trajectory
     * --------------------------------------------------------------- */
    printf("[test_antibody] Test 4: CDR-H3 flexible trajectory\n");

    float h3_score = 0.0f;
    AB_EpitopePatch *best_patch = (n_patches > 0) ? &patches[0] : NULL;
    float h3_result = nibble_antibody_h3_trajectory(
        antigen_grid, &ab, best_patch, 100, 0.593f, &h3_score
    );
    printf("[test_antibody]   CDR-H3 trajectory score: %.4f\n", h3_result);
    printf("[test_antibody]   H3 length: %d residues\n",
           ab.cdrs[AB_CDR_H3].n_residues);
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 5: Full composite docking score
     * --------------------------------------------------------------- */
    printf("[test_antibody] Test 5: Full composite docking score\n");

    AB_DockingResult dock_result;
    float composite = nibble_antibody_score(
        antigen_grid, &ab, 100, 2.0f, 0.593f, &dock_result
    );
    nibble_antibody_print_result(&dock_result);

    static const char *kd_labels[] = {
        "nM-class", "µM-class", "weak/non-binder"
    };
    printf("[test_antibody]   Composite: %.4f  [%s]\n",
           composite, kd_labels[dock_result.kd_class]);
    printf("[test_antibody]   Epitope patches: %d\n",
           dock_result.n_epitope_patches);

    if (composite < 0.0f) pass = 0;
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 6: Per-CDR contribution analysis
     * --------------------------------------------------------------- */
    printf("[test_antibody] Test 6: CDR contribution breakdown\n");
    float weighted_sum = 0.0f;
    for (int c = 0; c < AB_N_CDR_LOOPS; c++) {
        printf("[test_antibody]   %s (w=%.2f): score=%.4f  "
               "contribution=%.4f\n",
               AB_CDR_NAMES[c],
               AB_CDR_WEIGHTS[c],
               dock_result.cdr_scores[c],
               dock_result.cdr_scores[c] * AB_CDR_WEIGHTS[c]);
        weighted_sum += dock_result.cdr_scores[c] * AB_CDR_WEIGHTS[c];
    }
    printf("[test_antibody]   CDR weighted sum: %.4f\n\n", weighted_sum);

    /* ---------------------------------------------------------------
     * Cleanup and verdict
     * --------------------------------------------------------------- */
    nibble_antibody_free(&ab);
    nibble_free_grid(antigen_grid);

    printf("[test_antibody] Summary:\n");
    printf("  CDR residues built:  %d\n", n_cdr);
    printf("  Paratope grid:       %s\n", paratope ? "built" : "FAILED");
    printf("  Epitope patches:     %d\n", n_patches);
    printf("  H3 trajectory:       %.4f\n", h3_result);
    printf("  Composite score:     %.4f  [%s]\n",
           composite, kd_labels[dock_result.kd_class]);
    printf("\n[test_antibody] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
