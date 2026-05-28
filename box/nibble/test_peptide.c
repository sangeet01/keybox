/*
 * test_peptide.c
 * Smoke test for nibble_peptide.c
 *
 * Tests four peptide types:
 *   1. Linear coil peptide
 *   2. Alpha-helical peptide
 *   3. Beta-strand peptide
 *   4. Stapled helix (simulated constraint)
 *
 * Uses 1CRN.pdb as the target surface.
 *
 * Compile:
 *   gcc -O2 -I. test_peptide.c nibble.c nibble_tier3.c nibble_tier4.c \
 *       nibble_gaps.c nibble_ppi.c nibble_protac.c nibble_peptide.c \
 *       -lm -o test_peptide
 *
 * Run:
 *   ./test_peptide path/to/1CRN.pdb
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nibble.h"
#include "nibble_peptide.h"

int main(int argc, char *argv[]) {
    const char *pdb = (argc > 1) ? argv[1] : "1CRN.pdb";
    printf("[test_peptide] Using PDB: %s\n\n", pdb);

    int pass = 1;

    /* ---------------------------------------------------------------
     * Load protein surface grid
     * --------------------------------------------------------------- */
    printf("[test_peptide] Loading protein surface grid...\n");
    NibbleGrid *grid = nibble_create_grid(60, 60, 60, 0.5f);
    if (!grid) { fprintf(stderr, "FAIL: grid alloc\n"); return 1; }

    /* 1CRN centre ~(14, 14, 14) */
    int n_atoms = nibble_load_pdb_pocket(grid, pdb, 14.0f, 14.0f, 14.0f,
                                          20.0f, 1.5f);
    printf("[test_peptide]   Atoms loaded: %d\n\n", n_atoms);

    /* ---------------------------------------------------------------
     * Test 1: Linear coil peptide (ACDEFGHIKL)
     * --------------------------------------------------------------- */
    printf("[test_peptide] Test 1: Linear coil peptide\n");
    PEP_Peptide pep1;
    int n1 = nibble_peptide_build("ACDEFGHIKL", NULL, 0, &pep1);
    printf("[test_peptide]   Built: %d residues, SS=Coil, charge=%.1f\n",
           n1, pep1.net_charge);
    nibble_peptide_print(&pep1);

    PEP_DockingResult res1;
    float sc1 = nibble_peptide_dock(grid, &pep1, 2.0f, 0.593f, 100, &res1);
    nibble_peptide_print_result(&res1);
    printf("[test_peptide]   Score: %.4f  %s\n\n",
           sc1, sc1 >= 0.0f ? "OK" : "FAIL");
    if (sc1 < 0.0f) pass = 0;

    /* ---------------------------------------------------------------
     * Test 2: Alpha-helical peptide
     * --------------------------------------------------------------- */
    printf("[test_peptide] Test 2: Alpha-helical peptide\n");
    PEP_Peptide pep2;
    int n2 = nibble_peptide_build("ETFSDLWKLL", "HHHHHHHHHH", 0, &pep2);
    printf("[test_peptide]   Built: %d residues, SS=Helix\n", n2);
    printf("[test_peptide]   Hydrophobic moment: %.4f %s\n",
           pep2.hydrophobic_moment,
           pep2.hydrophobic_moment > 0.4f ? "(amphipathic)" : "");

    PEP_DockingResult res2;
    float sc2 = nibble_peptide_dock(grid, &pep2, 2.0f, 0.593f, 100, &res2);
    nibble_peptide_print_result(&res2);
    printf("[test_peptide]   Score: %.4f  %s\n\n",
           sc2, sc2 >= 0.0f ? "OK" : "FAIL");
    if (sc2 < 0.0f) pass = 0;

    /* ---------------------------------------------------------------
     * Test 3: Beta-strand peptide
     * --------------------------------------------------------------- */
    printf("[test_peptide] Test 3: Beta-strand peptide\n");
    PEP_Peptide pep3;
    int n3 = nibble_peptide_build("VTIYVF", "EEEEEE", 0, &pep3);
    printf("[test_peptide]   Built: %d residues, SS=Strand\n", n3);

    PEP_DockingResult res3;
    float sc3 = nibble_peptide_dock(grid, &pep3, 2.0f, 0.593f, 100, &res3);
    nibble_peptide_print_result(&res3);
    printf("[test_peptide]   Score: %.4f  %s\n\n",
           sc3, sc3 >= 0.0f ? "OK" : "FAIL");
    if (sc3 < 0.0f) pass = 0;

    /* ---------------------------------------------------------------
     * Test 4: Stapled helix (i, i+4 constraint)
     * --------------------------------------------------------------- */
    printf("[test_peptide] Test 4: Stapled helix\n");
    PEP_Peptide pep4;
    int n4 = nibble_peptide_build("ETFSDLWKLLPEN", "HHHHHHHHHHHHH", 0, &pep4);
    /* Add staple between residues 2 and 6 */
    int c_idx = nibble_peptide_add_constraint(
        &pep4, PEP_CONSTRAINT_STAPLE, 2, 6, 10.0f
    );
    printf("[test_peptide]   Built: %d residues, staple at (2,6), "
           "constraint idx=%d\n", n4, c_idx);
    printf("[test_peptide]   Stapled: %s, Cyclic: %s\n",
           pep4.is_stapled ? "yes" : "no",
           pep4.is_cyclic  ? "yes" : "no");

    PEP_DockingResult res4;
    float sc4 = nibble_peptide_dock(grid, &pep4, 2.0f, 0.593f, 100, &res4);
    nibble_peptide_print_result(&res4);
    printf("[test_peptide]   Score: %.4f  Constraint penalty: %.4f  %s\n\n",
           sc4, res4.constraint_penalty, sc4 >= 0.0f ? "OK" : "FAIL");
    if (sc4 < 0.0f) pass = 0;

    /* ---------------------------------------------------------------
     * Test 5: Groove detection
     * --------------------------------------------------------------- */
    printf("[test_peptide] Test 5: Groove detection\n");
    PEP_Groove grooves[PEP_MAX_GROOVES];
    int n_grooves = nibble_peptide_scan_grooves(grid, grooves, 6.0f);
    printf("[test_peptide]   Grooves detected: %d\n", n_grooves);
    for (int i = 0; i < n_grooves && i < 3; i++) {
        printf("[test_peptide]     Groove %d: length=%.1f Å  "
               "affinity=%.3f  H-bonds=%d  hydrophobic=%d\n",
               i, grooves[i].length, grooves[i].affinity_score,
               grooves[i].n_hbond_sites, grooves[i].n_hydrophobic_patches);
    }
    printf("\n");

    /* ---------------------------------------------------------------
     * Test 6: Cyclic peptide (RGD motif)
     * --------------------------------------------------------------- */
    printf("[test_peptide] Test 6: Cyclic peptide (RGD motif)\n");
    PEP_Peptide pep6;
    int n6 = nibble_peptide_build("CRGDFC", NULL, 1, &pep6);
    /* Disulfide between Cys 0 and Cys 5 */
    nibble_peptide_add_constraint(&pep6, PEP_CONSTRAINT_DISULFIDE, 0, 5, 15.0f);
    printf("[test_peptide]   Built: %d residues, cyclic=yes, disulfide=(0,5)\n", n6);

    PEP_DockingResult res6;
    float sc6 = nibble_peptide_dock(grid, &pep6, 2.0f, 0.593f, 100, &res6);
    nibble_peptide_print_result(&res6);
    printf("[test_peptide]   Score: %.4f  %s\n\n",
           sc6, sc6 >= 0.0f ? "OK" : "FAIL");
    if (sc6 < 0.0f) pass = 0;

    /* ---------------------------------------------------------------
     * Cleanup and verdict
     * --------------------------------------------------------------- */
    nibble_free_grid(grid);
    printf("[test_peptide] Summary:\n");
    printf("  Linear coil:   %.4f\n", sc1);
    printf("  Alpha helix:   %.4f\n", sc2);
    printf("  Beta strand:   %.4f\n", sc3);
    printf("  Stapled helix: %.4f\n", sc4);
    printf("  Cyclic RGD:    %.4f\n", sc6);
    printf("  Grooves found: %d\n\n", n_grooves);
    printf("[test_peptide] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
