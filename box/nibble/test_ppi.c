/*
 * test_ppi.c
 * Minimal smoke test for nibble_ppi.c using the bundled 1CRN.pdb.
 *
 * 1CRN (Crambin) is a single-chain protein but serves as a self-interface
 * test (chain A vs chain A) to verify all six phases run without error.
 * For a real PPI test, replace pdb_b and chain_b with a second structure.
 *
 * Compile and run:
 *   gcc -O2 -I. test_ppi.c nibble.c nibble_tier3.c nibble_tier4.c nibble_gaps.c nibble_ppi.c -lm -o test_ppi
 *   ./test_ppi path/to/1CRN.pdb
 *
 * Expected output (approximate):
 *   [nibble_ppi] PPI Interface Summary
 *     Residues A/B: N / N
 *     ...
 *   [nibble_ppi] PPI Score Summary
 *     Composite score: 0.XXXX
 *     Shape complementarity: 0.XXXX
 *     ...
 *   [test_ppi] All phases completed. PASS
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nibble.h"
#include "nibble_ppi.h"

int main(int argc, char *argv[]) {
    const char *pdb_path = (argc > 1) ? argv[1] : "1CRN.pdb";

    printf("[test_ppi] Using PDB: %s\n\n", pdb_path);

    /* ---------------------------------------------------------------
     * Phase I: detect interface
     * For single-chain test: use the same file twice, chain 0 (any).
     * This tests the O(n²) atom scan and residue centroid logic.
     * --------------------------------------------------------------- */
    PPI_Interface iface;
    memset(&iface, 0, sizeof(iface));

    printf("[test_ppi] Phase I: detecting interface...\n");
    int n_pairs = nibble_detect_interface(
        pdb_path, pdb_path,
        0, 0,           /* chain 0 = any */
        6.0f,
        &iface
    );
    printf("[test_ppi]   n_pairs=%d\n", n_pairs);
    nibble_ppi_print_interface(&iface);

    if (n_pairs < 0) {
        fprintf(stderr, "[test_ppi] FAIL: interface detection returned error.\n");
        return 1;
    }

    /* ---------------------------------------------------------------
     * Phase II: load surface patches
     * --------------------------------------------------------------- */
    printf("\n[test_ppi] Phase II: loading surface patches...\n");

    /* Grid: 80×80×80 at 0.5Å resolution */
    NibbleGrid *grid_a = nibble_create_grid(80, 80, 80, 0.5f);
    NibbleGrid *grid_b = nibble_create_grid(80, 80, 80, 0.5f);

    if (!grid_a || !grid_b) {
        fprintf(stderr, "[test_ppi] FAIL: nibble_create_grid returned NULL.\n");
        return 1;
    }

    int n_a = nibble_load_ppi_surface(grid_a, pdb_path, &iface, 1, 1.5f);
    int n_b = nibble_load_ppi_surface(grid_b, pdb_path, &iface, 0, 1.5f);
    printf("[test_ppi]   atoms projected: A=%d  B=%d\n", n_a, n_b);

    if (n_a <= 0 || n_b <= 0) {
        fprintf(stderr, "[test_ppi] FAIL: surface loading projected 0 atoms.\n");
        nibble_free_grid(grid_a);
        nibble_free_grid(grid_b);
        return 1;
    }

    /* ---------------------------------------------------------------
     * Phase III: shape complementarity
     * --------------------------------------------------------------- */
    printf("\n[test_ppi] Phase III: shape complementarity...\n");
    float sc = nibble_sc_score(grid_a, grid_b);
    printf("[test_ppi]   Sc = %.4f\n", sc);
    printf("[test_ppi]   Interpretation: %s\n",
           sc >= 0.70f ? "well-complementary (Sc >= 0.70)" :
           sc >= 0.50f ? "moderate (0.50 <= Sc < 0.70)"   :
                         "weak (Sc < 0.50)");

    /* ---------------------------------------------------------------
     * Phase IV: hotspot scoring
     * --------------------------------------------------------------- */
    printf("\n[test_ppi] Phase IV: hotspot scoring...\n");
    int n_hot = nibble_score_hotspots(grid_a, grid_b, &iface);
    printf("[test_ppi]   hotspot residues: %d\n", n_hot);
    printf("[test_ppi]   mean hotspot weight: %.4f\n",
           iface.mean_hotspot_weight);

    /* Print top 5 hotspots on protein A side */
    printf("[test_ppi]   top hotspots (protein A):\n");
    int printed = 0;
    for (int r = 0; r < iface.n_residues_a && printed < 5; r++) {
        if (iface.residues_a[r].is_hotspot) {
            printf("    %s%d (chain %c)  weight=%.3f\n",
                   iface.residues_a[r].resname,
                   iface.residues_a[r].resnum,
                   iface.residues_a[r].chain ? iface.residues_a[r].chain : '?',
                   iface.residues_a[r].hotspot_weight);
            printed++;
        }
    }

    /* ---------------------------------------------------------------
     * Phase V: full PPI affinity
     * --------------------------------------------------------------- */
    printf("\n[test_ppi] Phase V: full PPI affinity...\n");
    PPI_Score score;
    memset(&score, 0, sizeof(score));

    float composite = nibble_ppi_affinity(
        grid_a, grid_b, &iface,
        2.0f,   /* delta_G_bulk kcal/mol */
        0.593f, /* kT at 298K */
        &score
    );

    nibble_ppi_print_score(&score);
    printf("[test_ppi]   composite returned: %.4f\n", composite);

    /* ---------------------------------------------------------------
     * Phase VI: inhibitor pharmacophore extraction
     * --------------------------------------------------------------- */
    printf("\n[test_ppi] Phase VI: inhibitor pharmacophore extraction...\n");
    NibbleGrid *inhib = nibble_create_grid(80, 80, 80, 0.5f);
    if (!inhib) {
        fprintf(stderr, "[test_ppi] FAIL: nibble_create_grid for inhibitor returned NULL.\n");
        nibble_free_grid(grid_a);
        nibble_free_grid(grid_b);
        return 1;
    }

    int n_vox = nibble_load_ppi_inhibitor_site(
        grid_a, grid_b, &iface, inhib, 0.3f
    );
    printf("[test_ppi]   pharmacophore voxels: %d\n", n_vox);

    if (n_vox > 0) {
        printf("[test_ppi]   pharmacophore ready for Langevin docking.\n");
    } else {
        printf("[test_ppi]   note: 0 voxels — try lowering min_hotspot_weight.\n");
    }

    /* ---------------------------------------------------------------
     * Cleanup
     * --------------------------------------------------------------- */
    nibble_free_grid(grid_a);
    nibble_free_grid(grid_b);
    nibble_free_grid(inhib);

    /* ---------------------------------------------------------------
     * Final verdict
     * --------------------------------------------------------------- */
    int pass = (n_pairs > 0 && n_a > 0 && n_b > 0 && composite >= 0.0f);
    printf("\n[test_ppi] %s\n", pass ? "All phases completed. PASS" : "FAIL");
    return pass ? 0 : 1;
}
