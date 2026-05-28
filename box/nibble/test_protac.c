/*
 * test_protac.c
 * Smoke test for nibble_protac.c
 *
 * Uses the same PDB as both target and E3 (1CRN.pdb) to test all
 * six components without requiring a real PROTAC structure.
 * Two offset pocket centres simulate the target/E3 separation.
 *
 * Compile:
 *   gcc -O2 -I. test_protac.c nibble.c nibble_tier3.c nibble_tier4.c \
 *       nibble_gaps.c nibble_ppi.c nibble_protac.c -lm -o test_protac
 *
 * Run:
 *   ./test_protac path/to/1CRN.pdb
 *
 * Expected output:
 *   [nibble_protac] PROTAC Ternary Score Summary
 *     Composite score: 0.XXXX
 *     Degradation efficiency: 0.XXXX
 *     ...
 *   [test_protac] PASS
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nibble.h"
#include "nibble_ppi.h"
#include "nibble_protac.h"

/* -----------------------------------------------------------------------
 * Build a minimal test PROTAC:
 *   warhead_a: 15 atoms (atoms 0–14)  → target binder
 *   linker:    10 atoms (atoms 15–24) → chain bridge
 *   warhead_b: 12 atoms (atoms 25–36) → E3 binder
 * All atoms placed at pocket centres ± small offsets.
 * Channel values: 0.5 for all (neutral test values).
 * ----------------------------------------------------------------------- */
#define N_WA    15
#define N_LNK   10
#define N_WB    12
#define N_TOTAL (N_WA + N_LNK + N_WB)

static void _build_test_protac(
    float target_cx, float target_cy, float target_cz,
    float e3_cx,     float e3_cy,     float e3_cz,
    float *coords_out,   /* [N_TOTAL * 3] */
    float *chvals_out)   /* [N_TOTAL * N_CHANNELS] */
{
    /* Warhead A: cluster around target centre */
    for (int i = 0; i < N_WA; i++) {
        float r = (float)i * 0.8f;
        float a = (float)i * 0.5f;
        coords_out[i*3+0] = target_cx + r * cosf(a);
        coords_out[i*3+1] = target_cy + r * sinf(a);
        coords_out[i*3+2] = target_cz;
    }

    /* Linker: straight line from target to E3 */
    for (int i = 0; i < N_LNK; i++) {
        float t = (float)(i + 1) / (float)(N_LNK + 1);
        coords_out[(N_WA + i)*3+0] = target_cx + t * (e3_cx - target_cx);
        coords_out[(N_WA + i)*3+1] = target_cy + t * (e3_cy - target_cy);
        coords_out[(N_WA + i)*3+2] = target_cz + t * (e3_cz - target_cz);
    }

    /* Warhead B: cluster around E3 centre */
    for (int i = 0; i < N_WB; i++) {
        float r = (float)i * 0.8f;
        float a = (float)i * 0.5f;
        coords_out[(N_WA + N_LNK + i)*3+0] = e3_cx + r * cosf(a);
        coords_out[(N_WA + N_LNK + i)*3+1] = e3_cy + r * sinf(a);
        coords_out[(N_WA + N_LNK + i)*3+2] = e3_cz;
    }

    /* Neutral channel values */
    for (int i = 0; i < N_TOTAL * N_CHANNELS; i++) {
        chvals_out[i] = 0.5f;
    }
}

int main(int argc, char *argv[]) {
    const char *pdb = (argc > 1) ? argv[1] : "1CRN.pdb";
    printf("[test_protac] Using PDB: %s\n\n", pdb);

    /* ---------------------------------------------------------------
     * Two pocket centres: offset to simulate target / E3 separation.
     * 1CRN has a small extent (~20Å) so 12Å separation is realistic.
     * --------------------------------------------------------------- */
    float target_cx = 10.0f, target_cy = 10.0f, target_cz = 10.0f;
    float e3_cx     = 22.0f, e3_cy     = 10.0f, e3_cz     = 10.0f;
    float radius    = 8.0f,  blur      = 1.2f;

    /* ---------------------------------------------------------------
     * Load pocket grids
     * --------------------------------------------------------------- */
    printf("[test_protac] Loading pocket grids...\n");
    NibbleGrid *grid_target = nibble_create_grid(60, 60, 60, 0.5f);
    NibbleGrid *grid_e3     = nibble_create_grid(60, 60, 60, 0.5f);
    if (!grid_target || !grid_e3) {
        fprintf(stderr, "[test_protac] FAIL: nibble_create_grid returned NULL.\n");
        return 1;
    }

    int atoms_t = nibble_load_pdb_pocket(grid_target, pdb,
        target_cx, target_cy, target_cz, radius, blur);
    int atoms_e = nibble_load_pdb_pocket(grid_e3, pdb,
        e3_cx, e3_cy, e3_cz, radius, blur);
    printf("[test_protac]   Target pocket atoms: %d\n", atoms_t);
    printf("[test_protac]   E3 pocket atoms:     %d\n", atoms_e);

    /* ---------------------------------------------------------------
     * Build test PROTAC
     * --------------------------------------------------------------- */
    printf("\n[test_protac] Building test PROTAC (%d atoms: %d|%d|%d)...\n",
           N_TOTAL, N_WA, N_LNK, N_WB);

    float coords[N_TOTAL * 3];
    float chvals[N_TOTAL * N_CHANNELS];
    _build_test_protac(
        target_cx, target_cy, target_cz,
        e3_cx, e3_cy, e3_cz,
        coords, chvals
    );

    PROTAC_Molecule *protac = nibble_protac_create(
        N_TOTAL, coords, chvals,
        N_WA,           /* split_a */
        N_WA + N_LNK    /* split_b */
    );
    if (!protac) {
        fprintf(stderr, "[test_protac] FAIL: nibble_protac_create returned NULL.\n");
        nibble_free_grid(grid_target);
        nibble_free_grid(grid_e3);
        return 1;
    }
    printf("[test_protac]   Linker: %d atoms, contour=%.1f Å, flex=%.2f\n",
           protac->linker.n_atoms,
           protac->linker.contour_length,
           protac->linker.flexibility);

    /* ---------------------------------------------------------------
     * Build ternary geometry
     * --------------------------------------------------------------- */
    PROTAC_TernaryGeometry geom;
    memset(&geom, 0, sizeof(geom));
    geom.target_cx = target_cx;  geom.target_cy = target_cy;  geom.target_cz = target_cz;
    geom.e3_cx     = e3_cx;      geom.e3_cy     = e3_cy;      geom.e3_cz     = e3_cz;
    float dx = e3_cx - target_cx, dy = e3_cy - target_cy, dz = e3_cz - target_cz;
    geom.pocket_separation  = sqrtf(dx*dx + dy*dy + dz*dz);
    geom.pocket_sep_x       = dx;
    geom.pocket_sep_y       = dy;
    geom.pocket_sep_z       = dz;
    /* Warhead radii ~sqrt(n)*1.5 */
    float ra = sqrtf((float)N_WA)  * 1.5f;
    float rb = sqrtf((float)N_WB)  * 1.5f;
    geom.required_linker_ete = fmaxf(0.0f, geom.pocket_separation - ra - rb);
    printf("[test_protac]   Pocket separation: %.1f Å\n", geom.pocket_separation);
    printf("[test_protac]   Required linker ETE: %.1f Å\n", geom.required_linker_ete);

    /* ---------------------------------------------------------------
     * Test linker penalty
     * --------------------------------------------------------------- */
    printf("\n[test_protac] Linker geometry test...\n");
    float ete_out = 0.0f;
    float lp = nibble_protac_linker_penalty(
        &protac->linker, geom.required_linker_ete, &ete_out
    );
    printf("[test_protac]   Linker penalty: %.4f  (ETE RMS=%.1f Å)\n", lp, ete_out);

    /* ---------------------------------------------------------------
     * Test cooperativity
     * --------------------------------------------------------------- */
    printf("\n[test_protac] Cooperativity test...\n");
    float alpha = nibble_protac_cooperativity(0.6f, 0.6f, 0.4f, 0.593f);
    printf("[test_protac]   Alpha (α): %.4f  %s\n",
           alpha, alpha > 1.0f ? "(cooperative)" : "(anticooperative)");

    /* ---------------------------------------------------------------
     * Test neo-PPI
     * --------------------------------------------------------------- */
    printf("\n[test_protac] Neo-PPI test...\n");
    PPI_Interface neo_iface;
    memset(&neo_iface, 0, sizeof(neo_iface));
    float neo = nibble_protac_neo_ppi(
        grid_target, grid_e3, protac, &neo_iface, 0.593f
    );
    printf("[test_protac]   Neo-PPI score: %.4f\n", neo);

    /* ---------------------------------------------------------------
     * Full ternary score (static)
     * --------------------------------------------------------------- */
    printf("\n[test_protac] Full ternary score (static)...\n");
    PROTAC_TernaryScore score;
    memset(&score, 0, sizeof(score));
    float composite = nibble_protac_score(
        grid_target, grid_e3, protac, &geom,
        2.0f, 0.593f, &score
    );
    nibble_protac_print_score(&score);

    /* ---------------------------------------------------------------
     * Trajectory optimisation
     * --------------------------------------------------------------- */
    printf("\n[test_protac] Ternary Langevin trajectory (100 steps)...\n");
    PROTAC_TernaryScore traj_score;
    memset(&traj_score, 0, sizeof(traj_score));
    float traj_composite = nibble_protac_trajectory(
        grid_target, grid_e3, protac, &geom,
        100,    /* n_steps */
        0.002f, /* dt */
        1.0f,   /* gamma */
        0.593f, /* kT */
        &traj_score
    );
    printf("[test_protac]   Trajectory best composite: %.4f\n", traj_composite);
    printf("[test_protac]   Degradation efficiency:    %.4f\n",
           traj_score.deg_efficiency);
    printf("[test_protac]   Hook effect:               %s\n",
           traj_score.hook_effect ? "YES (predicted)" : "No");

    /* ---------------------------------------------------------------
     * Cleanup and verdict
     * --------------------------------------------------------------- */
    nibble_protac_free(protac);
    nibble_free_grid(grid_target);
    nibble_free_grid(grid_e3);

    int pass = (composite >= 0.0f && traj_composite >= 0.0f && lp >= 0.0f);
    printf("\n[test_protac] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
