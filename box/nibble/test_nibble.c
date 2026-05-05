#include "nibble.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

/* 
 * test_nibble.c — Verification of the Negative Space Matrix Multiplication.
 */

void test_pdb_loading() {
    printf("Testing PDB loading (1CRN.pdb)...\n");
    
    // Create a 30x30x30 grid for the Crambin protein
    // Range 10A around center (17, 13, 11)
    NibbleGrid *pocket = nibble_create_grid(30, 30, 30, 0.8f);
    
    int atoms = nibble_load_pdb_pocket(pocket, "1CRN.pdb", 17.0f, 13.0f, 11.0f, 10.0f);
    printf("Loaded %d atoms into the negative field.\n", atoms);
    assert(atoms > 0);
    
    // Check if some "Demand" was projected
    float total_demand = 0.0f;
    for (size_t i = 0; i < (size_t)30*30*30*N_CHANNELS; ++i) {
        total_demand += fabsf(pocket->data[i]);
    }
    printf("Total integrated demand in pocket: %.2f\n", total_demand);
    assert(total_demand > 0.0f);
    
    // Test a "Mock Drug" that matches the pocket center
    NibbleGrid *drug = nibble_create_grid(30, 30, 30, 0.8f);
    
    // Center of drug grid corresponds to center of pocket range
    // Let's project a Lipophilic atom into the center of the pocket
    float drug_atom[N_CHANNELS] = {0};
    drug_atom[CH_STERIC_DEMAND] = 1.0f;
    drug_atom[CH_LIPO_DEMAND]   = 1.0f;
    
    // Center of 30x30x30 range (10A radius) is 10.0f, 10.0f, 10.0f relative
    nibble_project_atom(drug, 10.0f, 10.0f, 10.0f, 1.5f, drug_atom);
    
    float affinity = nibble_compute_affinity(pocket, drug);
    printf("Mock Drug Affinity against 1CRN pocket: %.2f\n", affinity);
    
    nibble_free_grid(pocket);
    nibble_free_grid(drug);
    printf("PDB loading and affinity: SUCCESS\n\n");
}

int main() {
    printf("=== NIBBLE ENGINE PDB VERIFICATION ===\n");
    test_pdb_loading();
    printf("All nibble PDB tests passed.\n");
    return 0;
}
