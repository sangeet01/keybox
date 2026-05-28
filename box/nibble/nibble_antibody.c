/*
 * nibble_antibody.c
 * Antibody CDR-Antigen Docking Extension for the Nibble Voxel Engine.
 *
 * Implements:
 *   nibble_antibody_parse()               — PDB → CDR loops + Fv framework
 *   nibble_antibody_build_from_sequences()— sequences → CDR loops
 *   nibble_antibody_build_paratope()      — 6 CDRs → paratope demand grid
 *   nibble_antibody_scan_epitope()        — antigen surface → epitope patches
 *   nibble_antibody_h3_trajectory()       — CDR-H3 flexible Langevin
 *   nibble_antibody_score()               — full composite docking score
 *   nibble_antibody_free()                — cleanup
 *   nibble_antibody_print_result()        — diagnostic output
 *   nibble_antibody_print()               — CDR summary
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#include "nibble_antibody.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Internal constants
 * ----------------------------------------------------------------------- */

/* Composite score weights */
#define _WAB_SC     0.30f   /* shape complementarity                      */
#define _WAB_H3     0.35f   /* CDR-H3 affinity (dominant)                 */
#define _WAB_CDR    0.20f   /* canonical CDR contributions                */
#define _WAB_BSA    0.10f   /* buried surface area                        */
#define _WAB_VHVL  -0.05f   /* VH/VL orientation penalty                  */

/* Kd class thresholds */
#define _KD_NM_THRESHOLD  0.60f  /* composite >= 0.60 → nM-class          */
#define _KD_UM_THRESHOLD  0.35f  /* composite >= 0.35 → µM-class          */

/*
 * Kabat/Chothia CDR position ranges for VH and VL.
 * Using Chothia numbering (most commonly used in PDB files).
 * Format: {chain_type, start_resnum, end_resnum_min, end_resnum_max}
 * end_resnum_max accommodates CDR length variability.
 *
 * Chain type: 0 = VL (light), 1 = VH (heavy)
 */
typedef struct { int chain; int start; int end_min; int end_max; } _CDRRange;

static const _CDRRange _CHOTHIA_CDR_RANGES[AB_N_CDR_LOOPS] = {
    {0, 24,  34,  34},  /* L1: positions 24–34 (variable length)        */
    {0, 50,  56,  56},  /* L2: positions 50–56 (canonical 7 residues)   */
    {0, 89,  97,  97},  /* L3: positions 89–97 (variable)               */
    {1, 26,  32,  32},  /* H1: positions 26–32 (variable)               */
    {1, 52,  56,  56},  /* H2: positions 52–56 (variable)               */
    {1, 95, 102, 102},  /* H3: positions 95–102 (most variable)         */
};

/* -----------------------------------------------------------------------
 * Internal helpers
 * ----------------------------------------------------------------------- */

static inline float _clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float _dist2(
    float ax, float ay, float az,
    float bx, float by, float bz)
{
    float dx=ax-bx, dy=ay-by, dz=az-bz;
    return dx*dx + dy*dy + dz*dz;
}

/* One-letter to three-letter AA */
static const char *_one_to_three(char aa) {
    static const char *table[][2] = {
        {"A","ALA"},{"R","ARG"},{"N","ASN"},{"D","ASP"},{"C","CYS"},
        {"Q","GLN"},{"E","GLU"},{"G","GLY"},{"H","HIS"},{"I","ILE"},
        {"L","LEU"},{"K","LYS"},{"M","MET"},{"F","PHE"},{"P","PRO"},
        {"S","SER"},{"T","THR"},{"W","TRP"},{"Y","TYR"},{"V","VAL"},
        {NULL,NULL}
    };
    for (int i = 0; table[i][0]; i++) {
        if (table[i][0][0] == aa) return table[i][1];
    }
    return "ALA";
}

/* Three-letter to one-letter AA */
static char _three_to_one(const char *three) {
    static const char *table[][2] = {
        {"ALA","A"},{"ARG","R"},{"ASN","N"},{"ASP","D"},{"CYS","C"},
        {"GLN","Q"},{"GLU","E"},{"GLY","G"},{"HIS","H"},{"ILE","I"},
        {"LEU","L"},{"LYS","K"},{"MET","M"},{"PHE","F"},{"PRO","P"},
        {"SER","S"},{"THR","T"},{"TRP","W"},{"TYR","Y"},{"VAL","V"},
        {NULL,NULL}
    };
    for (int i = 0; table[i][0]; i++) {
        if (strncmp(table[i][0], three, 3) == 0) return table[i][1][0];
    }
    return 'A';
}

/* -----------------------------------------------------------------------
 * Internal: raw PDB atom for antibody parsing
 * ----------------------------------------------------------------------- */
typedef struct {
    char  chain;
    int   resnum;
    char  resname[4];
    char  atomname[5];
    float x, y, z;
    int   is_ca;
} _AB_Atom;

static int _parse_pdb_ca(
    const char *path, char chain_filter,
    _AB_Atom *atoms, int max_atoms)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[256];
    int n = 0;
    while (fgets(line, sizeof(line), fp) && n < max_atoms) {
        if (strncmp(line, "ATOM  ", 6) != 0) continue;
        if ((int)strlen(line) < 54) continue;
        char chain = line[21];
        if (chain_filter && chain != chain_filter) continue;
        /* Only Cα atoms */
        char atomname[5] = {0};
        memcpy(atomname, line+12, 4);
        int is_ca = (strncmp(atomname, " CA ", 4) == 0 ||
                     strncmp(atomname, "CA  ", 4) == 0);
        _AB_Atom *a = &atoms[n];
        a->chain   = chain;
        a->is_ca   = is_ca;
        char rnum[5]={0}; memcpy(rnum, line+22, 4);
        a->resnum  = atoi(rnum);
        memcpy(a->resname, line+17, 3); a->resname[3] = 0;
        memcpy(a->atomname, atomname, 4); a->atomname[4] = 0;
        char xb[9]={0},yb[9]={0},zb[9]={0};
        memcpy(xb,line+30,8); memcpy(yb,line+38,8); memcpy(zb,line+46,8);
        a->x = (float)atof(xb);
        a->y = (float)atof(yb);
        a->z = (float)atof(zb);
        n++;
    }
    fclose(fp);
    return n;
}

/* -----------------------------------------------------------------------
 * Internal: assign CDR residues from atom list
 * ----------------------------------------------------------------------- */
static int _assign_cdr_from_atoms(
    _AB_Atom *atoms, int n_atoms,
    int cdr_idx, char chain,
    AB_CDRLoop *cdr)
{
    const _CDRRange *range = &_CHOTHIA_CDR_RANGES[cdr_idx];
    cdr->cdr_idx    = cdr_idx;
    cdr->n_residues = 0;
    cdr->is_h3      = (cdr_idx == AB_CDR_H3);

    int seq_len = 0;
    for (int i = 0; i < n_atoms; i++) {
        if (!atoms[i].is_ca) continue;
        if (atoms[i].chain != chain) continue;
        int rn = atoms[i].resnum;
        if (rn < range->start || rn > range->end_max) continue;
        int ri = cdr->n_residues;
        if (ri >= AB_MAX_CDR_LEN) break;
        cdr->ca_x[ri] = atoms[i].x;
        cdr->ca_y[ri] = atoms[i].y;
        cdr->ca_z[ri] = atoms[i].z;
        /* One-letter from resname */
        char one = _three_to_one(atoms[i].resname);
        cdr->sequence[seq_len++] = one;
        cdr->n_residues++;
    }
    cdr->sequence[seq_len] = '\0';

    /* Anchor residues: immediately before and after CDR in PDB */
    int anchor_n_set = 0, anchor_c_set = 0;
    for (int i = 0; i < n_atoms; i++) {
        if (!atoms[i].is_ca || atoms[i].chain != chain) continue;
        if (atoms[i].resnum == range->start - 1 && !anchor_n_set) {
            cdr->anchor_n_x = atoms[i].x;
            cdr->anchor_n_y = atoms[i].y;
            cdr->anchor_n_z = atoms[i].z;
            anchor_n_set = 1;
        }
        if (atoms[i].resnum == range->end_max + 1 && !anchor_c_set) {
            cdr->anchor_c_x = atoms[i].x;
            cdr->anchor_c_y = atoms[i].y;
            cdr->anchor_c_z = atoms[i].z;
            anchor_c_set = 1;
        }
    }

    /* Build PEP_Peptide for this CDR loop */
    if (cdr->n_residues > 0) {
        /* All CDRs default to coil; H3 uses coil; canonical CDRs could use H */
        char ss_buf[AB_MAX_CDR_LEN + 1];
        for (int i = 0; i < cdr->n_residues; i++) ss_buf[i] = 'C';
        ss_buf[cdr->n_residues] = '\0';
        nibble_peptide_build(cdr->sequence, ss_buf, 0, &cdr->pep);
        /* Override Cα positions from PDB */
        for (int i = 0; i < cdr->n_residues && i < cdr->pep.n_residues; i++) {
            cdr->pep.residues[i].ca_x = cdr->ca_x[i];
            cdr->pep.residues[i].ca_y = cdr->ca_y[i];
            cdr->pep.residues[i].ca_z = cdr->ca_z[i];
        }
    }

    return cdr->n_residues;
}

/* -----------------------------------------------------------------------
 * nibble_antibody_parse
 * ----------------------------------------------------------------------- */
int nibble_antibody_parse(
    const char  *pdb_path,
    char         vh_chain,
    char         vl_chain,
    AB_Antibody *out)
{
    if (!pdb_path || !out) return -1;
    memset(out, 0, sizeof(AB_Antibody));
    strncpy(out->name, pdb_path, 31);

    _AB_Atom *atoms = (_AB_Atom*)malloc(50000 * sizeof(_AB_Atom));
    if (!atoms) return -1;

    /* Parse VH chain */
    int n_vh = _parse_pdb_ca(pdb_path, vh_chain, atoms, 50000);
    if (n_vh < 0) { free(atoms); return -1; }

    /* Assign VH CDRs (H1, H2, H3) */
    for (int c = AB_CDR_H1; c <= AB_CDR_H3; c++) {
        _assign_cdr_from_atoms(atoms, n_vh, c, vh_chain, &out->cdrs[c]);
    }

    /* VH framework COM */
    float sum_x=0, sum_y=0, sum_z=0; int cnt=0;
    for (int i = 0; i < n_vh; i++) {
        if (!atoms[i].is_ca) continue;
        int rn = atoms[i].resnum;
        /* Framework = outside all CDR ranges */
        int in_cdr = 0;
        for (int c = AB_CDR_H1; c <= AB_CDR_H3; c++) {
            const _CDRRange *r = &_CHOTHIA_CDR_RANGES[c];
            if (rn >= r->start && rn <= r->end_max) { in_cdr = 1; break; }
        }
        if (!in_cdr) {
            sum_x += atoms[i].x; sum_y += atoms[i].y;
            sum_z += atoms[i].z; cnt++;
        }
    }
    if (cnt > 0) {
        out->framework.vh_cx = sum_x / cnt;
        out->framework.vh_cy = sum_y / cnt;
        out->framework.vh_cz = sum_z / cnt;
    }

    /* Parse VL chain */
    int n_vl = _parse_pdb_ca(pdb_path, vl_chain, atoms, 50000);
    if (n_vl > 0) {
        for (int c = AB_CDR_L1; c <= AB_CDR_L3; c++) {
            _assign_cdr_from_atoms(atoms, n_vl, c, vl_chain, &out->cdrs[c]);
        }
        /* VL framework COM */
        sum_x = sum_y = sum_z = 0; cnt = 0;
        for (int i = 0; i < n_vl; i++) {
            if (!atoms[i].is_ca) continue;
            int rn = atoms[i].resnum;
            int in_cdr = 0;
            for (int c = AB_CDR_L1; c <= AB_CDR_L3; c++) {
                const _CDRRange *r = &_CHOTHIA_CDR_RANGES[c];
                if (rn >= r->start && rn <= r->end_max) { in_cdr=1; break; }
            }
            if (!in_cdr) {
                sum_x += atoms[i].x; sum_y += atoms[i].y;
                sum_z += atoms[i].z; cnt++;
            }
        }
        if (cnt > 0) {
            out->framework.vl_cx = sum_x / cnt;
            out->framework.vl_cy = sum_y / cnt;
            out->framework.vl_cz = sum_z / cnt;
        }
    }

    /* Compute VH-VL elbow angle */
    float dx = out->framework.vh_cx - out->framework.vl_cx;
    float dy = out->framework.vh_cy - out->framework.vl_cy;
    float dz = out->framework.vh_cz - out->framework.vl_cz;
    float d  = sqrtf(dx*dx + dy*dy + dz*dz);
    out->framework.elbow_angle = d > 1e-6f ? acosf(_clampf(dz/d, -1.f, 1.f)) * 57.296f : 160.0f;
    out->framework.qw = 1.0f;
    out->framework.mass = 25000.0f; /* ~25 kDa Fv */

    free(atoms);

    int total_cdr_res = 0;
    for (int c = 0; c < AB_N_CDR_LOOPS; c++)
        total_cdr_res += out->cdrs[c].n_residues;

    return total_cdr_res;
}

/* -----------------------------------------------------------------------
 * nibble_antibody_build_from_sequences
 * ----------------------------------------------------------------------- */
int nibble_antibody_build_from_sequences(
    const char  *sequences[AB_N_CDR_LOOPS],
    AB_Antibody *out)
{
    if (!out) return -1;
    memset(out, 0, sizeof(AB_Antibody));
    strncpy(out->name, "designed_ab", 31);

    int total = 0;
    for (int c = 0; c < AB_N_CDR_LOOPS; c++) {
        AB_CDRLoop *cdr = &out->cdrs[c];
        cdr->cdr_idx = c;
        cdr->is_h3   = (c == AB_CDR_H3);

        if (!sequences[c] || sequences[c][0] == '\0') {
            cdr->n_residues = 0;
            continue;
        }

        int n = (int)strlen(sequences[c]);
        if (n > AB_MAX_CDR_LEN) n = AB_MAX_CDR_LEN;
        cdr->n_residues = n;
        memcpy(cdr->sequence, sequences[c], n);
        cdr->sequence[n] = '\0';

        /* Build PEP_Peptide */
        char ss_buf[AB_MAX_CDR_LEN + 1];
        /* H3 and L3 often have turn/coil; others often short helices */
        char ss_char = (c == AB_CDR_H3 || c == AB_CDR_L3) ? 'C' : 'H';
        for (int i = 0; i < n; i++) ss_buf[i] = ss_char;
        ss_buf[n] = '\0';
        nibble_peptide_build(cdr->sequence, ss_buf, 0, &cdr->pep);

        /* Place CDR in a notional paratope position */
        /* Canonical placement: loops spread ~15Å apart in a circle */
        float angle = (float)c * 2.0f * 3.14159f / (float)AB_N_CDR_LOOPS;
        float r     = 15.0f;
        float base_z= (float)c * 2.0f;  /* slight z offset per CDR */
        for (int i = 0; i < cdr->pep.n_residues; i++) {
            float t = (float)i / (float)(cdr->pep.n_residues + 1);
            cdr->pep.residues[i].ca_x = r * cosf(angle) + t * 2.0f;
            cdr->pep.residues[i].ca_y = r * sinf(angle) + t * 2.0f;
            cdr->pep.residues[i].ca_z = base_z + t * 1.5f;
            cdr->ca_x[i] = cdr->pep.residues[i].ca_x;
            cdr->ca_y[i] = cdr->pep.residues[i].ca_y;
            cdr->ca_z[i] = cdr->pep.residues[i].ca_z;
        }

        total += n;
    }

    /* Framework placeholders */
    out->framework.vh_cx = 0.0f;  out->framework.vh_cy = 10.0f;  out->framework.vh_cz = -10.0f;
    out->framework.vl_cx = 0.0f;  out->framework.vl_cy = -10.0f; out->framework.vl_cz = -10.0f;
    out->framework.elbow_angle = 160.0f;
    out->framework.qw = 1.0f;
    out->framework.mass = 25000.0f;

    return total;
}

/* -----------------------------------------------------------------------
 * nibble_antibody_build_paratope
 * ----------------------------------------------------------------------- */
NibbleGrid *nibble_antibody_build_paratope(
    AB_Antibody *ab,
    int          grid_dim,
    float        grid_res)
{
    if (!ab) return NULL;
    if (grid_dim <= 0) grid_dim = 80;
    if (grid_res <= 0) grid_res = 0.5f;

    NibbleGrid *pgrid = nibble_create_grid(grid_dim, grid_dim, grid_dim, grid_res);
    if (!pgrid) return NULL;

    /* Project each CDR loop into the paratope grid, weighted by CDR weight */
    for (int c = 0; c < AB_N_CDR_LOOPS; c++) {
        AB_CDRLoop *cdr = &ab->cdrs[c];
        if (cdr->n_residues == 0) continue;

        float weight = AB_CDR_WEIGHTS[c];
        float blur   = (c == AB_CDR_H3) ? 1.2f : 1.5f;  /* H3 finer detail */

        /* Project residues with CDR weight applied to channel values */
        float gx0 = pgrid->dim_x * grid_res * 0.5f;
        float gy0 = pgrid->dim_y * grid_res * 0.5f;
        float gz0 = pgrid->dim_z * grid_res * 0.5f;

        for (int i = 0; i < cdr->pep.n_residues; i++) {
            PEP_Residue *res = &cdr->pep.residues[i];
            /* Weighted channel values */
            float wd[N_CHANNELS];
            for (int ch = 0; ch < N_CHANNELS; ch++) {
                wd[ch] = res->d[ch] * weight;
            }
            nibble_project_atom(
                pgrid,
                res->ca_x - gx0 + pgrid->dim_x * grid_res * 0.5f,
                res->ca_y - gy0 + pgrid->dim_y * grid_res * 0.5f,
                res->ca_z - gz0 + pgrid->dim_z * grid_res * 0.5f,
                blur,
                wd
            );
        }
    }

    ab->paratope_grid = pgrid;
    ab->paratope_built = 1;
    return pgrid;
}

/* -----------------------------------------------------------------------
 * nibble_antibody_scan_epitope
 * ----------------------------------------------------------------------- */
int nibble_antibody_scan_epitope(
    const NibbleGrid *antigen_grid,
    const NibbleGrid *paratope_grid,
    AB_EpitopePatch  *patches,
    float             threshold)
{
    if (!antigen_grid || !paratope_grid || !patches) return 0;
    if (threshold <= 0) threshold = 0.1f;

    float res = antigen_grid->resolution;
    int dx = antigen_grid->dim_x;
    int dy = antigen_grid->dim_y;
    int dz = antigen_grid->dim_z;
    int nc = N_CHANNELS;

    int   n_patches    = 0;
    float best_scores[AB_MAX_EPITOPE_PATCHES];
    for (int i = 0; i < AB_MAX_EPITOPE_PATCHES; i++) best_scores[i] = 0.0f;

    /* Paratope grid half-size */
    int ph = paratope_grid->dim_x / 2;

    /* Slide paratope over antigen surface */
    for (int x = ph; x < dx - ph && n_patches < AB_MAX_EPITOPE_PATCHES; x++) {
        for (int y = ph; y < dy - ph; y++) {
            for (int z = ph; z < dz - ph; z++) {
                size_t ag_idx = (size_t)(x * dy * dz + y * dz + z) * nc;
                float steric = antigen_grid->data[ag_idx + CH_STERIC_DEMAND];
                /* Only score at surface voxels */
                if (steric < 0.3f || steric > 0.8f) continue;

                /* Compute Frobenius inner product over paratope window */
                double fp_sum = 0.0;
                int    fp_count = 0;
                int    n_hb = 0, n_hydro = 0, n_chrg = 0;

                int pdx = paratope_grid->dim_x;
                int pdy = paratope_grid->dim_y;
                int pdz = paratope_grid->dim_z;

                for (int pi = 0; pi < pdx; pi++) {
                    int ai = x - ph + pi;
                    if (ai < 0 || ai >= dx) continue;
                    for (int pj = 0; pj < pdy; pj++) {
                        int aj = y - ph + pj;
                        if (aj < 0 || aj >= dy) continue;
                        for (int pk = 0; pk < pdz; pk++) {
                            int ak = z - ph + pk;
                            if (ak < 0 || ak >= dz) continue;
                            size_t a_idx = (size_t)(ai*dy*dz + aj*dz + ak) * nc;
                            size_t p_idx = (size_t)(pi*pdy*pdz + pj*pdz + pk) * nc;
                            /* Channel inner product */
                            for (int c = 1; c < nc; c++) {
                                fp_sum += antigen_grid->data[a_idx + c] *
                                          paratope_grid->data[p_idx + c];
                            }
                            /* Count interaction types */
                            if (antigen_grid->data[a_idx + CH_HBA_DEMAND] > 0.3f ||
                                antigen_grid->data[a_idx + CH_HBD_DEMAND] > 0.3f)
                                n_hb++;
                            if (antigen_grid->data[a_idx + CH_LIPO_DEMAND] > 0.5f)
                                n_hydro++;
                            if (fabsf(antigen_grid->data[a_idx + CH_ELEC_DEMAND]) > 0.3f)
                                n_chrg++;
                            fp_count++;
                        }
                    }
                }

                if (fp_count == 0) continue;
                float score = (float)(fp_sum / fp_count);
                if (score < threshold) continue;

                /* Record patch */
                if (n_patches < AB_MAX_EPITOPE_PATCHES) {
                    AB_EpitopePatch *p = &patches[n_patches];
                    p->cx = x * res;
                    p->cy = y * res;
                    p->cz = z * res;
                    p->complementarity = score;
                    p->area = (float)fp_count * res * res;
                    p->buried_area = p->area * 0.7f;
                    p->n_hbond_sites = n_hb;
                    p->n_hydrophobic = n_hydro;
                    p->n_charged     = n_chrg;

                    /* Surface normal: gradient of steric field */
                    float nx, ny, nz;
                    nibble_gradient(antigen_grid,
                                    (float)x, (float)y, (float)z,
                                    &nx, &ny, &nz);
                    float nnorm = sqrtf(nx*nx + ny*ny + nz*nz);
                    if (nnorm > 1e-6f) {
                        p->normal_x = nx / nnorm;
                        p->normal_y = ny / nnorm;
                        p->normal_z = nz / nnorm;
                    }

                    p->is_conformational = (n_hb + n_hydro > 3) ? 1 : 0;
                    p->is_linear_epitope = (n_hb > n_hydro) ? 1 : 0;
                    p->sc_score = _clampf(score / 2.0f, 0.0f, 1.0f);
                    best_scores[n_patches] = score;
                    n_patches++;
                }
            }
        }
    }

    /* Sort patches by complementarity descending (insertion sort) */
    for (int i = 1; i < n_patches; i++) {
        AB_EpitopePatch tmp = patches[i];
        float ts = best_scores[i];
        int j = i - 1;
        while (j >= 0 && best_scores[j] < ts) {
            patches[j+1] = patches[j];
            best_scores[j+1] = best_scores[j];
            j--;
        }
        patches[j+1] = tmp;
        best_scores[j+1] = ts;
    }

    return n_patches;
}

/* -----------------------------------------------------------------------
 * nibble_antibody_h3_trajectory
 * ----------------------------------------------------------------------- */
float nibble_antibody_h3_trajectory(
    const NibbleGrid      *antigen_grid,
    AB_Antibody           *ab,
    const AB_EpitopePatch *epitope,
    int                    n_steps,
    float                  kT,
    float                 *score_out)
{
    if (!antigen_grid || !ab) return 0.0f;
    if (n_steps <= 0) n_steps = 300;
    if (kT <= 0) kT = 0.593f;

    AB_CDRLoop *h3 = &ab->cdrs[AB_CDR_H3];
    if (h3->n_residues == 0) {
        if (score_out) *score_out = 0.0f;
        return 0.0f;
    }

    /* Add anchor constraints to H3 PEP_Peptide */
    /* Anchor N: first H3 residue constrained to framework anchor_n */
    /* Anchor C: last H3 residue constrained to framework anchor_c  */
    if (h3->pep.n_constraints == 0 && h3->n_residues >= 2) {
        /* Create virtual anchor constraint via cyclic type (N-C spring) */
        nibble_peptide_add_constraint(
            &h3->pep, PEP_CONSTRAINT_CYCLIC, 0, h3->pep.n_residues - 1, 20.0f
        );
    }

    /* Place H3 at epitope centre */
    if (epitope) {
        float entry_x = epitope->cx;
        float entry_y = epitope->cy;
        float entry_z = epitope->cz;
        float rise = 1.5f;  /* short coil rise */
        for (int i = 0; i < h3->pep.n_residues; i++) {
            h3->pep.residues[i].ca_x = entry_x + epitope->normal_x * i * rise;
            h3->pep.residues[i].ca_y = entry_y + epitope->normal_y * i * rise;
            h3->pep.residues[i].ca_z = entry_z + epitope->normal_z * i * rise;
        }
    }

    /* Run peptide Langevin trajectory for H3 */
    PEP_DockingResult h3_result;
    float h3_score = nibble_peptide_trajectory(
        antigen_grid, &h3->pep, NULL,
        n_steps, 0.002f, 0.5f, kT, &h3_result
    );

    /* Update stored Cα positions */
    for (int i = 0; i < h3->n_residues && i < h3->pep.n_residues; i++) {
        h3->ca_x[i] = h3->pep.residues[i].ca_x;
        h3->ca_y[i] = h3->pep.residues[i].ca_y;
        h3->ca_z[i] = h3->pep.residues[i].ca_z;
    }

    if (score_out) *score_out = h3_score;
    return h3_score;
}

/* -----------------------------------------------------------------------
 * nibble_antibody_score — full composite
 * ----------------------------------------------------------------------- */
float nibble_antibody_score(
    const NibbleGrid *antigen_grid,
    AB_Antibody      *ab,
    int               n_steps,
    float             delta_G,
    float             kT,
    AB_DockingResult *out)
{
    if (!antigen_grid || !ab || !out) return 0.0f;
    if (kT <= 0) kT = 0.593f;
    if (delta_G <= 0) delta_G = 2.0f;

    memset(out, 0, sizeof(AB_DockingResult));

    /* --- Build paratope if not already built --- */
    if (!ab->paratope_built || !ab->paratope_grid) {
        nibble_antibody_build_paratope(ab, 60, 0.5f);
    }
    if (!ab->paratope_grid) return 0.0f;

    /* --- Scan for epitope patches --- */
    AB_EpitopePatch patches[AB_MAX_EPITOPE_PATCHES];
    int n_patches = nibble_antibody_scan_epitope(
        antigen_grid, ab->paratope_grid, patches, 0.05f
    );
    out->n_epitope_patches = n_patches;

    if (n_patches == 0) {
        /* No complementary patch found */
        out->composite_score = 0.0f;
        out->kd_class = 2;
        return 0.0f;
    }

    out->best_epitope_idx           = 0;
    out->best_epitope_complementarity = patches[0].complementarity;

    /* --- Shape complementarity (Sc) using nibble_ppi_affinity --- */
    /* Create antigen surface grid for Sc computation */
    PPI_Interface iface;
    memset(&iface, 0, sizeof(iface));
    iface.centre_x = patches[0].cx;
    iface.centre_y = patches[0].cy;
    iface.centre_z = patches[0].cz;
    iface.extent_x = iface.extent_y = iface.extent_z = 15.0f;

    PPI_Score ppi_score;
    float sc_score = 0.0f;
    if (ab->paratope_grid) {
        nibble_compute_water_network((NibbleGrid*)antigen_grid, delta_G, kT);
        nibble_compute_water_network(ab->paratope_grid, delta_G, kT);
        float ppi_comp = nibble_ppi_affinity(
            antigen_grid, ab->paratope_grid,
            &iface, delta_G, kT, &ppi_score
        );
        sc_score = ppi_score.sc_score;
    }
    out->paratope_epitope_sc = sc_score;
    out->buried_surface_area = patches[0].area * 2.0f;  /* both sides */

    /* --- CDR-H3 flexible trajectory --- */
    float h3_score = 0.0f;
    nibble_antibody_h3_trajectory(
        antigen_grid, ab, &patches[0], n_steps, kT, &h3_score
    );
    out->h3_affinity = h3_score;
    out->cdr_scores[AB_CDR_H3] = h3_score;

    /* --- Canonical CDR scores (L1, L2, L3, H1, H2) --- */
    /* Project each CDR into a scratch grid and score against antigen */
    NibbleGrid *scratch = nibble_create_grid(
        antigen_grid->dim_x, antigen_grid->dim_y, antigen_grid->dim_z,
        antigen_grid->resolution
    );
    float canonical_sum = 0.0f;
    if (scratch) {
        for (int c = 0; c < AB_N_CDR_LOOPS - 1; c++) {  /* skip H3 */
            AB_CDRLoop *cdr = &ab->cdrs[c];
            if (cdr->n_residues == 0) continue;
            nibble_reset_steric(scratch);
            nibble_peptide_project(scratch, &cdr->pep, -1, 1.5f);
            float raw = nibble_compute_affinity_full(antigen_grid, scratch, delta_G);
            float sc  = _clampf(raw / 2000.0f, 0.0f, 1.0f);
            out->cdr_scores[c] = sc;
            canonical_sum += sc * AB_CDR_WEIGHTS[c];
        }
        /* Normalise by sum of non-H3 weights */
        float non_h3_weight = 0.0f;
        for (int c = 0; c < AB_N_CDR_LOOPS - 1; c++) non_h3_weight += AB_CDR_WEIGHTS[c];
        out->canonical_cdr_score = (non_h3_weight > 0)
            ? canonical_sum / non_h3_weight : 0.0f;
        nibble_free_grid(scratch);
    }

    /* --- VH/VL orientation penalty --- */
    /*
     * The VH-VL elbow angle should be in the range 140–180°.
     * Outside this range → steric clashes or domain misalignment.
     * We model this as a soft penalty outside [135, 185].
     */
    float ea = ab->framework.elbow_angle;
    float vhvl_penalty = 0.0f;
    if (ea < 135.0f) vhvl_penalty = (135.0f - ea) / 45.0f;
    if (ea > 185.0f) vhvl_penalty = (ea - 185.0f) / 45.0f;
    vhvl_penalty = _clampf(vhvl_penalty, 0.0f, 1.0f);
    out->vhvl_orientation_penalty = vhvl_penalty;

    /* --- Composite score --- */
    float norm_bsa = _clampf(out->buried_surface_area / 2000.0f, 0.0f, 1.0f);

    float composite =
        _WAB_SC   * sc_score                +
        _WAB_H3   * h3_score                +
        _WAB_CDR  * out->canonical_cdr_score +
        _WAB_BSA  * norm_bsa                 +
        _WAB_VHVL * vhvl_penalty;

    out->composite_score = _clampf(composite, 0.0f, 1.0f);

    /* Kd class */
    if (out->composite_score >= _KD_NM_THRESHOLD)      out->kd_class = 0;
    else if (out->composite_score >= _KD_UM_THRESHOLD) out->kd_class = 1;
    else                                                out->kd_class = 2;

    return out->composite_score;
}

/* -----------------------------------------------------------------------
 * nibble_antibody_free
 * ----------------------------------------------------------------------- */
void nibble_antibody_free(AB_Antibody *ab) {
    if (!ab) return;
    if (ab->paratope_grid) {
        nibble_free_grid(ab->paratope_grid);
        ab->paratope_grid = NULL;
        ab->paratope_built = 0;
    }
}

/* -----------------------------------------------------------------------
 * Utility printers
 * ----------------------------------------------------------------------- */
void nibble_antibody_print_result(const AB_DockingResult *r) {
    if (!r) return;
    static const char *kd_labels[] = {"nM-class", "µM-class", "weak/non-binder"};
    printf("[nibble_antibody] Docking Result\n");
    printf("  Composite score:          %.4f  [%s]\n",
           r->composite_score, kd_labels[r->kd_class]);
    printf("  Paratope-epitope Sc:      %.4f\n", r->paratope_epitope_sc);
    printf("  CDR-H3 affinity:          %.4f\n", r->h3_affinity);
    printf("  Canonical CDR score:      %.4f\n", r->canonical_cdr_score);
    printf("  Buried surface area:      %.1f Å²\n", r->buried_surface_area);
    printf("  VH/VL orientation penalty:%.4f\n", r->vhvl_orientation_penalty);
    printf("  Epitope patches found:    %d\n", r->n_epitope_patches);
    printf("  Best epitope score:       %.4f\n", r->best_epitope_complementarity);
    printf("  Per-CDR scores:\n");
    for (int c = 0; c < AB_N_CDR_LOOPS; c++) {
        printf("    %s: %.4f  (weight=%.2f)\n",
               AB_CDR_NAMES[c], r->cdr_scores[c], AB_CDR_WEIGHTS[c]);
    }
}

void nibble_antibody_print(const AB_Antibody *ab) {
    if (!ab) return;
    printf("[nibble_antibody] Antibody: %s\n", ab->name);
    printf("  VH/VL elbow angle: %.1f°\n", ab->framework.elbow_angle);
    printf("  Paratope built:    %s\n", ab->paratope_built ? "yes" : "no");
    printf("  CDR loops:\n");
    for (int c = 0; c < AB_N_CDR_LOOPS; c++) {
        const AB_CDRLoop *cdr = &ab->cdrs[c];
        if (cdr->n_residues == 0) {
            printf("    %s: (empty)\n", AB_CDR_NAMES[c]);
        } else {
            printf("    %s: %d residues  %s  %s\n",
                   AB_CDR_NAMES[c], cdr->n_residues, cdr->sequence,
                   cdr->is_h3 ? "[H3 flexible]" : "");
        }
    }
}
