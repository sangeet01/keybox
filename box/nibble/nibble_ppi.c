/*
 * nibble_ppi.c
 * Protein-Protein Interface (PPI) Extension for the Nibble Voxel Engine.
 *
 * Six-phase implementation:
 *   I.   nibble_detect_interface()       — inter-residue contact detection
 *   II.  nibble_load_ppi_surface()       — irregular surface patch loading
 *   III. nibble_sc_score()               — Lawrence-Coleman shape complementarity
 *   IV.  nibble_score_hotspots()         — alanine-scanning proxy hotspot detection
 *   V.   nibble_ppi_affinity()           — composite PPI binding score
 *   VI.  nibble_load_ppi_inhibitor_site()— pharmacophore extraction for inh. design
 *
 * All functions operate on existing NibbleGrid / NibbleMol types.
 * No modifications to nibble.c, nibble_tier3.c, nibble_tier4.c, nibble_gaps.c.
 *
 * Author: Khukuri / KeyBox project (sangeet01)
 * License: Apache 2.0 with Commons Clause
 */

#include "nibble_ppi.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Internal limits and weights
 * ----------------------------------------------------------------------- */

#define _PPI_MAX_RAW_ATOMS   50000
#define _PPI_HBOND_CUTOFF     3.5f   /* Å H-bond distance cutoff             */
#define _PPI_HYDRO_CUTOFF     4.5f   /* Å hydrophobic contact cutoff          */
#define _PPI_WATER_CUTOFF     3.5f   /* Å bridging water cutoff               */

/* Composite score weights (sum = 1.0) */
#define _W_AFFINITY           0.35f
#define _W_SC                 0.25f
#define _W_HOTSPOT            0.20f
#define _W_WATER              0.10f
#define _W_DESOLVATION       -0.10f  /* negative: penalty term */

/* Hotspot proxy weights */
#define _WH_STERIC            0.30f
#define _WH_ELEC              0.25f
#define _WH_HBOND             0.30f
#define _WH_LIPO              0.15f

/* Shape complementarity window radius */
#define _SC_WINDOW            7.5f

/* -----------------------------------------------------------------------
 * Internal atom record for PDB parsing
 * ----------------------------------------------------------------------- */
typedef struct {
    char  chain;
    int   resnum;
    char  resname[4];
    char  element;      /* first element char: C N O S P F H etc. */
    float x, y, z;
    float bfactor;
    int   is_hetatm;
    int   is_water;
} _PPI_Atom;

/* -----------------------------------------------------------------------
 * Internal: clamp float
 * ----------------------------------------------------------------------- */
static inline float _clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* -----------------------------------------------------------------------
 * Internal: squared distance between two 3D points
 * ----------------------------------------------------------------------- */
static inline float _dist2(float ax, float ay, float az,
                            float bx, float by, float bz) {
    float dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx*dx + dy*dy + dz*dz;
}

/* -----------------------------------------------------------------------
 * Internal: parse one PDB file into a flat atom array.
 * Returns number of atoms parsed, or -1 on error.
 * ----------------------------------------------------------------------- */
static int _parse_pdb(const char *path, _PPI_Atom *atoms, int max_atoms) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[nibble_ppi] Cannot open PDB: %s\n", path);
        return -1;
    }
    char line[256];
    int  n = 0;
    while (fgets(line, sizeof(line), fp) && n < max_atoms) {
        int is_atom   = (strncmp(line, "ATOM  ", 6) == 0);
        int is_hetatm = (strncmp(line, "HETATM", 6) == 0);
        if (!is_atom && !is_hetatm) continue;

        int len = (int)strlen(line);
        if (len < 54) continue;

        _PPI_Atom *a = &atoms[n];
        a->is_hetatm = is_hetatm;
        a->chain     = (len > 21) ? line[21] : ' ';
        a->bfactor   = 0.0f;

        /* Residue number */
        char tmp[6] = {0};
        memcpy(tmp, line + 22, 4);
        a->resnum = atoi(tmp);

        /* Residue name */
        memcpy(a->resname, line + 17, 3);
        a->resname[3] = '\0';

        /* Coordinates */
        char xb[9]={0}, yb[9]={0}, zb[9]={0};
        memcpy(xb, line+30, 8); memcpy(yb, line+38, 8); memcpy(zb, line+46, 8);
        a->x = (float)atof(xb);
        a->y = (float)atof(yb);
        a->z = (float)atof(zb);

        /* B-factor */
        if (len > 66) {
            char bb[7] = {0};
            memcpy(bb, line+60, 6);
            a->bfactor = (float)atof(bb);
        }

        /* Element (column 77-78, fallback to column 13) */
        char elem = ' ';
        if (len > 77 && line[76] != ' ') {
            elem = line[76];
        } else if (len > 13) {
            /* Skip leading spaces */
            char c = line[13];
            if (c == ' ' && len > 14) c = line[14];
            elem = c;
        }
        /* Uppercase */
        if (elem >= 'a' && elem <= 'z') elem -= 32;
        a->element = elem;

        /* Water flag */
        a->is_water = (strstr(a->resname, "HOH") != NULL ||
                       strstr(a->resname, "WAT") != NULL);

        n++;
    }
    fclose(fp);
    return n;
}

/* -----------------------------------------------------------------------
 * Internal: compute residue centroids from flat atom list
 * ----------------------------------------------------------------------- */
static int _build_residues(const _PPI_Atom *atoms, int n_atoms,
                            char filter_chain,
                            PPI_Residue *residues, int max_res) {
    int n_res = 0;

    for (int i = 0; i < n_atoms; i++) {
        const _PPI_Atom *a = &atoms[i];
        if (a->is_water || a->is_hetatm) continue;
        if (filter_chain != 0 && a->chain != filter_chain) continue;

        /* Find existing residue or create new */
        int found = -1;
        for (int r = 0; r < n_res; r++) {
            if (residues[r].resnum == a->resnum &&
                residues[r].chain  == a->chain) {
                found = r;
                break;
            }
        }
        if (found < 0) {
            if (n_res >= max_res) continue;
            found = n_res++;
            residues[found].chain    = a->chain;
            residues[found].resnum   = a->resnum;
            memcpy(residues[found].resname, a->resname, 4);
            residues[found].cx       = 0.0f;
            residues[found].cy       = 0.0f;
            residues[found].cz       = 0.0f;
            residues[found].n_atoms  = 0;
            residues[found].is_hotspot = 0;
            residues[found].hotspot_weight = 0.0f;
        }

        /* Running centroid sum */
        PPI_Residue *res = &residues[found];
        res->cx += a->x;
        res->cy += a->y;
        res->cz += a->z;
        res->n_atoms++;
    }

    /* Finalise centroids */
    for (int r = 0; r < n_res; r++) {
        if (residues[r].n_atoms > 0) {
            float inv = 1.0f / residues[r].n_atoms;
            residues[r].cx *= inv;
            residues[r].cy *= inv;
            residues[r].cz *= inv;
        }
    }
    return n_res;
}

/* -----------------------------------------------------------------------
 * Phase I: Interface detection
 * ----------------------------------------------------------------------- */
int nibble_detect_interface(
    const char *pdb_a,
    const char *pdb_b,
    char        chain_a,
    char        chain_b,
    float       cutoff,
    PPI_Interface *out)
{
    if (!pdb_a || !pdb_b || !out) return -1;
    if (cutoff <= 0.0f) cutoff = PPI_INTERFACE_CUTOFF;
    float cutoff2 = cutoff * cutoff;

    /* Allocate atom arrays on heap (large) */
    _PPI_Atom *atoms_a = (_PPI_Atom*)malloc(_PPI_MAX_RAW_ATOMS * sizeof(_PPI_Atom));
    _PPI_Atom *atoms_b = (_PPI_Atom*)malloc(_PPI_MAX_RAW_ATOMS * sizeof(_PPI_Atom));
    if (!atoms_a || !atoms_b) {
        free(atoms_a); free(atoms_b);
        return -1;
    }

    /* Parse both PDB files */
    int na = _parse_pdb(pdb_a, atoms_a, _PPI_MAX_RAW_ATOMS);
    int nb = _parse_pdb(pdb_b, atoms_b, _PPI_MAX_RAW_ATOMS);

    if (na <= 0 || nb <= 0) {
        free(atoms_a); free(atoms_b);
        fprintf(stderr, "[nibble_ppi] PDB parse failed: na=%d nb=%d\n", na, nb);
        return -1;
    }

    /* Build residue lists */
    memset(out, 0, sizeof(PPI_Interface));
    out->n_residues_a = _build_residues(atoms_a, na, chain_a,
                                         out->residues_a, PPI_MAX_RESIDUES);
    out->n_residues_b = _build_residues(atoms_b, nb, chain_b,
                                         out->residues_b, PPI_MAX_RESIDUES);

    /* Find interface: residues with at least one inter-protein atom contact */
    char *in_iface_a = (char*)calloc(out->n_residues_a, 1);
    char *in_iface_b = (char*)calloc(out->n_residues_b, 1);

    int n_hbonds = 0, n_hydro = 0, n_waters = 0;
    float cx = 0, cy = 0, cz = 0;
    int   n_iface_pairs = 0;

    /* O(na * nb) atom-atom scan — acceptable for PPI sizes (<50k atoms) */
    for (int i = 0; i < na; i++) {
        if (atoms_a[i].is_water || atoms_a[i].is_hetatm) continue;
        if (chain_a != 0 && atoms_a[i].chain != chain_a) continue;

        for (int j = 0; j < nb; j++) {
            if (atoms_b[j].is_water) continue;
            if (chain_b != 0 && atoms_b[j].chain != chain_b) continue;

            float d2 = _dist2(atoms_a[i].x, atoms_a[i].y, atoms_a[i].z,
                               atoms_b[j].x, atoms_b[j].y, atoms_b[j].z);
            if (d2 > cutoff2) continue;

            /* Mark residues as interface */
            for (int ra = 0; ra < out->n_residues_a; ra++) {
                if (out->residues_a[ra].resnum == atoms_a[i].resnum &&
                    out->residues_a[ra].chain  == atoms_a[i].chain) {
                    if (!in_iface_a[ra]) {
                        in_iface_a[ra] = 1;
                        cx += out->residues_a[ra].cx;
                        cy += out->residues_a[ra].cy;
                        cz += out->residues_a[ra].cz;
                        n_iface_pairs++;
                    }
                    break;
                }
            }
            for (int rb = 0; rb < out->n_residues_b; rb++) {
                if (out->residues_b[rb].resnum == atoms_b[j].resnum &&
                    out->residues_b[rb].chain  == atoms_b[j].chain) {
                    in_iface_b[rb] = 1;
                    break;
                }
            }

            /* Classify contact type */
            char ea = atoms_a[i].element;
            char eb = atoms_b[j].element;
            float d = sqrtf(d2);

            /* H-bond: N–O or O–N within 3.5Å */
            if (d <= _PPI_HBOND_CUTOFF) {
                if ((ea == 'N' && eb == 'O') || (ea == 'O' && eb == 'N') ||
                    (ea == 'O' && eb == 'O'))
                    n_hbonds++;
            }
            /* Hydrophobic: C–C within 4.5Å */
            if (d <= _PPI_HYDRO_CUTOFF && ea == 'C' && eb == 'C')
                n_hydro++;
        }
    }

    /* Count bridging waters */
    for (int i = 0; i < na; i++) {
        if (!atoms_a[i].is_water) continue;
        for (int j = 0; j < nb; j++) {
            if (!atoms_b[j].is_water) continue;
            float d2 = _dist2(atoms_a[i].x, atoms_a[i].y, atoms_a[i].z,
                               atoms_b[j].x, atoms_b[j].y, atoms_b[j].z);
            if (d2 <= _PPI_WATER_CUTOFF * _PPI_WATER_CUTOFF) {
                n_waters++;
                break;
            }
        }
    }

    out->n_hbonds           = n_hbonds;
    out->n_hydrophobic      = n_hydro;
    out->n_bridging_waters  = n_waters;

    /* Interface centre */
    if (n_iface_pairs > 0) {
        float inv = 1.0f / n_iface_pairs;
        out->centre_x = cx * inv;
        out->centre_y = cy * inv;
        out->centre_z = cz * inv;
    }

    /* Bounding box extents */
    float min_x = 1e9f, max_x = -1e9f;
    float min_y = 1e9f, max_y = -1e9f;
    float min_z = 1e9f, max_z = -1e9f;
    for (int ra = 0; ra < out->n_residues_a; ra++) {
        if (!in_iface_a[ra]) continue;
        float rx = out->residues_a[ra].cx;
        float ry = out->residues_a[ra].cy;
        float rz = out->residues_a[ra].cz;
        if (rx < min_x) min_x = rx; if (rx > max_x) max_x = rx;
        if (ry < min_y) min_y = ry; if (ry > max_y) max_y = ry;
        if (rz < min_z) min_z = rz; if (rz > max_z) max_z = rz;
    }
    out->extent_x = (max_x - min_x) * 0.5f + 5.0f;  /* +5Å padding */
    out->extent_y = (max_y - min_y) * 0.5f + 5.0f;
    out->extent_z = (max_z - min_z) * 0.5f + 5.0f;

    /* Buried surface area estimate: ~15 Å² per interface residue (empirical) */
    int n_a_iface = 0, n_b_iface = 0;
    for (int i = 0; i < out->n_residues_a; i++) if (in_iface_a[i]) n_a_iface++;
    for (int i = 0; i < out->n_residues_b; i++) if (in_iface_b[i]) n_b_iface++;
    out->buried_surface_area = (float)(n_a_iface + n_b_iface) * 15.0f;

    free(in_iface_a); free(in_iface_b);
    free(atoms_a); free(atoms_b);

    return n_iface_pairs;
}

/* -----------------------------------------------------------------------
 * Phase II: Surface loading
 * ----------------------------------------------------------------------- */
int nibble_load_ppi_surface(
    NibbleGrid          *grid,
    const char          *pdb_path,
    const PPI_Interface *iface,
    int                  use_side_a,
    float                blur_radius)
{
    if (!grid || !pdb_path || !iface) return -1;
    if (blur_radius <= 0.0f) blur_radius = 1.5f;

    /* Reset steric to 1.0 (void) */
    nibble_reset_steric(grid);

    /* Parse PDB */
    _PPI_Atom *atoms = (_PPI_Atom*)malloc(_PPI_MAX_RAW_ATOMS * sizeof(_PPI_Atom));
    if (!atoms) return -1;
    int n_atoms = _parse_pdb(pdb_path, atoms, _PPI_MAX_RAW_ATOMS);
    if (n_atoms <= 0) { free(atoms); return -1; }

    /* Origin of the grid = interface centre - extent */
    float sx = iface->centre_x - iface->extent_x;
    float sy = iface->centre_y - iface->extent_y;
    float sz = iface->centre_z - iface->extent_z;

    /* Surface range: atoms within extent + blur_radius of centre */
    float range2 = (iface->extent_x * iface->extent_x +
                    iface->extent_y * iface->extent_y +
                    iface->extent_z * iface->extent_z) * 1.5f; /* generous */

    /* Get interface residue list for this side */
    const PPI_Residue *res_list = use_side_a ? iface->residues_a : iface->residues_b;
    int n_res = use_side_a ? iface->n_residues_a : iface->n_residues_b;

    int n_projected = 0;

    for (int i = 0; i < n_atoms; i++) {
        const _PPI_Atom *a = &atoms[i];
        if (a->is_water) continue;

        /* Distance filter: within surface range of interface centre */
        float d2 = _dist2(a->x, a->y, a->z,
                           iface->centre_x, iface->centre_y, iface->centre_z);
        if (d2 > range2) continue;

        /* Adaptive B-factor blur */
        float atom_blur = blur_radius;
        if (a->bfactor > 0.0f) {
            float b_sigma = sqrtf(a->bfactor / 78.957f);
            atom_blur *= (1.0f + _clampf(b_sigma, 0.0f, 2.0f));
        }

        /* Find hotspot weight for this residue */
        float hotspot_w = 0.0f;
        for (int r = 0; r < n_res; r++) {
            if (res_list[r].resnum == a->resnum && res_list[r].chain == a->chain) {
                hotspot_w = res_list[r].hotspot_weight;
                break;
            }
        }

        /* Build demand channels */
        float d[N_CHANNELS];
        memset(d, 0, sizeof(d));
        d[CH_STERIC_DEMAND] = -1.0f;  /* protein atom = wall */

        /* Flexibility from residue type */
        d[CH_FLEXIBILITY] = 0.3f;
        if (strstr(a->resname, "GLY") || strstr(a->resname, "ALA"))
            d[CH_FLEXIBILITY] = 0.9f;
        else if (strstr(a->resname, "PRO"))
            d[CH_FLEXIBILITY] = 0.05f;
        else if (strstr(a->resname, "VAL") || strstr(a->resname, "LEU") ||
                 strstr(a->resname, "ILE"))
            d[CH_FLEXIBILITY] = 0.15f;

        /* Hotspot weight: stored in CH_PHOBIC_CORE channel
           (repurposed — interface between known channels) */
        d[CH_PHOBIC_CORE] = hotspot_w;

        /* Water records */
        if (a->is_hetatm && a->is_water) {
            d[CH_STERIC_DEMAND]  = 0.0f;  /* waters don't block */
            d[CH_HBA_DEMAND]     = 1.0f;
            d[CH_HBD_DEMAND]     = 1.0f;
            d[CH_WATER_CONSERVED]= (a->bfactor < 30.0f) ? 1.0f : 0.5f;
            nibble_project_atom(grid,
                a->x - sx, a->y - sy, a->z - sz,
                1.4f, d);
            n_projected++;
            continue;
        }

        /* Element-specific demands */
        switch (a->element) {
            case 'O':
                d[CH_HBA_DEMAND]  =  1.0f;
                d[CH_ELEC_DEMAND] = -0.6f;
                d[CH_SOLVENT_EXPO]=  0.4f;
                break;
            case 'N':
                d[CH_HBD_DEMAND]  =  1.0f;
                d[CH_ELEC_DEMAND] =  0.4f;
                break;
            case 'C':
                d[CH_LIPO_DEMAND] =  0.7f;
                /* CH_PHOBIC_CORE already set to hotspot_w above */
                break;
            case 'S':
                d[CH_HBA_DEMAND]   =  0.5f;
                d[CH_METAL_DEMAND] =  0.3f;
                d[CH_LIPO_DEMAND]  =  0.4f;
                break;
            case 'P':
                d[CH_ANION_DEMAND] =  0.8f;
                d[CH_HBA_DEMAND]   =  0.6f;
                break;
            default:
                break;
        }

        nibble_project_atom(grid,
            a->x - sx, a->y - sy, a->z - sz,
            atom_blur, d);
        n_projected++;
    }

    free(atoms);
    return n_projected;
}

/* -----------------------------------------------------------------------
 * Phase III: Shape complementarity (Lawrence-Coleman Sc)
 *
 * For each surface voxel in grid_a, estimate the outward surface normal
 * via the gradient of the steric field. Find the nearest complementary
 * voxel in grid_b. Dot the normals. Accumulate.
 *
 * Sc = (1/N) * sum_i [ n_a(i) · (-n_b(j_nearest)) ]
 * where n is the unit outward normal at each surface voxel.
 * ----------------------------------------------------------------------- */
float nibble_sc_score(
    const NibbleGrid *grid_a,
    const NibbleGrid *grid_b)
{
    if (!grid_a || !grid_b) return 0.0f;

    float res = grid_a->resolution;
    int dx = grid_a->dim_x, dy = grid_a->dim_y, dz = grid_a->dim_z;
    int nc = N_CHANNELS;
    float window2 = _SC_WINDOW * _SC_WINDOW;

    double sc_sum   = 0.0;
    int    sc_count = 0;

    /* Iterate over surface voxels of grid_a:
       surface = steric demand transitioning from wall (< 0.1) to void (> 0.5) */
    for (int x = 1; x < dx-1; x++) {
        for (int y = 1; y < dy-1; y++) {
            for (int z = 1; z < dz-1; z++) {
                size_t idx = (size_t)(x * dy * dz + y * dz + z) * nc;
                float ps = grid_a->data[idx + CH_STERIC_DEMAND];

                /* Surface voxel: void but adjacent to wall */
                if (ps < 0.3f || ps > 0.8f) continue;

                /* Estimate outward normal via finite difference of steric field */
                float gx = (grid_a->data[((size_t)((x+1)*dy*dz + y*dz + z)) * nc + CH_STERIC_DEMAND] -
                             grid_a->data[((size_t)((x-1)*dy*dz + y*dz + z)) * nc + CH_STERIC_DEMAND])
                             / (2.0f * res);
                float gy = (grid_a->data[((size_t)(x*dy*dz + (y+1)*dz + z)) * nc + CH_STERIC_DEMAND] -
                             grid_a->data[((size_t)(x*dy*dz + (y-1)*dz + z)) * nc + CH_STERIC_DEMAND])
                             / (2.0f * res);
                float gz = (grid_a->data[((size_t)(x*dy*dz + y*dz + (z+1))) * nc + CH_STERIC_DEMAND] -
                             grid_a->data[((size_t)(x*dy*dz + y*dz + (z-1))) * nc + CH_STERIC_DEMAND])
                             / (2.0f * res);

                float gnorm = sqrtf(gx*gx + gy*gy + gz*gz);
                if (gnorm < 1e-6f) continue;
                float inv_gnorm = 1.0f / gnorm;
                gx *= inv_gnorm;
                gy *= inv_gnorm;
                gz *= inv_gnorm;

                /* Find best-matching surface voxel in grid_b within window */
                float ax = x * res;
                float ay = y * res;
                float az = z * res;

                float best_dot = -2.0f;
                int   window_r = (int)(_SC_WINDOW / res) + 1;
                int   bx0 = x - window_r; if (bx0 < 1) bx0 = 1;
                int   bx1 = x + window_r; if (bx1 > dx-2) bx1 = dx-2;
                int   by0 = y - window_r; if (by0 < 1) by0 = 1;
                int   by1 = y + window_r; if (by1 > dy-2) by1 = dy-2;
                int   bz0 = z - window_r; if (bz0 < 1) bz0 = 1;
                int   bz1 = z + window_r; if (bz1 > dz-2) bz1 = dz-2;

                for (int bx = bx0; bx <= bx1; bx++) {
                    float vx = bx * res;
                    float ddx = (ax - vx) * (ax - vx);
                    if (ddx > window2) continue;
                    for (int by = by0; by <= by1; by++) {
                        float vy = by * res;
                        float ddy = ddx + (ay - vy) * (ay - vy);
                        if (ddy > window2) continue;
                        for (int bz = bz0; bz <= bz1; bz++) {
                            float vz = bz * res;
                            float d2 = ddy + (az - vz) * (az - vz);
                            if (d2 > window2) continue;

                            size_t bidx = (size_t)(bx * dy * dz + by * dz + bz) * nc;
                            float ps_b = grid_b->data[bidx + CH_STERIC_DEMAND];
                            if (ps_b < 0.3f || ps_b > 0.8f) continue;

                            /* Surface normal for grid_b */
                            float hx = (grid_b->data[((size_t)((bx+1)*dy*dz + by*dz + bz)) * nc + CH_STERIC_DEMAND] -
                                        grid_b->data[((size_t)((bx-1)*dy*dz + by*dz + bz)) * nc + CH_STERIC_DEMAND])
                                        / (2.0f * res);
                            float hy = (grid_b->data[((size_t)(bx*dy*dz + (by+1)*dz + bz)) * nc + CH_STERIC_DEMAND] -
                                        grid_b->data[((size_t)(bx*dy*dz + (by-1)*dz + bz)) * nc + CH_STERIC_DEMAND])
                                        / (2.0f * res);
                            float hz = (grid_b->data[((size_t)(bx*dy*dz + by*dz + (bz+1))) * nc + CH_STERIC_DEMAND] -
                                        grid_b->data[((size_t)(bx*dy*dz + by*dz + (bz-1))) * nc + CH_STERIC_DEMAND])
                                        / (2.0f * res);

                            float hnorm = sqrtf(hx*hx + hy*hy + hz*hz);
                            if (hnorm < 1e-6f) continue;
                            float inv_hnorm = 1.0f / hnorm;
                            hx *= inv_hnorm;
                            hy *= inv_hnorm;
                            hz *= inv_hnorm;

                            /* Sc: surfaces should face each other → dot of n_a and -n_b */
                            float dot = gx*(-hx) + gy*(-hy) + gz*(-hz);
                            if (dot > best_dot) best_dot = dot;
                        }
                    }
                }

                if (best_dot > -2.0f) {
                    sc_sum   += best_dot;
                    sc_count++;
                }
            }
        }
    }

    if (sc_count == 0) return 0.0f;
    /* Normalise to [0, 1]: raw dot product in [-1, 1], shift and scale */
    float sc_raw = (float)(sc_sum / sc_count);
    return _clampf((sc_raw + 1.0f) * 0.5f, 0.0f, 1.0f);
}

/* -----------------------------------------------------------------------
 * Phase IV: Hotspot scoring
 * ----------------------------------------------------------------------- */
int nibble_score_hotspots(
    const NibbleGrid *grid_a,
    const NibbleGrid *grid_b,
    PPI_Interface    *iface)
{
    if (!grid_a || !grid_b || !iface) return 0;

    float res = grid_a->resolution;
    int nc = N_CHANNELS;
    int n_hotspots = 0;

    /* For each residue on protein A, sample field values near its centroid */
    float sx = iface->centre_x - iface->extent_x;
    float sy = iface->centre_y - iface->extent_y;
    float sz = iface->centre_z - iface->extent_z;

    for (int r = 0; r < iface->n_residues_a; r++) {
        PPI_Residue *res_r = &iface->residues_a[r];

        /* Grid coordinates of residue centroid */
        float lx = res_r->cx - sx;
        float ly = res_r->cy - sy;
        float lz = res_r->cz - sz;

        int ix = (int)(lx / res);
        int iy = (int)(ly / res);
        int iz = (int)(lz / res);

        /* Sample grid_a at this residue's position */
        if (ix < 0 || iy < 0 || iz < 0 ||
            ix >= grid_a->dim_x || iy >= grid_a->dim_y || iz >= grid_a->dim_z)
            continue;

        size_t idx_a = (size_t)(ix * grid_a->dim_y * grid_a->dim_z +
                                 iy * grid_a->dim_z + iz) * nc;

        float steric_a = fabsf(grid_a->data[idx_a + CH_STERIC_DEMAND]);
        float elec_a   = fabsf(grid_a->data[idx_a + CH_ELEC_DEMAND]);
        float hba_a    = grid_a->data[idx_a + CH_HBA_DEMAND];
        float hbd_a    = grid_a->data[idx_a + CH_HBD_DEMAND];
        float lipo_a   = grid_a->data[idx_a + CH_LIPO_DEMAND];

        /* Sample grid_b at the same position (complementary demand) */
        float steric_b = 0.0f, elec_b = 0.0f, hba_b = 0.0f, hbd_b = 0.0f;
        if (ix < grid_b->dim_x && iy < grid_b->dim_y && iz < grid_b->dim_z) {
            size_t idx_b = (size_t)(ix * grid_b->dim_y * grid_b->dim_z +
                                     iy * grid_b->dim_z + iz) * nc;
            steric_b = fabsf(grid_b->data[idx_b + CH_STERIC_DEMAND]);
            elec_b   = fabsf(grid_b->data[idx_b + CH_ELEC_DEMAND]);
            hba_b    = grid_b->data[idx_b + CH_HBA_DEMAND];
            hbd_b    = grid_b->data[idx_b + CH_HBD_DEMAND];
        }

        /* Hotspot proxy: pairwise complementarity at this residue */
        float burial   = steric_a * steric_b;
        float elec_comp= elec_a   * elec_b;
        float hb_comp  = (hba_a * hbd_b + hbd_a * hba_b);
        float lipo_comp= lipo_a;

        float hw = (_WH_STERIC * burial  +
                    _WH_ELEC   * elec_comp +
                    _WH_HBOND  * hb_comp   +
                    _WH_LIPO   * lipo_comp);

        hw = _clampf(hw, 0.0f, 1.0f);
        res_r->hotspot_weight = hw;
        res_r->is_hotspot     = (hw >= PPI_HOTSPOT_ENERGY / 10.0f) ? 1 : 0;
        if (res_r->is_hotspot) n_hotspots++;
    }

    /* Repeat for protein B residues */
    for (int r = 0; r < iface->n_residues_b; r++) {
        PPI_Residue *res_r = &iface->residues_b[r];

        float lx = res_r->cx - sx;
        float ly = res_r->cy - sy;
        float lz = res_r->cz - sz;

        int ix = (int)(lx / res);
        int iy = (int)(ly / res);
        int iz = (int)(lz / res);

        if (ix < 0 || iy < 0 || iz < 0 ||
            ix >= grid_b->dim_x || iy >= grid_b->dim_y || iz >= grid_b->dim_z)
            continue;

        size_t idx_b = (size_t)(ix * grid_b->dim_y * grid_b->dim_z +
                                 iy * grid_b->dim_z + iz) * nc;

        float steric_b = fabsf(grid_b->data[idx_b + CH_STERIC_DEMAND]);
        float elec_b   = fabsf(grid_b->data[idx_b + CH_ELEC_DEMAND]);
        float hba_b    = grid_b->data[idx_b + CH_HBA_DEMAND];
        float hbd_b    = grid_b->data[idx_b + CH_HBD_DEMAND];
        float lipo_b   = grid_b->data[idx_b + CH_LIPO_DEMAND];

        float steric_a = 0.0f, elec_a = 0.0f, hba_a = 0.0f, hbd_a = 0.0f;
        if (ix < grid_a->dim_x && iy < grid_a->dim_y && iz < grid_a->dim_z) {
            size_t idx_a = (size_t)(ix * grid_a->dim_y * grid_a->dim_z +
                                     iy * grid_a->dim_z + iz) * nc;
            steric_a = fabsf(grid_a->data[idx_a + CH_STERIC_DEMAND]);
            elec_a   = fabsf(grid_a->data[idx_a + CH_ELEC_DEMAND]);
            hba_a    = grid_a->data[idx_a + CH_HBA_DEMAND];
            hbd_a    = grid_a->data[idx_a + CH_HBD_DEMAND];
        }

        float burial    = steric_b * steric_a;
        float elec_comp = elec_b   * elec_a;
        float hb_comp   = (hba_b * hbd_a + hbd_b * hba_a);
        float lipo_comp = lipo_b;

        float hw = (_WH_STERIC * burial    +
                    _WH_ELEC   * elec_comp +
                    _WH_HBOND  * hb_comp   +
                    _WH_LIPO   * lipo_comp);

        hw = _clampf(hw, 0.0f, 1.0f);
        res_r->hotspot_weight = hw;
        res_r->is_hotspot     = (hw >= PPI_HOTSPOT_ENERGY / 10.0f) ? 1 : 0;
        if (res_r->is_hotspot) n_hotspots++;
    }

    /* Update interface mean hotspot weight */
    float total_hw = 0.0f;
    int   total_res = 0;
    for (int r = 0; r < iface->n_residues_a; r++) {
        total_hw += iface->residues_a[r].hotspot_weight;
        total_res++;
    }
    for (int r = 0; r < iface->n_residues_b; r++) {
        total_hw += iface->residues_b[r].hotspot_weight;
        total_res++;
    }
    iface->mean_hotspot_weight = (total_res > 0)
        ? total_hw / total_res : 0.0f;

    return n_hotspots;
}

/* -----------------------------------------------------------------------
 * Phase V: Full PPI affinity score
 * ----------------------------------------------------------------------- */
float nibble_ppi_affinity(
    const NibbleGrid *grid_a,
    const NibbleGrid *grid_b,
    PPI_Interface    *iface,
    float             delta_G_bulk,
    float             kT,
    PPI_Score        *out)
{
    if (!grid_a || !grid_b || !iface || !out) return 0.0f;
    if (delta_G_bulk <= 0.0f) delta_G_bulk = 2.0f;
    if (kT <= 0.0f) kT = 0.593f;

    memset(out, 0, sizeof(PPI_Score));

    /* Raw affinity: use full scoring with water network */
    /* First compute water network on both grids */
    nibble_compute_water_network((NibbleGrid*)grid_a, delta_G_bulk, kT);
    nibble_compute_water_network((NibbleGrid*)grid_b, delta_G_bulk, kT);

    float raw_aff = nibble_compute_affinity_full(grid_a, grid_b, delta_G_bulk);
    out->raw_affinity = raw_aff;

    /* Shape complementarity */
    float sc = nibble_sc_score(grid_a, grid_b);
    iface->sc_score = sc;
    out->sc_score   = sc;

    /* Hotspot score */
    int n_hot = nibble_score_hotspots(grid_a, grid_b, iface);
    out->n_hotspots   = n_hot;
    out->hotspot_score = iface->mean_hotspot_weight;

    /* Water bonus: bridging waters × H-bond energy proxy */
    float water_bonus = (float)iface->n_bridging_waters * 0.5f * kT;
    out->water_bonus  = water_bonus;

    /* Desolvation penalty: estimated from buried surface area */
    /* ~0.025 kcal/mol per Å² (standard ASA transfer energy) */
    float desolv = iface->buried_surface_area * 0.025f;
    out->desolvation_penalty = desolv;

    /* Buried area estimates (split evenly — approximation) */
    out->buried_area_a = iface->buried_surface_area * 0.5f;
    out->buried_area_b = iface->buried_surface_area * 0.5f;

    /* Normalise components for composite (avoid dominance by raw_aff scale) */
    /* raw_aff can be very large — scale by 1/1000 for mixing */
    float norm_aff  = _clampf(raw_aff / 1000.0f, 0.0f, 1.0f);
    float norm_hot  = _clampf(out->hotspot_score, 0.0f, 1.0f);
    float norm_wat  = _clampf(water_bonus / 5.0f, 0.0f, 1.0f);
    float norm_des  = _clampf(desolv / 100.0f, 0.0f, 1.0f);

    float composite = (_W_AFFINITY    * norm_aff  +
                       _W_SC          * sc        +
                       _W_HOTSPOT     * norm_hot  +
                       _W_WATER       * norm_wat  +
                       _W_DESOLVATION * norm_des);   /* _W_DESOLVATION is negative */

    out->composite_score = _clampf(composite, 0.0f, 1.0f);

    return out->composite_score;
}

/* -----------------------------------------------------------------------
 * Phase VI: PPI inhibitor site extraction
 * ----------------------------------------------------------------------- */
int nibble_load_ppi_inhibitor_site(
    const NibbleGrid    *grid_a,
    const NibbleGrid    *grid_b,
    const PPI_Interface *iface,
    NibbleGrid          *inhibitor_grid,
    float                min_hotspot_weight)
{
    if (!grid_a || !grid_b || !iface || !inhibitor_grid) return 0;
    if (min_hotspot_weight <= 0.0f) min_hotspot_weight = 0.3f;

    /* Reset inhibitor grid */
    nibble_reset_steric(inhibitor_grid);

    int nc = N_CHANNELS;
    size_t n_vox = (size_t)grid_a->dim_x * grid_a->dim_y * grid_a->dim_z;
    int n_included = 0;

    /*
     * Algorithm:
     *   For each voxel in grid_a:
     *     If it is a surface voxel (transitional steric)
     *     AND the corresponding voxel in grid_b has protein presence
     *     (meaning protein B occupies that space)
     *     AND the hotspot weight at that voxel > threshold:
     *
     *   Then this voxel is part of the pharmacophore:
     *     - Invert the protein B field: what protein B presents becomes
     *       what the inhibitor must present (demand inversion)
     *     - Copy that demand into inhibitor_grid
     *
     * Result: a pocket grid that encodes what a small molecule must look
     * like to sit in the space where protein B binds protein A.
     * Compatible with nibble_trajectory() for Langevin docking.
     */

    /* Find hotspot-weighted voxels using CH_PHOBIC_CORE as hotspot map */
    for (size_t i = 0; i < n_vox; i++) {
        const float *pa = &grid_a->data[i * nc];
        const float *pb = &grid_b->data[i * nc];
        float       *pi = &inhibitor_grid->data[i * nc];

        float ps_a = pa[CH_STERIC_DEMAND];   /* protein A steric */
        float ps_b = pb[CH_STERIC_DEMAND];   /* protein B steric */
        float hw_a = pa[CH_PHOBIC_CORE];     /* hotspot weight stored here */

        /* Pharmacophore voxel:
           - grid_a surface (transitional: 0.1 < steric < 0.7)
           - grid_b has protein presence (ps_b < 0.3 → protein B is here)
           - hotspot weight above threshold */
        if (ps_a < 0.1f || ps_a > 0.7f) continue;
        if (ps_b > 0.3f) continue;           /* protein B not present */
        if (hw_a < min_hotspot_weight) continue;

        /*
         * Demand inversion:
         * What protein B presents is what the inhibitor must present.
         * Protein B H-bond donor → inhibitor must have H-bond acceptor.
         * Protein B negative charge → inhibitor must be positive.
         * Protein B hydrophobic → inhibitor must be hydrophobic.
         *
         * The inhibitor_grid is set up as a demand grid just like
         * a protein pocket — the drug side will fill it.
         */

        /* Steric: create a void where protein B was */
        pi[CH_STERIC_DEMAND] = 1.0f;   /* void — drug can enter here */

        /* Electrostatic: complement protein B's charge */
        pi[CH_ELEC_DEMAND] = -pb[CH_ELEC_DEMAND];

        /* H-bond: swap donor/acceptor */
        pi[CH_HBA_DEMAND]  = pb[CH_HBD_DEMAND];  /* drug needs HBA where B had HBD */
        pi[CH_HBD_DEMAND]  = pb[CH_HBA_DEMAND];  /* drug needs HBD where B had HBA */

        /* Hydrophobic: same preference */
        pi[CH_LIPO_DEMAND] = pb[CH_LIPO_DEMAND];

        /* Aromatic: same */
        pi[CH_AROM_DEMAND] = pb[CH_AROM_DEMAND];

        /* Metal: same */
        pi[CH_METAL_DEMAND] = pb[CH_METAL_DEMAND];

        /* Hotspot weight as phobic core signal */
        pi[CH_PHOBIC_CORE] = hw_a;

        /* Flexibility: inherit from protein A surface */
        pi[CH_FLEXIBILITY] = pa[CH_FLEXIBILITY];

        /* Fukui reactivity: inherit from protein B (covalent opportunity) */
        pi[CH_NUCLEOPHILICITY]  = pb[CH_NUCLEOPHILICITY];
        pi[CH_ELECTROPHILICITY] = pb[CH_ELECTROPHILICITY];

        /* Water: inherit conserved water positions */
        pi[CH_WATER_CONSERVED] = pb[CH_WATER_CONSERVED];

        n_included++;
    }

    return n_included;
}

/* -----------------------------------------------------------------------
 * Utility: print PPI_Score
 * ----------------------------------------------------------------------- */
void nibble_ppi_print_score(const PPI_Score *s) {
    if (!s) return;
    printf("[nibble_ppi] PPI Score Summary\n");
    printf("  Composite score:      %.4f\n",  s->composite_score);
    printf("  Raw affinity:         %.2f\n",  s->raw_affinity);
    printf("  Shape complementarity:%.4f\n",  s->sc_score);
    printf("  Hotspot score:        %.4f  (%d hotspots)\n",
           s->hotspot_score, s->n_hotspots);
    printf("  Water bonus:          %.4f\n",  s->water_bonus);
    printf("  Desolvation penalty:  %.4f\n",  s->desolvation_penalty);
    printf("  Buried area A/B:      %.1f / %.1f Å²\n",
           s->buried_area_a, s->buried_area_b);
}

/* -----------------------------------------------------------------------
 * Utility: print PPI_Interface
 * ----------------------------------------------------------------------- */
void nibble_ppi_print_interface(const PPI_Interface *iface) {
    if (!iface) return;
    printf("[nibble_ppi] PPI Interface Summary\n");
    printf("  Centre:         (%.2f, %.2f, %.2f)\n",
           iface->centre_x, iface->centre_y, iface->centre_z);
    printf("  Extents:        (%.2f, %.2f, %.2f)\n",
           iface->extent_x, iface->extent_y, iface->extent_z);
    printf("  Residues A/B:   %d / %d\n",
           iface->n_residues_a, iface->n_residues_b);
    printf("  H-bonds:        %d\n",  iface->n_hbonds);
    printf("  Hydrophobic:    %d\n",  iface->n_hydrophobic);
    printf("  Bridging waters:%d\n",  iface->n_bridging_waters);
    printf("  Buried area:    %.1f Å²\n", iface->buried_surface_area);
    printf("  Sc score:       %.4f\n", iface->sc_score);
    printf("  Mean hotspot w: %.4f\n", iface->mean_hotspot_weight);
}
