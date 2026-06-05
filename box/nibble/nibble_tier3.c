/*
 * nibble_tier3.c - Langevin Trajectory Engine (Tier 3)
 * Implements gradient computation and Langevin dynamics on the Nibble demand
 * field.
 */
#include "nibble.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Simple xorshift RNG for thermal noise */
static float rng_float(unsigned int *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return ((float)(*state & 0x7FFFFFFF)) / (float)0x7FFFFFFF;
}

/* Box-Muller: normally distributed noise */
static float rng_normal(unsigned int *state) {
  float u1 = rng_float(state) + 1e-8f;
  float u2 = rng_float(state);
  return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530f * u2);
}

static float score_at(const NibbleGrid *pocket, float x, float y, float z) {
  float inv_res = 1.0f / pocket->resolution;
  int ix = (int)(x * inv_res);
  int iy = (int)(y * inv_res);
  int iz = (int)(z * inv_res);
  float radius = 1.5f;
  int r_vox = (int)(radius * inv_res) + 1;
  float r2inv = 1.0f / (2.0f * radius * radius);
  float lim_sq = 4.0f * radius * radius;

  int x0 = ix - r_vox;
  if (x0 < 0)
    x0 = 0;
  int x1 = ix + r_vox;
  if (x1 >= pocket->dim_x)
    x1 = pocket->dim_x - 1;
  int y0 = iy - r_vox;
  if (y0 < 0)
    y0 = 0;
  int y1 = iy + r_vox;
  if (y1 >= pocket->dim_y)
    y1 = pocket->dim_y - 1;
  int z0 = iz - r_vox;
  if (z0 < 0)
    z0 = 0;
  int z1 = iz + r_vox;
  if (z1 >= pocket->dim_z)
    z1 = pocket->dim_z - 1;

  float score = 0.0f;
  int dy = pocket->dim_y;
  int dz = pocket->dim_z;
  int nc = N_CHANNELS;

  for (int nx = x0; nx <= x1; ++nx) {
    float vx = nx * pocket->resolution;
    float dxs = (x - vx) * (x - vx);
    if (dxs > lim_sq)
      continue;
    for (int ny = y0; ny <= y1; ++ny) {
      float vy = ny * pocket->resolution;
      float dxys = dxs + (y - vy) * (y - vy);
      if (dxys > lim_sq)
        continue;
      for (int nz = z0; nz <= z1; ++nz) {
        float vz = nz * pocket->resolution;
        float d2 = dxys + (z - vz) * (z - vz);
        if (d2 >= lim_sq)
          continue;

        float ds = expf(-d2 * r2inv);
        size_t idx = (size_t)(nx * dy * dz + ny * dz + nz) * nc;
        float ps = pocket->data[idx + CH_STERIC_DEMAND];

        if (ds > 0.1f && ps < 0.05f) {
          score -= 500.0f * ds;
        } else {
          score += ps * ds;
        }
      }
    }
  }
  return score;
}

/* Gradient of affinity field via finite difference */
void nibble_gradient(const NibbleGrid *pocket, float x, float y, float z,
                     float *gx, float *gy, float *gz) {
  float delta = pocket->resolution;
  *gx =
      (score_at(pocket, x + delta, y, z) - score_at(pocket, x - delta, y, z)) /
      (2.0f * delta);
  *gy =
      (score_at(pocket, x, y + delta, z) - score_at(pocket, x, y - delta, z)) /
      (2.0f * delta);
  *gz =
      (score_at(pocket, x, y, z + delta) - score_at(pocket, x, y, z - delta)) /
      (2.0f * delta);
}

/* -------------------------------------------------------------------
 * Quaternion utilities for rotational dynamics
 * ------------------------------------------------------------------- */

/* Normalize a quaternion to unit length */
void nibble_quat_normalize(float *qw, float *qx, float *qy, float *qz) {
  float norm = sqrtf(*qw * *qw + *qx * *qx + *qy * *qy + *qz * *qz);
  if (norm < 1e-9f) norm = 1.0f;
  float inv_norm = 1.0f / norm;
  *qw *= inv_norm;
  *qx *= inv_norm;
  *qy *= inv_norm;
  *qz *= inv_norm;
}

/* Rotate a 3D point using a quaternion (q is unit quaternion) */
void nibble_quat_rotate_point(float x, float y, float z,
                               float qw, float qx, float qy, float qz,
                               float *rx, float *ry, float *rz) {
  /* p' = q * p * q^-1, where q^-1 = conjugate for unit quaternion */
  float qw_conj = qw;
  float qx_conj = -qx;
  float qy_conj = -qy;
  float qz_conj = -qz;

  /* q * p */
  float p1w = -qx * x - qy * y - qz * z;
  float p1x = qw * x + qy * z - qz * y;
  float p1y = qw * y + qz * x - qx * z;
  float p1z = qw * z + qx * y - qy * x;

  /* (q*p) * q^-1 */
  *rx = p1x * qw_conj + p1w * qx_conj + p1y * qz_conj - p1z * qy_conj;
  *ry = p1y * qw_conj + p1w * qy_conj + p1z * qx_conj - p1x * qz_conj;
  *rz = p1z * qw_conj + p1w * qz_conj + p1x * qy_conj - p1y * qx_conj;
}

/* Integrate quaternion using angular velocity (Euler integration) */
void nibble_quat_integrate(float *qw, float *qx, float *qy, float *qz,
                            float wx, float wy, float wz, float dt) {
  /* dq/dt = 0.5 * Omega(w) * q */
  float dqw = 0.5f * (-wx * *qx - wy * *qy - wz * *qz);
  float dqx = 0.5f * (wx * *qw + wy * *qz - wz * *qy);
  float dqy = 0.5f * (wy * *qw + wz * *qx - wx * *qz);
  float dqz = 0.5f * (wz * *qw + wx * *qy - wy * *qx);

  *qw += dqw * dt;
  *qx += dqx * dt;
  *qy += dqy * dt;
  *qz += dqz * dt;
  
  nibble_quat_normalize(qw, qx, qy, qz);
}

/* Compute 3x3 inertia tensor from atom positions */
void nibble_compute_inertia_tensor(const NibbleMol *mol, float I[3][3]) {
  /* Initialize to zero */
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      I[i][j] = 0.0f;

  /* Assume unit mass per atom */
  for (int a = 0; a < mol->n_atoms; ++a) {
    float x = mol->local_coords[a * 3 + 0];
    float y = mol->local_coords[a * 3 + 1];
    float z = mol->local_coords[a * 3 + 2];

    I[0][0] += y * y + z * z;    /* Ixx */
    I[1][1] += x * x + z * z;    /* Iyy */
    I[2][2] += x * x + y * y;    /* Izz */
    I[0][1] -= x * y;            /* Ixy */
    I[0][2] -= x * z;            /* Ixz */
    I[1][2] -= y * z;            /* Iyz */
  }
  
  I[1][0] = I[0][1];
  I[2][0] = I[0][2];
  I[2][1] = I[1][2];
}

/* Full-channel per-atom score used for torque gradient.
 *
 * score_at() only uses CH_STERIC_DEMAND — so rotational forces were
 * driven purely by steric clashes. H-bond, electrostatic, hydrophobic,
 * and Fukui channels had zero influence on torque, meaning the molecule
 * could not rotate toward a good H-bond geometry.
 *
 * This function evaluates the full Hadamard sum over all N_CHANNELS
 * for a given atom at position (ax, ay, az) with its own channel vector.
 * It is used only for torque gradient — translation still uses the
 * simpler score_at() at COM.
 */
static float score_atom_full(const NibbleGrid *pocket,
                              float ax, float ay, float az,
                              const float *atom_ch) {
  float inv_res = 1.0f / pocket->resolution;
  int ix = (int)(ax * inv_res);
  int iy = (int)(ay * inv_res);
  int iz = (int)(az * inv_res);
  float radius  = 1.5f;
  int   r_vox   = (int)(radius * inv_res) + 1;
  float r2inv   = 1.0f / (2.0f * radius * radius);
  float lim_sq  = 4.0f * radius * radius;
  int   nc      = N_CHANNELS;

  int x0 = ix - r_vox; if (x0 < 0)                 x0 = 0;
  int x1 = ix + r_vox; if (x1 >= pocket->dim_x)    x1 = pocket->dim_x - 1;
  int y0 = iy - r_vox; if (y0 < 0)                 y0 = 0;
  int y1 = iy + r_vox; if (y1 >= pocket->dim_y)    y1 = pocket->dim_y - 1;
  int z0 = iz - r_vox; if (z0 < 0)                 z0 = 0;
  int z1 = iz + r_vox; if (z1 >= pocket->dim_z)    z1 = pocket->dim_z - 1;

  float score = 0.0f;
  int dy = pocket->dim_y, dz = pocket->dim_z;

  for (int nx = x0; nx <= x1; ++nx) {
    float vx  = nx * pocket->resolution;
    float dxs = (ax - vx) * (ax - vx);
    if (dxs > lim_sq) continue;
    for (int ny = y0; ny <= y1; ++ny) {
      float vy  = ny * pocket->resolution;
      float dxy = dxs + (ay - vy) * (ay - vy);
      if (dxy > lim_sq) continue;
      for (int nz = z0; nz <= z1; ++nz) {
        float vz = nz * pocket->resolution;
        float d2 = dxy + (az - vz) * (az - vz);
        if (d2 >= lim_sq) continue;

        float w   = expf(-d2 * r2inv);
        size_t idx = (size_t)(nx * dy * dz + ny * dz + nz) * nc;
        const float *pp = &pocket->data[idx];
        float ps = pp[CH_STERIC_DEMAND];
        float ds = atom_ch[CH_STERIC_DEMAND];

        if (ds > 0.1f && ps < 0.05f) {
          score -= 500.0f * w * ds;
        } else {
          /* Full Hadamard sum with same cross-complementary rules
           * as nibble_compute_affinity, scaled by Gaussian weight w */
          float s = ps * ds;                          /* steric         */
          s += pp[1]  * atom_ch[1];                  /* electrostatics */
          s += pp[2]  * atom_ch[2];                  /* HBA            */
          s += pp[3]  * atom_ch[3];                  /* HBD            */
          s += pp[4]  * atom_ch[4];                  /* lipo           */
          s += pp[5]  * atom_ch[5];                  /* arom           */
          s += pp[6]  * atom_ch[6];                  /* metal          */
          s += pp[7]  * atom_ch[7];                  /* cation         */
          s += pp[8]  * atom_ch[8];                  /* anion          */
          s += pp[9]  * atom_ch[9];                  /* phobic core    */
          s += pp[10] * atom_ch[10];                 /* solvent expo   */
          s += 0.5f  * pp[11] * ds;                 /* flexibility    */
          s += 0.8f  * (pp[12] * atom_ch[13]        /* Fukui cross    */
                      + pp[13] * atom_ch[12]);
          s -= 0.3f  * pp[14] * ds;                 /* conserved H2O  */
          s -= 0.1f  * pp[15] * ds;                 /* dynamic H2O    */
          score += w * s;
        }
      }
    }
  }
  return score;
}

/* Full-channel gradient for a single atom (used for torque computation) */
static void atom_gradient_full(const NibbleGrid *pocket,
                                float ax, float ay, float az,
                                const float *atom_ch,
                                float *gx, float *gy, float *gz) {
  float d = pocket->resolution;
  *gx = (score_atom_full(pocket, ax+d, ay,   az,   atom_ch)
       - score_atom_full(pocket, ax-d, ay,   az,   atom_ch)) / (2.0f * d);
  *gy = (score_atom_full(pocket, ax,   ay+d, az,   atom_ch)
       - score_atom_full(pocket, ax,   ay-d, az,   atom_ch)) / (2.0f * d);
  *gz = (score_atom_full(pocket, ax,   ay,   az+d, atom_ch)
       - score_atom_full(pocket, ax,   ay,   az-d, atom_ch)) / (2.0f * d);
}

/* Compute body torque from per-atom gradients using full channel scoring.
 *
 * Previous bug: used score_at() which only reads CH_STERIC_DEMAND.
 * Rotation was driven purely by steric — H-bond and other channels
 * had zero influence on orientation.
 *
 * Fix: use atom_gradient_full() which evaluates the complete demand
 * field for each atom's channel vector. Molecules now rotate toward
 * H-bond geometries, not just away from steric clashes.
 */
void nibble_compute_body_torque(const NibbleMol *mol, const NibbleGrid *pocket,
                                 float *tau_x, float *tau_y, float *tau_z) {
  *tau_x = 0.0f;
  *tau_y = 0.0f;
  *tau_z = 0.0f;

  for (int a = 0; a < mol->n_atoms; ++a) {
    float atom_x = mol->cx + mol->local_coords[a * 3 + 0];
    float atom_y = mol->cy + mol->local_coords[a * 3 + 1];
    float atom_z = mol->cz + mol->local_coords[a * 3 + 2];

    float gx, gy, gz;
    atom_gradient_full(pocket, atom_x, atom_y, atom_z,
                       &mol->ch_vals[a * N_CHANNELS],
                       &gx, &gy, &gz);

    /* r relative to COM */
    float rx = mol->local_coords[a * 3 + 0];
    float ry = mol->local_coords[a * 3 + 1];
    float rz = mol->local_coords[a * 3 + 2];

    /* tau += r x F */
    *tau_x += ry * gz - rz * gy;
    *tau_y += rz * gx - rx * gz;
    *tau_z += rx * gy - ry * gx;
  }
}

/* Rotate all atoms using the current quaternion applied to the REFERENCE
 * (initial) coordinates, not the already-rotated coords.
 *
 * Previous bug: local_coords were rotated in-place each step, so after
 * N steps the atoms had been rotated N times by a cumulative quaternion —
 * equivalent to rotating by q^N instead of q. This caused the molecule
 * to spin out of control over a trajectory.
 *
 * Fix: mol->ref_coords stores the original atom positions (set at creation).
 * Each step we rotate ref_coords by the current cumulative quaternion into
 * local_coords. local_coords are used only for scoring and gradient
 * evaluation, never as the source for the next rotation.
 */
void nibble_rotate_mol_atoms(NibbleMol *mol) {
  for (int a = 0; a < mol->n_atoms; ++a) {
    float x = mol->ref_coords[a * 3 + 0];
    float y = mol->ref_coords[a * 3 + 1];
    float z = mol->ref_coords[a * 3 + 2];

    nibble_quat_rotate_point(x, y, z,
                              mol->qw, mol->qx, mol->qy, mol->qz,
                              &mol->local_coords[a * 3 + 0],
                              &mol->local_coords[a * 3 + 1],
                              &mol->local_coords[a * 3 + 2]);
  }
}

NibbleMol *nibble_mol_create(int n_atoms, const float *local_coords,
                             const float *ch_vals) {
  NibbleMol *mol = (NibbleMol *)malloc(sizeof(NibbleMol));
  if (!mol)
    return NULL;
  mol->n_atoms = n_atoms;
  mol->cx = mol->cy = mol->cz = 0.0f;
  mol->vx = mol->vy = mol->vz = 0.0f;
  mol->qw = 1.0f;
  mol->qx = mol->qy = mol->qz = 0.0f;
  mol->wx = mol->wy = mol->wz = 0.0f;
  mol->local_coords = (float *)malloc(n_atoms * 3 * sizeof(float));
  mol->ref_coords   = (float *)malloc(n_atoms * 3 * sizeof(float));
  mol->ch_vals      = (float *)malloc(n_atoms * N_CHANNELS * sizeof(float));
  if (!mol->local_coords || !mol->ref_coords || !mol->ch_vals) {
    nibble_mol_free(mol);
    return NULL;
  }
  memcpy(mol->local_coords, local_coords, n_atoms * 3 * sizeof(float));
  memcpy(mol->ref_coords,   local_coords, n_atoms * 3 * sizeof(float));
  memcpy(mol->ch_vals,      ch_vals,      n_atoms * N_CHANNELS * sizeof(float));
  return mol;
}

void nibble_mol_free(NibbleMol *mol) {
  if (!mol)
    return;
  if (mol->local_coords) free(mol->local_coords);
  if (mol->ref_coords)   free(mol->ref_coords);
  if (mol->ch_vals)      free(mol->ch_vals);
  free(mol);
}

void nibble_mol_project(const NibbleMol *mol, NibbleGrid *drug_grid) {
  for (int i = 0; i < mol->n_atoms; ++i) {
    float ax = mol->cx + mol->local_coords[i * 3 + 0];
    float ay = mol->cy + mol->local_coords[i * 3 + 1];
    float az = mol->cz + mol->local_coords[i * 3 + 2];
    nibble_project_atom(drug_grid, ax, ay, az, 1.5f,
                        &mol->ch_vals[i * N_CHANNELS]);
  }
}

float nibble_mol_score(const NibbleMol *mol, const NibbleGrid *pocket,
                       NibbleGrid *scratch_drug) {
  /* Optimize: only zero the bounding box of the molecule */
  if (mol->n_atoms == 0)
    return 0.0f;
  float min_x = 1e9f, max_x = -1e9f;
  float min_y = 1e9f, max_y = -1e9f;
  float min_z = 1e9f, max_z = -1e9f;
  for (int i = 0; i < mol->n_atoms; ++i) {
    float ax = mol->cx + mol->local_coords[i * 3 + 0];
    float ay = mol->cy + mol->local_coords[i * 3 + 1];
    float az = mol->cz + mol->local_coords[i * 3 + 2];
    if (ax < min_x)
      min_x = ax;
    if (ax > max_x)
      max_x = ax;
    if (ay < min_y)
      min_y = ay;
    if (ay > max_y)
      max_y = ay;
    if (az < min_z)
      min_z = az;
    if (az > max_z)
      max_z = az;
  }
  float radius = 1.5f;
  float inv_res = 1.0f / scratch_drug->resolution;
  int r_vox = (int)(radius * inv_res) + 1;
  int x0 = (int)(min_x * inv_res) - r_vox;
  if (x0 < 0)
    x0 = 0;
  int x1 = (int)(max_x * inv_res) + r_vox;
  if (x1 >= scratch_drug->dim_x)
    x1 = scratch_drug->dim_x - 1;
  int y0 = (int)(min_y * inv_res) - r_vox;
  if (y0 < 0)
    y0 = 0;
  int y1 = (int)(max_y * inv_res) + r_vox;
  if (y1 >= scratch_drug->dim_y)
    y1 = scratch_drug->dim_y - 1;
  int z0 = (int)(min_z * inv_res) - r_vox;
  if (z0 < 0)
    z0 = 0;
  int z1 = (int)(max_z * inv_res) + r_vox;
  if (z1 >= scratch_drug->dim_z)
    z1 = scratch_drug->dim_z - 1;

  int dy = scratch_drug->dim_y;
  int dz = scratch_drug->dim_z;
  int nc = N_CHANNELS;

  /* Zero out ONLY the bounding box in the scratch drug grid */
  for (int nx = x0; nx <= x1; ++nx) {
    for (int ny = y0; ny <= y1; ++ny) {
      size_t idx = (size_t)(nx * dy * dz + ny * dz + z0) * nc;
      size_t count = (z1 - z0 + 1) * nc;
      memset(&scratch_drug->data[idx], 0, count * sizeof(float));
    }
  }

  nibble_mol_project(mol, scratch_drug);

  /* Compute affinity ONLY inside the bounding box,
   * using the same channel rules as nibble_compute_affinity:
   *   CH_FLEXIBILITY    : pocket multiplier on drug steric (not self-dot)
   *   CH_NUCL/ELEC      : cross-complementary (pocket_nucl * drug_elec)
   *   CH_WATER_CONSERVED: displacement cost -0.3 * p_con * ds
   *   CH_WATER_DYNAMIC  : displacement cost -0.1 * p_dyn * ds
   * Previous code used raw pp[c]*dp[c] for all channels including 11-15,
   * which was inconsistent with nibble_compute_affinity. */
  float score = 0.0f;
  for (int nx = x0; nx <= x1; ++nx) {
    for (int ny = y0; ny <= y1; ++ny) {
      for (int nz = z0; nz <= z1; ++nz) {
        size_t idx = (size_t)(nx * dy * dz + ny * dz + nz) * nc;
        const float *pp = &pocket->data[idx];
        const float *dp = &scratch_drug->data[idx];

        float ps = pp[CH_STERIC_DEMAND];
        float ds = dp[CH_STERIC_DEMAND];

        if (ds > 0.1f && ps < 0.05f) {
          score -= 500.0f * ds;
        } else {
          score += ps      * ds;
          score += pp[1]  * dp[1];
          score += pp[2]  * dp[2];
          score += pp[3]  * dp[3];
          score += pp[4]  * dp[4];
          score += pp[5]  * dp[5];
          score += pp[6]  * dp[6];
          score += pp[7]  * dp[7];
          score += pp[8]  * dp[8];
          score += pp[9]  * dp[9];
          score += pp[10] * dp[10];
          score += 0.5f  * pp[11] * ds;                    /* flexibility multiplier  */
          score += 0.8f  * (pp[12] * dp[13] + pp[13] * dp[12]); /* Fukui cross        */
          score -= 0.3f  * pp[14] * ds;                    /* conserved water cost    */
          score -= 0.1f  * pp[15] * ds;                    /* dynamic water cost      */
        }
      }
    }
  }
  return score;
}

void nibble_langevin_step(NibbleMol *mol, const NibbleGrid *pocket,
                          NibbleGrid *scratch_drug, float dt, float gamma,
                          float kT, unsigned int *rng_state) {
  (void)scratch_drug; /* used by nibble_trajectory via nibble_mol_score, not here */
  float gx, gy, gz;
  nibble_gradient(pocket, mol->cx, mol->cy, mol->cz, &gx, &gy, &gz);

  float noise_scale   = sqrtf(2.0f * gamma * kT * dt);
  float exp_gdt       = expf(-gamma * dt);
  float gamma_rot     = gamma / 3.0f;
  float exp_gdt_rot   = expf(-gamma_rot * dt);
  float noise_rot_sc  = sqrtf(2.0f * gamma_rot * kT * dt);

  /* ========== TRANSLATIONAL DYNAMICS ========== */
  mol->vx = mol->vx * exp_gdt + gx * dt + noise_scale * rng_normal(rng_state);
  mol->vy = mol->vy * exp_gdt + gy * dt + noise_scale * rng_normal(rng_state);
  mol->vz = mol->vz * exp_gdt + gz * dt + noise_scale * rng_normal(rng_state);

  mol->cx += mol->vx * dt;
  mol->cy += mol->vy * dt;
  mol->cz += mol->vz * dt;

  /* ========== ROTATIONAL DYNAMICS ========== */
  float tau_x, tau_y, tau_z;
  nibble_compute_body_torque(mol, pocket, &tau_x, &tau_y, &tau_z);

  float I[3][3];
  nibble_compute_inertia_tensor(mol, I);

  /*
   * Full inertia tensor angular acceleration: alpha = I^{-1} * tau
   *
   * Previous code used only diagonal elements I[0][0], I[1][1], I[2][2],
   * ignoring off-diagonal coupling terms. For non-spherical molecules
   * (every real drug) the off-diagonal Ixy, Ixz, Iyz terms couple the
   * rotation axes — ignoring them gives wrong angular acceleration
   * whenever the molecule is elongated, L-shaped, or asymmetric.
   *
   * We solve I * alpha = tau via Cramer's rule (exact for 3x3, O(1)).
   * If det(I) is near zero (degenerate — e.g. linear molecule), we fall
   * back to the diagonal approximation.
   *
   * Cramer's rule: alpha_i = det(I with column i replaced by tau) / det(I)
   */
  float det = I[0][0] * (I[1][1]*I[2][2] - I[1][2]*I[2][1])
            - I[0][1] * (I[1][0]*I[2][2] - I[1][2]*I[2][0])
            + I[0][2] * (I[1][0]*I[2][1] - I[1][1]*I[2][0]);

  float ax_rot, ay_rot, az_rot;
  if (fabsf(det) > 1e-6f) {
    float inv_det = 1.0f / det;

    /* alpha_x = det(I with col0 replaced by tau) / det */
    float det_x = tau_x * (I[1][1]*I[2][2] - I[1][2]*I[2][1])
                - I[0][1]* (tau_y *I[2][2]  - I[1][2]*tau_z )
                + I[0][2]* (tau_y *I[2][1]  - I[1][1]*tau_z );
    /* alpha_y */
    float det_y = I[0][0]* (tau_y *I[2][2]  - I[1][2]*tau_z )
                - tau_x *  (I[1][0]*I[2][2]  - I[1][2]*I[2][0])
                + I[0][2]* (I[1][0]*tau_z    - tau_y  *I[2][0]);
    /* alpha_z */
    float det_z = I[0][0]* (I[1][1]*tau_z    - tau_y  *I[2][1])
                - I[0][1]* (I[1][0]*tau_z    - tau_y  *I[2][0])
                + tau_x *  (I[1][0]*I[2][1]  - I[1][1]*I[2][0]);

    ax_rot = det_x * inv_det;
    ay_rot = det_y * inv_det;
    az_rot = det_z * inv_det;
  } else {
    /* Degenerate fallback: diagonal only */
    float Ixx = I[0][0] < 1e-9f ? 1.0f : I[0][0];
    float Iyy = I[1][1] < 1e-9f ? 1.0f : I[1][1];
    float Izz = I[2][2] < 1e-9f ? 1.0f : I[2][2];
    ax_rot = tau_x / Ixx;
    ay_rot = tau_y / Iyy;
    az_rot = tau_z / Izz;
  }

  /* Langevin angular velocity update */
  mol->wx = mol->wx * exp_gdt_rot + ax_rot * dt
          + noise_rot_sc * rng_normal(rng_state);
  mol->wy = mol->wy * exp_gdt_rot + ay_rot * dt
          + noise_rot_sc * rng_normal(rng_state);
  mol->wz = mol->wz * exp_gdt_rot + az_rot * dt
          + noise_rot_sc * rng_normal(rng_state);

  nibble_quat_integrate(&mol->qw, &mol->qx, &mol->qy, &mol->qz,
                        mol->wx, mol->wy, mol->wz, dt);

  /* Rotate from reference frame — not in-place (drift fix) */
  nibble_rotate_mol_atoms(mol);

  /* ========== BOUNDARY CONDITIONS ========== */
  float max_x = (pocket->dim_x - 2) * pocket->resolution;
  float max_y = (pocket->dim_y - 2) * pocket->resolution;
  float max_z = (pocket->dim_z - 2) * pocket->resolution;
  if (mol->cx < 0)      mol->cx = 0;
  if (mol->cx > max_x)  mol->cx = max_x;
  if (mol->cy < 0)      mol->cy = 0;
  if (mol->cy > max_y)  mol->cy = max_y;
  if (mol->cz < 0)      mol->cz = 0;
  if (mol->cz > max_z)  mol->cz = max_z;
}

float nibble_trajectory(NibbleMol *mol, const NibbleGrid *pocket,
                        NibbleGrid *scratch_drug, int n_steps, float dt,
                        float gamma, float kT, float *best_cx_out,
                        float *best_cy_out, float *best_cz_out) {
  /*
   * RNG seed from pointer address XOR'd with n_steps for variety.
   * Previous code always used 12345u — every trajectory was identical,
   * making thermal noise useless as an escape mechanism.
   */
  unsigned int rng = (unsigned int)(size_t)mol ^ (unsigned int)n_steps ^ 0xDEADBEEFu;

  float best_score = nibble_mol_score(mol, pocket, scratch_drug);
  float bcx = mol->cx, bcy = mol->cy, bcz = mol->cz;
  /* Save best quaternion — previously only COM was saved, orientation lost */
  float bqw = mol->qw, bqx = mol->qx, bqy = mol->qy, bqz = mol->qz;

  for (int s = 0; s < n_steps; ++s) {
    nibble_langevin_step(mol, pocket, scratch_drug, dt, gamma, kT, &rng);
    float sc = nibble_mol_score(mol, pocket, scratch_drug);
    if (sc > best_score) { /* Higher = better binding */
      best_score = sc;
      bcx = mol->cx; bcy = mol->cy; bcz = mol->cz;
      bqw = mol->qw; bqx = mol->qx; bqy = mol->qy; bqz = mol->qz;
    }
  }

  /* Restore best pose: both position AND orientation */
  mol->cx = bcx; mol->cy = bcy; mol->cz = bcz;
  mol->qw = bqw; mol->qx = bqx; mol->qy = bqy; mol->qz = bqz;
  nibble_rotate_mol_atoms(mol); /* re-apply best orientation */

  if (best_cx_out) *best_cx_out = bcx;
  if (best_cy_out) *best_cy_out = bcy;
  if (best_cz_out) *best_cz_out = bcz;
  return best_score;
}
