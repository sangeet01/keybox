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

/* Compute body torque from per-atom gradients */
void nibble_compute_body_torque(const NibbleMol *mol, const NibbleGrid *pocket,
                                 float *tau_x, float *tau_y, float *tau_z) {
  *tau_x = 0.0f;
  *tau_y = 0.0f;
  *tau_z = 0.0f;

  /* Sum torques: tau = r x F for each atom */
  for (int a = 0; a < mol->n_atoms; ++a) {
    float atom_x = mol->cx + mol->local_coords[a * 3 + 0];
    float atom_y = mol->cy + mol->local_coords[a * 3 + 1];
    float atom_z = mol->cz + mol->local_coords[a * 3 + 2];

    float gx, gy, gz;
    nibble_gradient(pocket, atom_x, atom_y, atom_z, &gx, &gy, &gz);

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

/* Rotate all atoms in molecule using current quaternion */
void nibble_rotate_mol_atoms(NibbleMol *mol) {
  for (int a = 0; a < mol->n_atoms; ++a) {
    float x = mol->local_coords[a * 3 + 0];
    float y = mol->local_coords[a * 3 + 1];
    float z = mol->local_coords[a * 3 + 2];

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
  mol->ch_vals = (float *)malloc(n_atoms * N_CHANNELS * sizeof(float));
  if (!mol->local_coords || !mol->ch_vals) {
    nibble_mol_free(mol);
    return NULL;
  }
  memcpy(mol->local_coords, local_coords, n_atoms * 3 * sizeof(float));
  memcpy(mol->ch_vals, ch_vals, n_atoms * N_CHANNELS * sizeof(float));
  return mol;
}

void nibble_mol_free(NibbleMol *mol) {
  if (!mol)
    return;
  if (mol->local_coords)
    free(mol->local_coords);
  if (mol->ch_vals)
    free(mol->ch_vals);
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

  /* Compute affinity ONLY inside the bounding box */
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
          for (int c = 0; c < nc; ++c) {
            score += pp[c] * dp[c];
          }
        }
      }
    }
  }
  return score;
}

void nibble_langevin_step(NibbleMol *mol, const NibbleGrid *pocket,
                          NibbleGrid *scratch_drug, float dt, float gamma,
                          float kT, unsigned int *rng_state) {
  float gx, gy, gz;
  /* Gradient at center of mass gives force direction toward energy minimum */
  nibble_gradient(pocket, mol->cx, mol->cy, mol->cz, &gx, &gy, &gz);

  float noise_scale = sqrtf(2.0f * gamma * kT * dt);
  float exp_gdt = expf(-gamma * dt);

  /* ========== TRANSLATIONAL DYNAMICS ========== */
  /* Velocity Verlet-Langevin: v_new = v*exp(-gamma*dt) + F*dt + noise */
  mol->vx = mol->vx * exp_gdt + gx * dt + noise_scale * rng_normal(rng_state);
  mol->vy = mol->vy * exp_gdt + gy * dt + noise_scale * rng_normal(rng_state);
  mol->vz = mol->vz * exp_gdt + gz * dt + noise_scale * rng_normal(rng_state);

  mol->cx += mol->vx * dt;
  mol->cy += mol->vy * dt;
  mol->cz += mol->vz * dt;

  /* ========== ROTATIONAL DYNAMICS ========== */
  /* Compute body torque from per-atom forces */
  float tau_x, tau_y, tau_z;
  nibble_compute_body_torque(mol, pocket, &tau_x, &tau_y, &tau_z);

  /* Compute inertia tensor (diagonal approximation for efficiency) */
  float I[3][3];
  nibble_compute_inertia_tensor(mol, I);

  /* Simple diagonal inertia assumption: w_new = tau / I */
  float I_xx = I[0][0];
  float I_yy = I[1][1];
  float I_zz = I[2][2];
  
  if (I_xx < 1e-9f) I_xx = 1.0f;
  if (I_yy < 1e-9f) I_yy = 1.0f;
  if (I_zz < 1e-9f) I_zz = 1.0f;

  /* Langevin angular dynamics: w_new = w*exp(-gamma*dt) + (tau/I)*dt + noise_rot */
  float noise_rot_scale = sqrtf(2.0f * gamma * kT * dt);
  mol->wx = mol->wx * exp_gdt + (tau_x / I_xx) * dt + 
            noise_rot_scale * rng_normal(rng_state) / I_xx;
  mol->wy = mol->wy * exp_gdt + (tau_y / I_yy) * dt + 
            noise_rot_scale * rng_normal(rng_state) / I_yy;
  mol->wz = mol->wz * exp_gdt + (tau_z / I_zz) * dt + 
            noise_rot_scale * rng_normal(rng_state) / I_zz;

  /* Integrate quaternion */
  nibble_quat_integrate(&mol->qw, &mol->qx, &mol->qy, &mol->qz,
                        mol->wx, mol->wy, mol->wz, dt);

  /* Apply rotation to all atoms */
  nibble_rotate_mol_atoms(mol);

  /* ========== BOUNDARY CONDITIONS ========== */
  /* Clamp to stay inside grid bounds */
  float max_x = (pocket->dim_x - 2) * pocket->resolution;
  float max_y = (pocket->dim_y - 2) * pocket->resolution;
  float max_z = (pocket->dim_z - 2) * pocket->resolution;
  if (mol->cx < 0)
    mol->cx = 0;
  if (mol->cx > max_x)
    mol->cx = max_x;
  if (mol->cy < 0)
    mol->cy = 0;
  if (mol->cy > max_y)
    mol->cy = max_y;
  if (mol->cz < 0)
    mol->cz = 0;
  if (mol->cz > max_z)
    mol->cz = max_z;
}

float nibble_trajectory(NibbleMol *mol, const NibbleGrid *pocket,
                        NibbleGrid *scratch_drug, int n_steps, float dt,
                        float gamma, float kT, float *best_cx_out,
                        float *best_cy_out, float *best_cz_out) {
  unsigned int rng = 12345u;
  float best_score = nibble_mol_score(mol, pocket, scratch_drug);
  float bcx = mol->cx, bcy = mol->cy, bcz = mol->cz;

  for (int s = 0; s < n_steps; ++s) {
    nibble_langevin_step(mol, pocket, scratch_drug, dt, gamma, kT, &rng);
    float sc = nibble_mol_score(mol, pocket, scratch_drug);
    if (sc < best_score) { /* Lower = better binding (more negative) */
      best_score = sc;
      bcx = mol->cx;
      bcy = mol->cy;
      bcz = mol->cz;
    }
  }

  if (best_cx_out)
    *best_cx_out = bcx;
  if (best_cy_out)
    *best_cy_out = bcy;
  if (best_cz_out)
    *best_cz_out = bcz;
  return best_score;
}
