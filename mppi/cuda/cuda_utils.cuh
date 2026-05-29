#pragma once

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>

// ============================================================
// CUDA Error Macros
// ============================================================
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d  %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(_err));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(call)                                                     \
  do {                                                                         \
    curandStatus_t _err = (call);                                              \
    if (_err != CURAND_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuRAND error at %s:%d  code=%d\n", __FILE__, __LINE__,  \
              (int)_err);                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================
// WMRobot Device Functions (CUDA compiler only)
// State: x = [px, py, theta]  Control: u = [v, omega]
// Memory layout for controls: Ui[i * dim_u * T + d * T + t]
// ============================================================
#ifdef __CUDACC__

__device__ __forceinline__ void f_wmrobot(const double *x, double v,
                                          double omega, double *x_dot) {
  x_dot[0] = v * cos(x[2]);
  x_dot[1] = v * sin(x[2]);
  x_dot[2] = omega;
}

__device__ __forceinline__ double p_wmrobot(const double *x,
                                            const double *x_target, int dim_x) {
  double s = 0.0;
  for (int d = 0; d < dim_x; ++d) {
    double diff = x[d] - x_target[d];
    s += diff * diff;
  }
  return sqrt(s);
}

// Clamp a single (dim_u x T) block in-place
__device__ __forceinline__ void h_wmrobot(double *u, int T) {
  for (int t = 0; t < T; ++t) {
    u[0 * T + t] = fmax(0.0, fmin(1.0, u[0 * T + t]));
    u[1 * T + t] = fmax(-M_PI / 2.0, fmin(M_PI / 2.0, u[1 * T + t]));
  }
}

__device__ __forceinline__ void f_quadrotor(const double *x, const double *u,
                                            double *x_dot) {
  x_dot[0] = x[3];
  x_dot[1] = x[4];
  x_dot[2] = x[5];
  x_dot[3] = u[0];
  x_dot[4] = u[1];
  x_dot[5] = u[2] - 9.81;
}

__device__ __forceinline__ double
p_quadrotor(const double *x, const double *x_target, int dim_x) {
  double s = 0.0;
  for (int d = 0; d < 3; ++d) {
    double diff = x[d] - x_target[d];
    s += diff * diff;
  }
  return sqrt(s);
}

__device__ __forceinline__ void h_quadrotor(double *u, int T) {
  for (int t = 0; t < T; ++t) {
    double u0 = u[0 * T + t];
    double u1 = u[1 * T + t];
    double u2 = u[2 * T + t];

    double norm_u = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
    if (norm_u >= 20.0) {
      double mask = 1.0 / norm_u;
      u0 *= mask;
      u1 *= mask;
      u2 *= mask;
    }

    double V_val = sqrt(u0 * u0 + u1 * u1);
    double S_val = u2 * 1.73205080756887729; // tan(M_PI / 3.0)

    if (V_val < -S_val) {
      u0 = 0.0;
      u1 = 0.0;
      u2 = 0.0;
      V_val = 0.0;
      S_val = 0.0;
    }

    if (V_val > fabs(S_val)) {
      double mul = 0.5 * (1.0 + S_val / V_val);
      u0 = u0 * mul;
      u1 = u1 * mul;
      u2 = V_val * mul;
    }

    u[0 * T + t] = u0;
    u[1 * T + t] = u1;
    u[2 * T + t] = u2;
  }
}

__device__ __forceinline__ void f_velo(const double *x, const double *u,
                                       double *x_dot) {
  x_dot[0] = x[2];
  x_dot[1] = x[3];
  x_dot[2] = u[0];
  x_dot[3] = u[1];
}

__device__ __forceinline__ double p_velo(const double *x,
                                         const double *x_target, int dim_x) {
  double s = 0.0;
  for (int d = 0; d < 2; ++d) {
    double diff = x[d] - x_target[d];
    s += diff * diff;
  }
  return sqrt(s);
}

__device__ __forceinline__ void h_velo(double *u, int T) {
  for (int t = 0; t < T; ++t) {
    u[0 * T + t] = fmax(0.0, fmin(1.0, u[0 * T + t]));
    u[1 * T + t] = fmax(0.0, fmin(1.0, u[1 * T + t]));
  }
}

// ============================================================
// Manipulator Device Functions (CUDA compiler only)
// State: x = [q0..q5]  Control: u = [q_dot0..q_dot5]
// 1st-order kinematic: x_dot = u
// ============================================================
__device__ __forceinline__ void f_manipulator(const double *x, const double *u,
                                              double *x_dot, int dim_u) {
  for (int d = 0; d < dim_u; ++d)
    x_dot[d] = u[d];
}

__device__ __forceinline__ double p_manipulator(const double *x,
                                                const double *x_target,
                                                int dim_x) {
  double s = 0.0;
  for (int d = 0; d < dim_x; ++d) {
    double diff = x[d] - x_target[d];
    s += diff * diff;
  }
  return sqrt(s);
}

// q_dot_max: [1.5, 1.5, 1.5, 2.0, 2.0, 2.0] rad/s (RB5 limits)
__device__ __forceinline__ void h_manipulator(double *u, int dim_u, int T) {
  const double q_dot_max[6] = {1.5, 1.5, 1.5, 2.0, 2.0, 2.0};
  for (int d = 0; d < dim_u && d < 6; ++d)
    for (int t = 0; t < T; ++t)
      u[d * T + t] = fmax(-q_dot_max[d], fmin(q_dot_max[d], u[d * T + t]));
}

// ============================================================
// Collision Detection Device Functions
// ============================================================

__device__ __forceinline__ bool collision_grid_map(const double *x,
                                                   const double *d_map,
                                                   int max_row, int max_col,
                                                   double resolution) {
  int nx = (int)round(x[0] / resolution);
  int ny = (int)round(x[1] / resolution);
  if (nx < 0 || nx >= max_row)
    return true;
  if (ny < 0 || ny >= max_col)
    return false;
  return (d_map[nx * max_col + ny] == 10.0);
}

__device__ __forceinline__ bool
collision_polygon(const double *x, const double *circles, int n_circles,
                  const double *rects, int n_rects) {
  for (int i = 0; i < n_circles; ++i) {
    double dx = circles[i * 4 + 0] - x[0];
    double dy = circles[i * 4 + 1] - x[1];
    if (dx * dx + dy * dy <= circles[i * 4 + 3])
      return true;
  }
  for (int i = 0; i < n_rects; ++i) {
    if (x[0] >= rects[i * 4 + 0] && x[0] <= rects[i * 4 + 1] &&
        x[1] >= rects[i * 4 + 2] && x[1] <= rects[i * 4 + 3])
      return true;
  }
  return false;
}

__device__ __forceinline__ bool
check_collision(const double *x, bool with_map, const double *d_map,
                int max_row, int max_col, double resolution,
                const double *circles, int n_circles, const double *rects,
                int n_rects) {
  if (with_map)
    return collision_grid_map(x, d_map, max_row, max_col, resolution);
  return collision_polygon(x, circles, n_circles, rects, n_rects);
}

// ============================================================
// GPU Reduction Utilities
// ============================================================

// Warp-level min reduction
__device__ __forceinline__ double warp_reduce_min(double val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

// Warp-level sum reduction
__device__ __forceinline__ double warp_reduce_sum(double val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// Shared forward/weighted-sum kernels are declared in shared_kernels.cuh
// and defined in shared_kernels.cu (single TU to avoid ODR violations).

// ============================================================
// Kernel body macros: each .cu defines its own local kernel
// using these macros to avoid ODR violations across TUs.
// ============================================================
#define DEFINE_FORWARD_ROLLOUT_KERNEL(KERNEL_NAME)                             \
  __global__ void KERNEL_NAME(                                                 \
      const double *__restrict__ d_U0, double *d_Ui,                           \
      const double *__restrict__ d_noise, const double *__restrict__ d_sigma,  \
      const double *__restrict__ d_x_init,                                     \
      const double *__restrict__ d_x_target, double *d_costs, double *d_Di,    \
      bool with_map, const double *d_map, int max_row, int max_col,            \
      double res, const double *d_circles, int n_circ, const double *d_rects,  \
      int n_rect, int N, int dim_u, int dim_x, int T, double dt,               \
      double gamma_u, int model_type) {                                        \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= N)                                                                \
      return;                                                                  \
    double *Ui_i = d_Ui + i * (dim_u * T);                                     \
    for (int d = 0; d < dim_u; ++d)                                            \
      for (int t = 0; t < T; ++t)                                              \
        Ui_i[d * T + t] = d_U0[d * T + t] +                                    \
                          d_sigma[d] * d_noise[i * (dim_u * T) + d * T + t];   \
    if (model_type == 0) {                                                     \
      h_wmrobot(Ui_i, T);                                                      \
    } else if (model_type == 1) {                                              \
      h_quadrotor(Ui_i, T);                                                    \
    } else if (model_type == 2) {                                              \
      h_velo(Ui_i, T);                                                         \
    } else if (model_type == 3) {                                              \
      h_manipulator(Ui_i, dim_u, T);                                           \
    }                                                                          \
    double *Di_i = d_Di + i * dim_u;                                           \
    for (int d = 0; d < dim_u; ++d) {                                          \
      double acc = 0.0;                                                        \
      for (int t = 0; t < T; ++t)                                              \
        acc += Ui_i[d * T + t] - d_U0[d * T + t];                              \
      Di_i[d] = acc / T;                                                       \
    }                                                                          \
    double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd_[GPU_MAX_DIM_X];            \
    for (int d = 0; d < dim_x; ++d)                                            \
      x[d] = d_x_init[d];                                                      \
    double cost = 0.0;                                                         \
    bool hit = false;                                                          \
    for (int t = 0; t < T; ++t) {                                              \
      if (model_type == 0) {                                                   \
        cost += p_wmrobot(x, d_x_target, dim_x);                               \
        double v = Ui_i[0 * T + t], omega = Ui_i[1 * T + t];                   \
        f_wmrobot(x, v, omega, xd_);                                           \
      } else if (model_type == 1) {                                            \
        cost += p_quadrotor(x, d_x_target, dim_x);                             \
        double u_val[3] = {Ui_i[0 * T + t], Ui_i[1 * T + t], Ui_i[2 * T + t]}; \
        f_quadrotor(x, u_val, xd_);                                            \
      } else if (model_type == 2) {                                            \
        cost += p_velo(x, d_x_target, dim_x);                                  \
        double u_val2[2] = {Ui_i[0 * T + t], Ui_i[1 * T + t]};                 \
        f_velo(x, u_val2, xd_);                                                \
      } else if (model_type == 3) {                                            \
        cost += p_manipulator(x, d_x_target, dim_x);                           \
        double u_manip[GPU_MAX_DIM_U];                                          \
        for (int _d = 0; _d < dim_u; ++_d)                                     \
          u_manip[_d] = Ui_i[_d * T + t];                                      \
        f_manipulator(x, u_manip, xd_, dim_u);                                 \
      }                                                                        \
      for (int d = 0; d < dim_x; ++d)                                          \
        xn[d] = x[d] + dt * xd_[d];                                            \
      if (!hit && check_collision(x, with_map, d_map, max_row, max_col, res,   \
                                  d_circles, n_circ, d_rects, n_rect)) {       \
        hit = true;                                                            \
        cost = 1e8;                                                            \
      }                                                                        \
      for (int d = 0; d < dim_x; ++d)                                          \
        x[d] = xn[d];                                                          \
    }                                                                          \
    if (!hit) {                                                                \
      if (model_type == 0) {                                                   \
        cost += p_wmrobot(x, d_x_target, dim_x);                               \
      } else if (model_type == 1) {                                            \
        cost += p_quadrotor(x, d_x_target, dim_x);                             \
      } else if (model_type == 2) {                                            \
        cost += p_velo(x, d_x_target, dim_x);                                  \
      } else if (model_type == 3) {                                            \
        cost += p_manipulator(x, d_x_target, dim_x);                           \
      }                                                                        \
      if (check_collision(x, with_map, d_map, max_row, max_col, res,           \
                          d_circles, n_circ, d_rects, n_rect))                 \
        cost = 1e8;                                                            \
    }                                                                          \
    d_costs[i] = cost;                                                         \
  }

#define DEFINE_WEIGHTED_SUM_KERNEL(KERNEL_NAME)                                \
  __global__ void KERNEL_NAME(const double *d_Ui, const double *d_costs,       \
                              double min_cost, double gamma_u, double *d_Uo,   \
                              int N, int dim_u, int T) {                       \
    int dt_idx = blockIdx.x;                                                   \
    int d = dt_idx / T;                                                        \
    int t = dt_idx % T;                                                        \
    if (d >= dim_u)                                                            \
      return;                                                                  \
    extern __shared__ double sdata[];                                          \
    double *s_wsum = sdata;                                                    \
    double *s_wctrl = sdata + blockDim.x;                                      \
    int tid = threadIdx.x;                                                     \
    double w_acc = 0.0, wc_acc = 0.0;                                          \
    for (int i = tid; i < N; i += blockDim.x) {                                \
      double w = exp(-gamma_u * (d_costs[i] - min_cost));                      \
      w_acc += w;                                                              \
      wc_acc += w * d_Ui[i * (dim_u * T) + d * T + t];                         \
    }                                                                          \
    s_wsum[tid] = w_acc;                                                       \
    s_wctrl[tid] = wc_acc;                                                     \
    __syncthreads();                                                           \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                             \
      if (tid < s) {                                                           \
        s_wsum[tid] += s_wsum[tid + s];                                        \
        s_wctrl[tid] += s_wctrl[tid + s];                                      \
      }                                                                        \
      __syncthreads();                                                         \
    }                                                                          \
    if (tid == 0)                                                              \
      d_Uo[dt_idx] = s_wctrl[0] / s_wsum[0];                                   \
  }

#endif // __CUDACC__
