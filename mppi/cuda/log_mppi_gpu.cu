#include "log_mppi_gpu.cuh"
#include <cassert>
#include <cstring>
#include <limits>
#include <algorithm>

// ============================================================
// GPU Kernel: Log-MPPI rollout
//
// Noise model (per Log-MPPI paper):
//   Ui(d,t) = U0(d,t) + σ_d * η(d,t) * exp(ξ(d,t))
// where η ~ N(0,1) (d_noise) and ξ ~ N(0,1) (d_log_noise).
// The lognormal factor exp(ξ) scales the normal perturbation
// so that rare large-magnitude explorations are promoted.
// ============================================================

#ifndef GPU_MAX_DIM_X
#define GPU_MAX_DIM_X 16
#endif

__global__ void log_mppi_rollout_kernel(
    const double* __restrict__ d_U0,
    double*       d_Ui,
    const double* __restrict__ d_noise,      // N(0,1) for normal part
    const double* __restrict__ d_log_noise,  // N(0,1) for lognormal exponent
    const double* __restrict__ d_sigma,
    const double* __restrict__ d_x_init,
    const double* __restrict__ d_x_target,
    double* d_costs,
    double* d_Di,
    bool with_map, const double* d_map,
    int max_row, int max_col, double res,
    const double* d_circles, int n_circ,
    const double* d_rects,   int n_rect,
    int N, int dim_u, int dim_x, int T, double dt, double gamma_u,
    int model_type)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  double* Ui_i = d_Ui + i * (dim_u * T);

  // Build perturbed control with Log-MPPI noise
  for (int d = 0; d < dim_u; ++d)
    for (int t = 0; t < T; ++t) {
      double eta  = d_noise    [i * (dim_u * T) + d * T + t];  // N(0,1)
      double xi   = d_log_noise[i * (dim_u * T) + d * T + t];  // N(0,1)
      double log_factor = exp(xi);                               // lognormal
      Ui_i[d * T + t] = d_U0[d * T + t] + d_sigma[d] * eta * log_factor;
    }

  // Clamp controls
  if (model_type == 0) {
    h_wmrobot(Ui_i, T);
  } else if (model_type == 1) {
    h_quadrotor(Ui_i, T);
  } else if (model_type == 2) {
    h_velo(Ui_i, T);
  } else if (model_type == 3) {
    h_manipulator(Ui_i, dim_u, T);
  }

  // Mean deviation Di
  double* Di_i = d_Di + i * dim_u;
  for (int d = 0; d < dim_u; ++d) {
    double acc = 0.0;
    for (int t = 0; t < T; ++t) acc += Ui_i[d * T + t] - d_U0[d * T + t];
    Di_i[d] = acc / T;
  }

  // Rollout
  double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd_[GPU_MAX_DIM_X];
  for (int d = 0; d < dim_x; ++d) x[d] = d_x_init[d];
  double cost = 0.0;
  bool hit = false;

  for (int t = 0; t < T; ++t) {
    if (model_type == 0) {
      cost += p_wmrobot(x, d_x_target, dim_x);
      double v = Ui_i[0 * T + t], omega = Ui_i[1 * T + t];
      f_wmrobot(x, v, omega, xd_);
    } else if (model_type == 1) {
      cost += p_quadrotor(x, d_x_target, dim_x);
      double u_val[3] = { Ui_i[0 * T + t], Ui_i[1 * T + t], Ui_i[2 * T + t] };
      f_quadrotor(x, u_val, xd_);
    } else if (model_type == 2) {
      cost += p_velo(x, d_x_target, dim_x);
      double u_v[2] = { Ui_i[0 * T + t], Ui_i[1 * T + t] };
      f_velo(x, u_v, xd_);
    } else if (model_type == 3) {
      cost += p_manipulator(x, d_x_target, dim_x);
      double u_manip[GPU_MAX_DIM_U];
      for (int _d = 0; _d < dim_u; ++_d) u_manip[_d] = Ui_i[_d * T + t];
      f_manipulator(x, u_manip, xd_, dim_u);
    }
    for (int d = 0; d < dim_x; ++d) xn[d] = x[d] + dt * xd_[d];
    if (!hit && check_collision(x, with_map, d_map, max_row, max_col, res,
                                d_circles, n_circ, d_rects, n_rect)) {
      hit = true; cost = 1e8;
    }
    for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
  }
  if (!hit) {
    if (model_type == 0) {
      cost += p_wmrobot(x, d_x_target, dim_x);
    } else if (model_type == 1) {
      cost += p_quadrotor(x, d_x_target, dim_x);
    } else if (model_type == 2) {
      cost += p_velo(x, d_x_target, dim_x);
    } else if (model_type == 3) {
      cost += p_manipulator(x, d_x_target, dim_x);
    }
    if (check_collision(x, with_map, d_map, max_row, max_col, res,
                        d_circles, n_circ, d_rects, n_rect))
      cost = 1e8;
  }
  d_costs[i] = cost;
}

// Weighted sum kernel (same as standard MPPI, reused locally)
DEFINE_WEIGHTED_SUM_KERNEL(log_mppi_weighted_sum_kernel)


// ============================================================
// LogMPPI_GPU implementation
// ============================================================
LogMPPI_GPU::~LogMPPI_GPU() {
  if (d_log_noise) { cudaFree(d_log_noise); d_log_noise = nullptr; }
  // base class destructor handles the rest
}

// Override: generate two independent N(0,1) buffers
void LogMPPI_GPU::generateLogNoise() {
  size_t count = (size_t)N * dim_u * T;
  size_t count_even = (count % 2 != 0) ? count + 1 : count;

  // Normal part (η)
  CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_noise, count_even, 0.0, 1.0));

  // Lognormal exponent part (ξ) — allocate on first call or after init
  if (!d_log_noise) {
    CUDA_CHECK(cudaMalloc(&d_log_noise, count_even * sizeof(double)));
  }
  CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_log_noise, count_even, 0.0, 1.0));
}

void LogMPPI_GPU::solve() {
  start = std::chrono::high_resolution_clock::now();

  uploadControl();
  uploadState();
  generateLogNoise();

  // Allocate d_log_noise if not yet done (size: N*dim_u*T, even-rounded)
  // (handled inside generateLogNoise)

  // Launch Log-MPPI rollout kernel
  const int BLOCK = 256;
  int grid = (N + BLOCK - 1) / BLOCK;
  log_mppi_rollout_kernel<<<grid, BLOCK>>>(
      d_U0, d_Ui, d_noise, d_log_noise, d_sigma,
      d_x_init, d_x_target,
      d_costs, d_Di,
      with_map, d_map, map_max_row, map_max_col, map_resolution,
      d_circles, n_circles, d_rects, n_rects,
      N, dim_u, dim_x, T, (double)dt, gamma_u, model_type);
  CUDA_CHECK(cudaGetLastError());

  // Copy costs to host for min
  std::vector<double> h_costs(N);
  CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs,
                        N * sizeof(double), cudaMemcpyDeviceToHost));
  double min_cost = *std::min_element(h_costs.begin(), h_costs.end());

  // Weighted sum
  int dt_blocks    = dim_u * T;
  int wsum_threads = std::min(256, N);
  size_t shared_sz = 2 * wsum_threads * sizeof(double);
  log_mppi_weighted_sum_kernel<<<dt_blocks, wsum_threads, shared_sz>>>(
      d_Ui, d_costs, min_cost, gamma_u, d_Uo, N, dim_u, T);
  CUDA_CHECK(cudaGetLastError());

  // Copy Uo to host
  std::vector<double> h_Uo(dim_u * T);
  CUDA_CHECK(cudaMemcpy(h_Uo.data(), d_Uo,
                        dim_u * T * sizeof(double), cudaMemcpyDeviceToHost));

  Uo.resize(dim_u, T);
  for (int d = 0; d < dim_u; ++d)
    for (int t = 0; t < T; ++t)
      Uo(d, t) = h_Uo[d * T + t];

  // h() clamping
  h(Uo);

  CUDA_CHECK(cudaDeviceSynchronize());
  finish = std::chrono::high_resolution_clock::now();
  elapsed_1          = finish - start;
  elapsed_rollout    = elapsed_1.count();
  elapsed_clustering = 0.0;
  elapsed_connection = 0.0;
  elapsed_guide      = 0.0;
  elapsed            = elapsed_rollout;

  u0 = Uo.col(0);

  // Reconstruct optimal trajectory on CPU
  Xo.col(0) = x_init;
  for (int j = 0; j < T; ++j) {
    Xo.col(j + 1) = Xo.col(j) + (double)dt * f(Xo.col(j), Uo.col(j));
  }

  visual_traj.push_back(x_init);
}
