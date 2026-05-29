#include "mppi_gpu.cuh"
// cuda_utils.cuh에 shared_forward_rollout_kernel, shared_weighted_sum_kernel 정의됨
#include <cassert>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>

// ── Vis helper: reconstruct WMRobot forward trajectories from flat Ui ──
static void vis_reconstruct_forward(
    const std::vector<double> &Ui_flat, // [N * dim_u * T], layout: i*(du*T)+d*T+t
    const Eigen::VectorXd &x0, int N, int dim_u, int dim_x, int T, double dt,
    const std::vector<int> &indices,
    std::vector<Eigen::MatrixXd> &out,
    std::vector<double> &out_costs,
    const std::vector<double> &all_costs,
    const std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> &f_func) {
  out.clear();
  out_costs.clear();
  for (int idx : indices) {
    Eigen::MatrixXd Xi(dim_x, T + 1);
    Xi.col(0) = x0;
    for (int t = 0; t < T; ++t) {
      Eigen::VectorXd u(dim_u);
      for (int d = 0; d < dim_u; ++d) {
        u(d) = Ui_flat[idx * (dim_u * T) + d * T + t];
      }
      Xi.col(t + 1) = Xi.col(t) + dt * f_func(Xi.col(t), u);
    }
    out.push_back(Xi);
    out_costs.push_back(all_costs[idx]);
  }
}

// Select stratified sample indices sorted by cost
static std::vector<int> vis_select_indices(const std::vector<double> &costs,
                                           int N, int max_save) {
  std::vector<int> sorted_idx(N);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(),
            [&](int a, int b) { return costs[a] < costs[b]; });
  std::vector<int> selected;
  int stride = std::max(1, N / max_save);
  for (int i = 0; i < N && (int)selected.size() < max_save; i += stride)
    selected.push_back(sorted_idx[i]);
  return selected;
}


// (커널들은 cuda_utils.cuh에서 shared_forward_rollout_kernel, shared_weighted_sum_kernel로 정의됨)
DEFINE_FORWARD_ROLLOUT_KERNEL(mppi_rollout_kernel)
DEFINE_WEIGHTED_SUM_KERNEL(mppi_weighted_sum_kernel)


MPPI_GPU::~MPPI_GPU() {
  freeGPU();
  curandDestroyGenerator(curand_gen);
}

void MPPI_GPU::init(MPPIParam param) {
  dt       = param.dt;
  T        = param.T;
  N        = param.N;
  gamma_u  = param.gamma_u;
  x_init   = param.x_init;
  x_target = param.x_target;

  sigma_diag.resize(dim_u);
  for (int d = 0; d < dim_u; ++d)
    sigma_diag[d] = param.sigma_u(d, d);

  u0 = Eigen::VectorXd::Zero(dim_u);
  Xo = Eigen::MatrixXd::Zero(dim_x, T + 1);

  allocGPU();
}

void MPPI_GPU::setCollisionChecker(CollisionChecker *cc) {
  collision_checker = cc;
  uploadCollisionData();
}

// ---- Memory management ----
void MPPI_GPU::allocGPU() {
  size_t sz_U  = (size_t)dim_u * T * sizeof(double);
  size_t sz_Ui = (size_t)N * dim_u * T * sizeof(double);
  size_t sz_N  = (size_t)N * sizeof(double);
  size_t sz_Di = (size_t)N * dim_u * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_U0,       sz_U));
  CUDA_CHECK(cudaMalloc(&d_Ui,       sz_Ui));
  CUDA_CHECK(cudaMalloc(&d_noise,    sz_Ui));
  CUDA_CHECK(cudaMalloc(&d_costs,    sz_N));
  CUDA_CHECK(cudaMalloc(&d_Uo,       sz_U));
  CUDA_CHECK(cudaMalloc(&d_Di,       sz_Di));
  CUDA_CHECK(cudaMalloc(&d_x_init,   dim_x * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_x_target, dim_x * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sigma,    dim_u * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_sigma, sigma_diag.data(),
                        dim_u * sizeof(double), cudaMemcpyHostToDevice));
}

void MPPI_GPU::freeGPU() {
  auto safe_free = [](double *&p) {
    if (p) { cudaFree(p); p = nullptr; }
  };
  safe_free(d_U0);   safe_free(d_Ui);  safe_free(d_noise);
  safe_free(d_costs);safe_free(d_Uo);  safe_free(d_Di);
  safe_free(d_x_init);safe_free(d_x_target);safe_free(d_sigma);
  safe_free(d_map);  safe_free(d_circles); safe_free(d_rects);
}

void MPPI_GPU::uploadCollisionData() {
  with_map       = collision_checker->with_map;
  map_resolution = collision_checker->resolution;
  map_max_row    = (int)collision_checker->map.size();
  map_max_col    = map_max_row > 0 ? (int)collision_checker->map[0].size() : 0;

  if (with_map && map_max_row > 0) {
    size_t map_sz = (size_t)map_max_row * map_max_col * sizeof(double);
    CUDA_CHECK(cudaFree(d_map));
    CUDA_CHECK(cudaMalloc(&d_map, map_sz));
    std::vector<double> flat(map_max_row * map_max_col);
    for (int r = 0; r < map_max_row; ++r)
      for (int c = 0; c < map_max_col; ++c)
        flat[r * map_max_col + c] = collision_checker->map[r][c];
    CUDA_CHECK(cudaMemcpy(d_map, flat.data(), map_sz, cudaMemcpyHostToDevice));
  }

  // Circles
  n_circles = (int)collision_checker->circles.size();
  if (n_circles > 0) {
    size_t sz = n_circles * 4 * sizeof(double);
    CUDA_CHECK(cudaFree(d_circles));
    CUDA_CHECK(cudaMalloc(&d_circles, sz));
    std::vector<double> cbuf(n_circles * 4);
    for (int i = 0; i < n_circles; ++i)
      for (int j = 0; j < 4; ++j)
        cbuf[i * 4 + j] = collision_checker->circles[i][j];
    CUDA_CHECK(cudaMemcpy(d_circles, cbuf.data(), sz, cudaMemcpyHostToDevice));
  }

  // Rectangles
  n_rects = (int)collision_checker->rectangles.size();
  if (n_rects > 0) {
    size_t sz = n_rects * 4 * sizeof(double);
    CUDA_CHECK(cudaFree(d_rects));
    CUDA_CHECK(cudaMalloc(&d_rects, sz));
    std::vector<double> rbuf(n_rects * 4);
    for (int i = 0; i < n_rects; ++i)
      for (int j = 0; j < 4; ++j)
        rbuf[i * 4 + j] = collision_checker->rectangles[i][j];
    CUDA_CHECK(cudaMemcpy(d_rects, rbuf.data(), sz, cudaMemcpyHostToDevice));
  }
}

void MPPI_GPU::uploadControl() {
  // U_0 (Eigen, col-major) → device row-major [d*T+t]
  std::vector<double> u0_flat(dim_u * T);
  for (int d = 0; d < dim_u; ++d)
    for (int t = 0; t < T; ++t)
      u0_flat[d * T + t] = U_0(d, t);
  CUDA_CHECK(cudaMemcpy(d_U0, u0_flat.data(),
                        dim_u * T * sizeof(double), cudaMemcpyHostToDevice));
}

void MPPI_GPU::uploadState() {
  CUDA_CHECK(cudaMemcpy(d_x_init, x_init.data(),
                        dim_x * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x_target, x_target.data(),
                        dim_x * sizeof(double), cudaMemcpyHostToDevice));
}

void MPPI_GPU::generateNoise() {
  size_t count = (size_t)N * dim_u * T;
  // curand requires even count
  if (count % 2 != 0) count++;
  CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_noise, count, 0.0, 1.0));
}

// ---- Main solve ----
void MPPI_GPU::solve() {
  start = std::chrono::high_resolution_clock::now();

  uploadControl();
  uploadState();
  generateNoise();

  // Launch rollout
  const int BLOCK = 256;
  int grid = (N + BLOCK - 1) / BLOCK;
  mppi_rollout_kernel<<<grid, BLOCK>>>(
      d_U0, d_Ui, d_noise, d_sigma,
      d_x_init, d_x_target,
      d_costs, d_Di,
      with_map, d_map, map_max_row, map_max_col, map_resolution,
      d_circles, n_circles, d_rects, n_rects,
      N, dim_u, dim_x, T, (double)dt, gamma_u, model_type);
  CUDA_CHECK(cudaGetLastError());

  // Copy costs to CPU for min
  std::vector<double> h_costs(N);
  CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs,
                        N * sizeof(double), cudaMemcpyDeviceToHost));

  double min_cost = *std::min_element(h_costs.begin(), h_costs.end());

  // Weighted sum kernel: one block per (d,t), threads reduce over i
  int dt_blocks = dim_u * T;
  int wsum_threads = std::min(256, N);
  size_t shared_sz = 2 * wsum_threads * sizeof(double);
  mppi_weighted_sum_kernel<<<dt_blocks, wsum_threads, shared_sz>>>(
      d_Ui, d_costs, min_cost, gamma_u, d_Uo, N, dim_u, T);
  CUDA_CHECK(cudaGetLastError());

  // Copy Uo back to host
  std::vector<double> h_Uo(dim_u * T);
  CUDA_CHECK(cudaMemcpy(h_Uo.data(), d_Uo,
                        dim_u * T * sizeof(double), cudaMemcpyDeviceToHost));
  // Clamp on CPU (small operation)
  Uo.resize(dim_u, T);
  for (int d = 0; d < dim_u; ++d)
    for (int t = 0; t < T; ++t)
      Uo(d, t) = h_Uo[d * T + t];
  // h() clamping
  h(Uo);

  CUDA_CHECK(cudaDeviceSynchronize());
  finish = std::chrono::high_resolution_clock::now();
  elapsed_1 = finish - start;
  elapsed_rollout    = elapsed_1.count();
  elapsed_clustering = 0.0;
  elapsed_connection = 0.0;
  elapsed_guide      = 0.0;
  elapsed            = elapsed_rollout;

  u0 = Uo.col(0);

  // Reconstruct optimal trajectory on CPU (dim_x small)
  Xo.col(0) = x_init;
  for (int j = 0; j < T; ++j) {
    Xo.col(j + 1) = Xo.col(j) + (double)dt * f(Xo.col(j), Uo.col(j));
  }

  // ── Visualization data export ──
  if (vis_logger && vis_logger->enabled) {
    // Copy Ui from GPU
    std::vector<double> h_Ui((size_t)N * dim_u * T);
    CUDA_CHECK(cudaMemcpy(h_Ui.data(), d_Ui,
                          (size_t)N * dim_u * T * sizeof(double),
                          cudaMemcpyDeviceToHost));
    auto sel = vis_select_indices(h_costs, N, 50);
    std::vector<Eigen::MatrixXd> sample_trajs;
    std::vector<double> sample_costs;
    vis_reconstruct_forward(h_Ui, x_init, N, dim_u, dim_x, T, dt,
                            sel, sample_trajs, sample_costs, h_costs, f);
    vis_logger->saveTrajectories("rollouts", sample_trajs);
    vis_logger->saveCosts("rollouts", sample_costs);
    vis_logger->saveTrajectory("optimal", Xo);
    vis_logger->savePosition(x_init);
  }

  visual_traj.push_back(x_init);
}

void MPPI_GPU::move() {
  // Euler step: x_init += dt * f(x_init, u0)
  x_init = x_init + (double)dt * f(x_init, u0);

  U_0.leftCols(T - 1) = Uo.rightCols(T - 1);
}
