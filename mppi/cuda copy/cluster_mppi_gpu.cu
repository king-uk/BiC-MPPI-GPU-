#include "cluster_mppi_gpu.cuh"

#include <algorithm>
#include <limits>
#include <numeric>

// Local rollout kernel (ODR-safe: unique name per TU)
DEFINE_FORWARD_ROLLOUT_KERNEL(cluster_rollout_kernel)


ClusterMPPI_GPU::~ClusterMPPI_GPU() {}

// ---- DBSCAN (CPU, identical logic to CPU ClusterMPPI) ----
void ClusterMPPI_GPU::dbscan(std::vector<std::vector<int>> &clusters,
                              const Eigen::MatrixXd &Di,
                              const Eigen::VectorXd &costs,
                              int N_samples, int T_steps) {
  clusters.clear();
  std::vector<bool> core_pts(N_samples, false);
  std::map<int, std::vector<int>> core_tree;

#pragma omp parallel for
  for (int i = 0; i < N_samples; ++i) {
    if (costs(i) > 1e7) continue;
    for (int j = i + 1; j < N_samples; ++j) {
      if (costs(j) > 1e7) continue;
      if (deviation_mu * (Di.col(i) - Di.col(j)).norm() < epsilon) {
#pragma omp critical
        {
          core_tree[i].push_back(j);
          core_tree[j].push_back(i);
        }
      }
    }
  }

  for (int i = 0; i < N_samples; ++i)
    if ((int)core_tree[i].size() > minpts) core_pts[i] = true;

  std::vector<bool> visited(N_samples, false);
  for (int i = 0; i < N_samples; ++i) {
    if (!core_pts[i] || visited[i]) continue;
    std::deque<int> branch;
    std::vector<int> cluster;
    branch.push_back(i);
    cluster.push_back(i);
    visited[i] = true;
    while (!branch.empty()) {
      int now = branch.front(); branch.pop_front();
      for (int nb : core_tree[now]) {
        if (visited[nb]) continue;
        visited[nb] = true;
        cluster.push_back(nb);
        if (core_pts[nb]) branch.push_back(nb);
      }
    }
    clusters.push_back(cluster);
  }
}

void ClusterMPPI_GPU::calculateU(Eigen::MatrixXd &Uout,
                                  const std::vector<std::vector<int>> &clusters,
                                  const Eigen::VectorXd &costs,
                                  const Eigen::MatrixXd &Ui_cpu,
                                  int T_steps) {
  int n_cl = (int)clusters.size();
  Uout = Eigen::MatrixXd::Zero(n_cl * dim_u, T_steps);

#pragma omp parallel for
  for (int idx = 0; idx < n_cl; ++idx) {
    int pts = (int)clusters[idx].size();
    double min_c = std::numeric_limits<double>::max();
    for (int k : clusters[idx]) min_c = std::min(min_c, costs(k));

    std::vector<double> wts(pts);
    double tw = 0.0;
    for (int i = 0; i < pts; ++i) {
      wts[i] = std::exp(-gamma_u * (costs(clusters[idx][i]) - min_c));
      tw += wts[i];
    }
    for (int i = 0; i < pts; ++i) {
      int k = clusters[idx][i];
      Uout.middleRows(idx * dim_u, dim_u) +=
          (wts[i] / tw) * Ui_cpu.middleRows(k * dim_u, dim_u);
    }
    // Clamp
    Uout.block(idx * dim_u, 0, 1, T_steps) =
        Uout.block(idx * dim_u, 0, 1, T_steps).cwiseMax(0.0).cwiseMin(1.0);
    Uout.block(idx * dim_u + 1, 0, 1, T_steps) =
        Uout.block(idx * dim_u + 1, 0, 1, T_steps)
            .cwiseMax(-M_PI / 2.0)
            .cwiseMin(M_PI / 2.0);
  }
}

// ---- Override solve() ----
void ClusterMPPI_GPU::solve() {
  std::vector<int> full_cluster(N);
  std::iota(full_cluster.begin(), full_cluster.end(), 0);

  start = std::chrono::high_resolution_clock::now();

  // --- GPU rollout ---
  uploadControl();
  uploadState();
  generateNoise();

  const int BLOCK = 256;
  int grid = (N + BLOCK - 1) / BLOCK;
  cluster_rollout_kernel<<<grid, BLOCK>>>(
      d_U0, d_Ui, d_noise, d_sigma,
      d_x_init, d_x_target,
      d_costs, d_Di,
      with_map, d_map, map_max_row, map_max_col, map_resolution,
      d_circles, n_circles, d_rects, n_rects,
      N, dim_u, dim_x, T, (double)dt, gamma_u);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t_cluster_start = std::chrono::high_resolution_clock::now();
  elapsed_rollout = std::chrono::duration<double>(t_cluster_start - start).count();

  // Copy costs + Ui + Di to host for clustering
  std::vector<double> h_costs(N);
  CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Ui host copy: N x dim_u x T → Eigen (N*dim_u) x T
  std::vector<double> h_Ui_flat((size_t)N * dim_u * T);
  CUDA_CHECK(cudaMemcpy(h_Ui_flat.data(), d_Ui,
                        (size_t)N * dim_u * T * sizeof(double), cudaMemcpyDeviceToHost));
  Eigen::MatrixXd Ui_cpu(N * dim_u, T);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < dim_u; ++d)
      for (int t = 0; t < T; ++t)
        Ui_cpu(i * dim_u + d, t) = h_Ui_flat[i * (dim_u * T) + d * T + t];

  // Di host copy
  std::vector<double> h_Di_flat((size_t)N * dim_u);
  CUDA_CHECK(cudaMemcpy(h_Di_flat.data(), d_Di,
                        (size_t)N * dim_u * sizeof(double), cudaMemcpyDeviceToHost));
  Eigen::MatrixXd Di_cpu(dim_u, N);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < dim_u; ++d)
      Di_cpu(d, i) = h_Di_flat[i * dim_u + d];

  Eigen::VectorXd costs_cpu = Eigen::Map<Eigen::VectorXd>(h_costs.data(), N);

  // --- DBSCAN clustering (CPU) ---
  bool all_feasible = (costs_cpu.array() < 1e7).all();
  std::vector<std::vector<int>> clusters;
  if (!all_feasible) dbscan(clusters, Di_cpu, costs_cpu, N, T);
  if (clusters.empty()) clusters.push_back(full_cluster);
  calculateU(U, clusters, costs_cpu, Ui_cpu, T);

  // Select best cluster
  double min_cost = std::numeric_limits<double>::max();
  int min_idx = 0;
  for (int ci = 0; ci < (int)clusters.size(); ++ci) {
    Eigen::MatrixXd Xi(dim_x, T + 1);
    Xi.col(0) = x_init;
    double cost = 0.0;
    for (int j = 0; j < T; ++j) {
      Eigen::VectorXd xd(dim_x);
      double v = U(ci * dim_u, j), omega = U(ci * dim_u + 1, j);
      xd(0) = v * cos(Xi(2, j));
      xd(1) = v * sin(Xi(2, j));
      xd(2) = omega;
      Xi.col(j + 1) = Xi.col(j) + (double)dt * xd;
      cost += (Xi.col(j) - x_target).norm();
    }
    cost += (Xi.col(T) - x_target).norm();
    if (collision_checker->getCollisionGrid(Xi.col(T))) cost = 1e8;
    if (cost < min_cost) { min_cost = cost; min_idx = ci; }
  }

  Uo = U.middleRows(min_idx * dim_u, dim_u);

  auto t_end = std::chrono::high_resolution_clock::now();
  finish = t_end;
  elapsed_clustering = std::chrono::duration<double>(t_end - t_cluster_start).count();
  elapsed_connection = 0.0;
  elapsed_guide      = 0.0;
  elapsed_1 = finish - start;
  elapsed   = elapsed_rollout + elapsed_clustering;

  u0 = Uo.col(0);

  Xo.col(0) = x_init;
  for (int j = 0; j < T; ++j) {
    Eigen::VectorXd xd(dim_x);
    double v = Uo(0, j), omega = Uo(1, j);
    xd(0) = v * cos(Xo(2, j));
    xd(1) = v * sin(Xo(2, j));
    xd(2) = omega;
    Xo.col(j + 1) = Xo.col(j) + (double)dt * xd;
  }

  visual_traj.push_back(x_init);
}
