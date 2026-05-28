with open("mppi/cuda/svgd_mppi_gpu.cu", "r") as f:
    text = f.read()

forward_cpu_block = """
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> hc(Nf), hUi((size_t)Nf * dim_u * Tf), hDi((size_t)Nf * dim_u);
  CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_f, Nf * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hUi.data(), d_Ufi, (size_t)Nf * dim_u * Tf * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hDi.data(), d_Di_f, (size_t)Nf * dim_u * sizeof(double), cudaMemcpyDeviceToHost));
  Eigen::VectorXd costs_f = Eigen::Map<Eigen::VectorXd>(hc.data(), Nf);
  Eigen::MatrixXd Ui_f = flat_Ui_to_eigen(hUi, Nf, dim_u, Tf);
  Eigen::MatrixXd Di_f = flat_Di_to_eigen_col(hDi, Nf, dim_u);
  bool ok = (costs_f.array() < 1e7).all();
  clusters_f.clear();
  if (!ok) dbscan(clusters_f, Di_f, costs_f, Nf);
  if (clusters_f.empty()) clusters_f.push_back(full_cluster_f);
  calculateU(Uf, clusters_f, costs_f, Ui_f, Tf);
  Xf.resize(clusters_f.size() * dim_x, Tf + 1);
  for (int ci = 0; ci < (int)clusters_f.size(); ++ci) {
    Xf.block(ci * dim_x, 0, dim_x, 1) = x_init;
    for (int t = 0; t < Tf; ++t) {
      Eigen::VectorXd xd(dim_x);
      double v = Uf(ci * dim_u, t), om = Uf(ci * dim_u + 1, t);
      xd(0) = v * cos(Xf(ci * dim_x + 2, t));
      xd(1) = v * sin(Xf(ci * dim_x + 2, t));
      xd(2) = om;
      Xf.block(ci * dim_x, t + 1, dim_x, 1) = Xf.block(ci * dim_x, t, dim_x, 1) + (double)dt * xd;
    }
  }
  elapsed_clustering += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();
"""

backward_cpu_block = """
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<double> hc(Nb), hUi((size_t)Nb * dim_u * Tb), hDi((size_t)Nb * dim_u);
  CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_b, Nb * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hUi.data(), d_Ubi, (size_t)Nb * dim_u * Tb * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hDi.data(), d_Di_b, (size_t)Nb * dim_u * sizeof(double), cudaMemcpyDeviceToHost));
  Eigen::VectorXd costs_b = Eigen::Map<Eigen::VectorXd>(hc.data(), Nb);
  Eigen::MatrixXd Ui_b = flat_Ui_to_eigen(hUi, Nb, dim_u, Tb);
  Eigen::MatrixXd Di_b = flat_Di_to_eigen_col(hDi, Nb, dim_u);
  bool ok = (costs_b.array() < 1e7).all();
  clusters_b.clear();
  if (!ok) dbscan(clusters_b, Di_b, costs_b, Nb);
  if (clusters_b.empty()) clusters_b.push_back(full_cluster_b);
  calculateU(Ub, clusters_b, costs_b, Ui_b, Tb);
  Xb.resize(clusters_b.size() * dim_x, Tb + 1);
  for (int ci = 0; ci < (int)clusters_b.size(); ++ci) {
    Xb.block(ci * dim_x, Tb, dim_x, 1) = x_target;
    for (int t = Tb - 1; t >= 0; --t) {
      Eigen::VectorXd xd(dim_x);
      double v, om;
      if (t == Tb - 1) {
        v = Ub(ci * dim_u, t);
        om = Ub(ci * dim_u + 1, t);
      } else {
        v = Ub(ci * dim_u, t + 1);
        om = Ub(ci * dim_u + 1, t + 1);
      }
      xd(0) = v * cos(Xb(ci * dim_x + 2, t + 1));
      xd(1) = v * sin(Xb(ci * dim_x + 2, t + 1));
      xd(2) = om;
      Xb.block(ci * dim_x, t, dim_x, 1) = Xb.block(ci * dim_x, t + 1, dim_x, 1) - (double)dt * xd;
    }
  }
  elapsed_clustering += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();
"""

# Inject into forwardRollout
fwd_end = text.find("elapsed_rollout += std::chrono::duration<double>(")
if fwd_end != -1:
    fwd_end = text.find(";\n}", fwd_end) + 1
    text = text[:fwd_end] + forward_cpu_block + text[fwd_end:]

# Inject into backwardRollout
bwd_end = text.find("elapsed_rollout += std::chrono::duration<double>(", fwd_end)
if bwd_end != -1:
    bwd_end = text.find(";\n}", bwd_end) + 1
    text = text[:bwd_end] + backward_cpu_block + text[bwd_end:]

with open("mppi/cuda/svgd_mppi_gpu.cu", "w") as f:
    f.write(text)

print("Injected CPU clustering logic back into rollouts!")
