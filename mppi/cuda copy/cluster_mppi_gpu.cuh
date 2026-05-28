#pragma once

#include "mppi_gpu.cuh"

#include <deque>
#include <map>
#include <numeric>

// ============================================================
// ClusterMPPI_GPU — GPU rollout + CPU DBSCAN clustering
//
// Interface mirrors the CPU ClusterMPPI class.
// ============================================================
class ClusterMPPI_GPU : public MPPI_GPU {
public:
  template <typename ModelClass> ClusterMPPI_GPU(ModelClass model);
  ~ClusterMPPI_GPU();

  Eigen::MatrixXd U;  // dim_u x (clusters * T)  cluster controls
  Eigen::MatrixXd X;

  void solve() override;

  void dbscan(std::vector<std::vector<int>> &clusters,
              const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs,
              int N_samples, int T_steps);

  void calculateU(Eigen::MatrixXd &Uout,
                  const std::vector<std::vector<int>> &clusters,
                  const Eigen::VectorXd &costs,
                  const Eigen::MatrixXd &Ui_cpu,
                  int T_steps);

private:
  double deviation_mu;
  double epsilon;
  int    minpts;
};

template <typename ModelClass>
ClusterMPPI_GPU::ClusterMPPI_GPU(ModelClass model) : MPPI_GPU(model) {
  deviation_mu = 1.0;
  epsilon      = 0.01;
  minpts       = 5;
}
