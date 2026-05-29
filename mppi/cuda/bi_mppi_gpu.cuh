#pragma once

#include "collision_checker.h"
#include "cuda_utils.cuh"
#include "mppi_vis_logger.h"
#include "model_base.h"
#include "mppi_param.h"

#include <Eigen/Dense>
#include <chrono>
#include <ctime>
#include <deque>
#include <map>
#include <numeric>
#include <vector>

// ============================================================
// BiMPPI_GPU — Bidirectional MPPI with GPU rollout
//
// Forward/backward rollout + guide MPPI on GPU.
// DBSCAN, selectConnection, concatenate stay on CPU.
// ============================================================
class BiMPPI_GPU {
public:
  template <typename ModelClass> BiMPPI_GPU(ModelClass model);
  ~BiMPPI_GPU();

  void init(BiMPPIParam param);
  void setCollisionChecker(CollisionChecker *cc);
  void solve();
  void move();
  void setVisLogger(MPPIVisLogger *logger) { vis_logger = logger; }

  // ---- Public state (mirrors CPU BiMPPI) ----
  Eigen::MatrixXd U_f0; // dim_u x Tf
  Eigen::MatrixXd U_b0; // dim_u x Tb
  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::VectorXd dummy_u;
  Eigen::MatrixXd Uo;
  Eigen::MatrixXd Xo;
  Eigen::VectorXd u0;

  // Timing
  std::chrono::time_point<std::chrono::high_resolution_clock> start, finish;
  std::chrono::duration<double> elapsed_1, elapsed_2, elapsed_3, elapsed_4;
  double elapsed, elapsed_rollout, elapsed_clustering, elapsed_connection,
      elapsed_guide;
  std::vector<Eigen::VectorXd> visual_traj;

protected:
  int dim_x, dim_u;
  float dt;
  int Tf, Tb, Nf, Nb, Nr;
  double gamma_u;
  std::vector<double> sigma_diag;
  double deviation_mu, epsilon;
  int minpts;
  double psi;

  CollisionChecker *collision_checker;
  MPPIVisLogger *vis_logger = nullptr;

  // CPU cluster data
  std::vector<std::vector<int>> clusters_f, clusters_b;
  std::vector<int> full_cluster_f, full_cluster_b;
  Eigen::MatrixXd Uf, Ub, Xf, Xb;
  std::vector<std::vector<int>> joints;
  std::vector<Eigen::MatrixXd> Xc, Uc;
  std::vector<Eigen::MatrixXd> Ur, Xr;
  std::vector<double> Cr;

  // GPU buffers (forward)
  double *d_Uf0, *d_Ufi, *d_noise_f, *d_costs_f, *d_Uf_out, *d_Di_f;
  // GPU buffers (backward)
  double *d_Ub0, *d_Ubi, *d_noise_b, *d_costs_b, *d_Ub_out, *d_Di_b;
  // GPU buffers (guide)
  double *d_Ur0, *d_Uri, *d_noise_r, *d_costs_r, *d_Ur_out, *d_Xref;

  double *d_x_init, *d_x_target, *d_sigma;

  // Collision
  double *d_map, *d_circles, *d_rects;
  int map_max_row, map_max_col, n_circles, n_rects;
  double map_resolution;
  bool with_map;
  curandGenerator_t curand_gen;

  int alloc_Nf, alloc_Nb, alloc_Tf, alloc_Tb; // last allocated sizes

  // ---- Model-independent callbacks & type ----
  int model_type;
  std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
  std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

  void allocForward();
  void allocBackward();
  void allocGuide();
  void freeForward();
  void freeBackward();
  void freeGuide();
  void freeCommon();
  void uploadCollisionData();

  void backwardRollout();
  void forwardRollout();
  void selectConnection();
  void concatenate();
  void guideMPPI();
  void partitioningControl();

  void dbscan(std::vector<std::vector<int>> &clusters,
              const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs,
              int N_samples);
  void calculateU(Eigen::MatrixXd &Uout,
                  const std::vector<std::vector<int>> &clusters,
                  const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui_cpu,
                  int T_steps);
};

template <typename ModelClass> BiMPPI_GPU::BiMPPI_GPU(ModelClass model) {
  dim_x = model.dim_x;
  dim_u = model.dim_u;
  this->f = model.f;
  this->q = model.q;
  this->p = model.p;
  this->h = model.h;

  if (dim_x == 6 && dim_u == 3) {
    model_type = 1; // Quadrotor
  } else if (dim_x == 4 && dim_u == 2) {
    model_type = 2; // Velo
  } else if (dim_x == 6 && dim_u == 6) {
    model_type = 3; // Manipulator (6-DOF velocity control)
  } else {
    model_type = 0; // WMRobot
  }

  d_Uf0 = d_Ufi = d_noise_f = d_costs_f = d_Uf_out = d_Di_f = nullptr;
  d_Ub0 = d_Ubi = d_noise_b = d_costs_b = d_Ub_out = d_Di_b = nullptr;
  d_Ur0 = d_Uri = d_noise_r = d_costs_r = d_Ur_out = d_Xref = nullptr;
  d_x_init = d_x_target = d_sigma = nullptr;
  d_map = d_circles = d_rects = nullptr;
  n_circles = n_rects = 0;
  with_map = false;
  alloc_Nf = alloc_Nb = alloc_Tf = alloc_Tb = 0;

  CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen, static_cast<unsigned long long>(std::time(nullptr))));
}
