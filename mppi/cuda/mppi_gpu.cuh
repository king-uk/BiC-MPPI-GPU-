#pragma once

#include "cuda_utils.cuh"
#include "mppi_vis_logger.h"
#include "collision_checker.h"
#include "model_base.h"
#include "mppi_param.h"

#include <Eigen/Dense>
#include <chrono>
#include <ctime>
#include <iostream>
#include <vector>

// ============================================================
// MPPI_GPU — MPPI solver with CUDA GPU rollout
//
// Interface mirrors the CPU MPPI class. Replace:
//   #include <mppi.h>       →  #include <mppi_gpu.cuh>
//   using Solver = MPPI;    →  using Solver = MPPI_GPU;
// ============================================================
class MPPI_GPU {
public:
  template <typename ModelClass> MPPI_GPU(ModelClass model);
  ~MPPI_GPU();

  void init(MPPIParam param);
  void setCollisionChecker(CollisionChecker *cc);
  virtual void solve();
  void move();
  void setVisLogger(MPPIVisLogger *logger) { vis_logger = logger; }

  // ---- Public state (same names as CPU version) ----
  Eigen::MatrixXd U_0;    // dim_u x T  warm-start control
  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::MatrixXd Uo;     // dim_u x T  optimal control
  Eigen::MatrixXd Xo;     // dim_x x (T+1)  optimal trajectory

  // ---- Timing (same fields as CPU version) ----
  std::chrono::time_point<std::chrono::high_resolution_clock> start, finish;
  std::chrono::duration<double> elapsed_1;
  double elapsed, elapsed_rollout, elapsed_clustering,
         elapsed_connection, elapsed_guide;

protected:
  int dim_x, dim_u;
  float dt;
  int T, N;
  double gamma_u;
  // Diagonal elements of sigma_u  (dim_u)
  std::vector<double> sigma_diag;

  CollisionChecker *collision_checker;
  Eigen::VectorXd u0;
  std::vector<Eigen::VectorXd> visual_traj;
  MPPIVisLogger *vis_logger = nullptr;

  // ---- GPU device buffers ----
  double *d_U0;      // dim_u x T
  double *d_Ui;      // N x dim_u x T   (perturbed controls, in-place)
  double *d_noise;   // N x dim_u x T   (raw normal noise)
  double *d_costs;   // N
  double *d_Uo;      // dim_u x T
  double *d_Di;      // N x dim_u       (mean deviation per sample)
  double *d_x_init;  // dim_x
  double *d_x_target;// dim_x
  double *d_sigma;   // dim_u

  // Collision data on GPU
  double *d_map;
  int    map_max_row, map_max_col;
  double map_resolution;
  bool   with_map;
  double *d_circles;     // n_circles x 4
  int    n_circles;
  double *d_rects;       // n_rects x 4
  int    n_rects;

  curandGenerator_t curand_gen;

  // ---- Model-independent callbacks & type ----
  int model_type;
  std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
  std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

  // ---- Helper methods ----
  void allocGPU();
  void freeGPU();
  void uploadCollisionData();
  void uploadControl();          // U_0 → d_U0
  void uploadState();            // x_init, x_target → device
  void generateNoise();          // fill d_noise with N(0,1)
  void launchRollout();          // kernel: rollout + cost per sample
  void weightedControlSum(Eigen::MatrixXd &Uo_out); // reduction → Uo
};

// ---- Template constructor ----
template <typename ModelClass>
MPPI_GPU::MPPI_GPU(ModelClass model) {
  dim_x = model.dim_x;
  dim_u = model.dim_u;
  this->f = model.f;
  this->q = model.q;
  this->p = model.p;
  this->h = model.h;

  if (dim_x == 6 && dim_u == 3) {
    model_type = 1; // Quadrotor
  } else if (dim_x == 4 && dim_u == 2) {
    model_type = 2; // Velo (2D double integrator)
  } else if (dim_x == 6 && dim_u == 6) {
    model_type = 3; // Manipulator (6-DOF velocity control)
  } else {
    model_type = 0; // WMRobot
  }

  // GPU buffers initialised in allocGPU() after init()
  d_U0 = d_Ui = d_noise = d_costs = d_Uo = d_Di = nullptr;
  d_x_init = d_x_target = d_sigma = nullptr;
  d_map = d_circles = d_rects = nullptr;
  n_circles = n_rects = 0;
  with_map = false;

  CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen, static_cast<unsigned long long>(std::time(nullptr))));
}
