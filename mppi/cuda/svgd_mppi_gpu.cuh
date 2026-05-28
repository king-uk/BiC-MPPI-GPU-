#pragma once

#include "cuda_utils.cuh"
#include "mppi_vis_logger.h"
#include "collision_checker.h"
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
// SVGDMPPI_GPU
//
// SVGD-MPPI GPU 가속 버전.
// 아키텍처:
//   - Forward / Backward rollout의 비용 평가 → GPU 커널
//   - SVGD surrogate gradient step → GPU 커널 (particle별 병렬)
//   - DBSCAN 클러스터링 / selectConnection / guideMPPI → CPU Eigen
// ============================================================
class SVGDMPPI_GPU {
public:
  template <typename ModelClass> SVGDMPPI_GPU(ModelClass model);
  ~SVGDMPPI_GPU();

  void init(SVGDMPPIParam param);
  void setCollisionChecker(CollisionChecker *cc);
  void solve();
  void move();
  void setVisLogger(MPPIVisLogger *logger) { vis_logger = logger; }

  // ── 공개 상태 (CPU BiMPPI와 동일 인터페이스) ──
  Eigen::MatrixXd U_f0;  // dim_u x Tf
  Eigen::MatrixXd U_b0;  // dim_u x Tb
  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::VectorXd dummy_u;
  Eigen::MatrixXd Uo;
  Eigen::MatrixXd Xo;
  Eigen::VectorXd u0;

  // Timing
  std::chrono::time_point<std::chrono::high_resolution_clock> start, finish;
  std::chrono::duration<double> elapsed_1, elapsed_2, elapsed_3, elapsed_4;
  double elapsed, elapsed_rollout, elapsed_clustering,
         elapsed_connection, elapsed_guide;
  std::vector<Eigen::VectorXd> visual_traj;

protected:
  int dim_x, dim_u;
  float dt;
  int Tf, Tb, Nf, Nb, Nr;
  int Ns;    // surrogate samples
  int istep; // SVGD inner iterations

  double gamma_u;
  std::vector<double> sigma_diag; // diagonal of sigma_u

  double deviation_mu, epsilon, psi, cost_mu;
  int minpts;

  CollisionChecker *collision_checker{nullptr};
  MPPIVisLogger *vis_logger = nullptr;

  // CPU-side cluster / traj data
  std::vector<std::vector<int>> clusters_f, clusters_b;
  std::vector<int> full_cluster_f, full_cluster_b;
  Eigen::MatrixXd Uf, Ub, Xf, Xb;
  std::vector<std::vector<int>> joints;
  std::vector<Eigen::MatrixXd> Xc, Uc;
  std::vector<Eigen::MatrixXd> Ur, Xr;
  std::vector<double> Cr;

  // GPU buffers – forward
  double *d_Uf0, *d_Ufi, *d_noise_f, *d_costs_f, *d_Di_f;
  double *d_noise_samples_f, *d_sample_costs_f, *d_cov_acc_f;
  // GPU buffers – backward
  double *d_Ub0, *d_Ubi, *d_noise_b, *d_costs_b, *d_Di_b;
  double *d_noise_samples_b, *d_sample_costs_b, *d_cov_acc_b;
  // GPU buffers – guide
  double *d_Ur0, *d_Uri, *d_noise_r, *d_costs_r, *d_Xref;
  // shared
  double *d_x_init, *d_x_target, *d_sigma;
  // collision
  double *d_map, *d_circles, *d_rects;
  int map_max_row, map_max_col, n_circles, n_rects;
  double map_resolution;
  bool with_map;

  curandGenerator_t curand_gen;
  int alloc_Nf, alloc_Nb, alloc_Tf, alloc_Tb;

  // GPU 메모리 관리
  void allocForward();
  void allocBackward();
  void allocGuide();
  void freeForward();
  void freeBackward();
  void freeGuide();
  void freeCommon();
  void uploadCollisionData();

  // 핵심 단계
  void forwardRollout();
  void backwardRollout();
  void selectConnection();
  void concatenate();
  void guideMPPI();
  void partitioningControl();

  // CPU 클러스터링 (DBSCAN)
  void dbscan(std::vector<std::vector<int>> &clusters,
              const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs,
              int N_samples);
  void calculateU(Eigen::MatrixXd &Uout,
                  const std::vector<std::vector<int>> &clusters,
                  const Eigen::VectorXd &costs,
                  const Eigen::MatrixXd &Ui_cpu,
                  int T_steps);
};

// ── 템플릿 생성자 ─────────────────────────────────────────────
template <typename ModelClass>
SVGDMPPI_GPU::SVGDMPPI_GPU(ModelClass model) {
  dim_x = model.dim_x;
  dim_u = model.dim_u;
  d_Uf0 = d_Ufi = d_noise_f = d_costs_f = d_Di_f = nullptr;
  d_Ub0 = d_Ubi = d_noise_b = d_costs_b = d_Di_b = nullptr;
  d_x_init = d_x_target = d_sigma = nullptr;
  d_map = d_circles = d_rects = nullptr;
  n_circles = n_rects = 0;
  with_map = false;
  alloc_Nf = alloc_Nb = alloc_Tf = alloc_Tb = 0;

  CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen, static_cast<unsigned long long>(std::time(nullptr))));
}
