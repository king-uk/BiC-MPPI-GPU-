#include "manipulator.h"
#include "svgd_mppi_3D.h"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

int main() {
  std::cout << "Starting Manipulator SVGD-MPPI (Velocity Control)..."
            << std::endl;

  // ==========================================================================
  // 환경(Environment) 설정
  // ==========================================================================
  CollisionChecker collision_checker = CollisionChecker();
  collision_checker.addRectangle(-0.60, -0.60, -1.00, 0.20, 0.20, 1.5);

  auto model = Manipulator();
  for (const auto &rect : collision_checker.rectangles) {
    model.addObstacle(rect[0], rect[2], rect[4], rect[1] - rect[0],
                      rect[3] - rect[2], rect[5] - rect[4]);
  }

  using Solver = SVGDMPPI3D;
  using SolverParam = SVGDMPPIParam;

  SolverParam param;

  // [시간 스텝]
  param.dt = 0.1;

  // [Horizon]
  param.Tf = 40;
  param.Tb = 40;

  // [초기/목표 상태]
  param.x_init.resize(model.dim_x);
  param.x_init.setZero();
  param.x_target.resize(model.dim_x);
  param.x_target.setZero();
  param.x_target(0) = M_PI / 2.0;

  // [샘플 수] SVGD는 particle 수가 적어도 surrogate gradient로 다양성 확보
  param.Nf = 60;
  param.Nb = 60;
  param.Nr = 2000;
  param.Ns = 10;   // surrogate samples per SVGD step
  param.istep = 5; // SVGD inner iterations

  // [gamma_u]
  param.gamma_u = 0.1;

  // [sigma_u]
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;
  param.sigma_u = sigma_u.asDiagonal();

  // [DBSCAN]
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 3;
  param.epsilon = 0.5;
  param.psi = 0.6;

  // ==========================================================================
  // Solver 초기화
  // ==========================================================================
  int maxiter = 500;
  double tolerance = 0.15;

  Solver solver(model);
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);

  std::cout << "Obstacle: Set dynamically from CollisionChecker\n";

  solver.init(param);
  solver.setCollisionChecker(&collision_checker);

  // ==========================================================================
  // CSV 로깅
  // ==========================================================================
  std::ofstream csv_file("x_init_svgd_mppi_log.csv");
  if (!csv_file.is_open()) {
    std::cerr << "Error: Could not open x_init_svgd_mppi_log.csv" << std::endl;
    return -1;
  }

  csv_file << "# START";
  for (int j = 0; j < model.dim_x; ++j)
    csv_file << "," << param.x_init(j);
  csv_file << "\n";

  csv_file << "# TARGET";
  for (int j = 0; j < model.dim_x; ++j)
    csv_file << "," << param.x_target(j);
  csv_file << "\n";

  for (const auto &rect : collision_checker.rectangles) {
    csv_file << "# OBS," << rect[0] << "," << rect[1] << "," << rect[2] << ","
             << rect[3] << "," << rect[4] << "," << rect[5] << "\n";
  }

  csv_file << "Iter";
  for (int j = 0; j < model.dim_x; ++j)
    csv_file << ",q_" << j;
  csv_file << ",err\n";

  // ==========================================================================
  // Main Loop
  // ==========================================================================
  bool is_reached = false;
  double total_elapsed = 0.0;
  double f_err = 0.0;
  int i = 0;

  for (i = 0; i < maxiter; ++i) {
    solver.solve();
    solver.move();
    total_elapsed += solver.elapsed;

    f_err = (solver.x_init - param.x_target).norm();

    csv_file << i;
    for (int j = 0; j < model.dim_x; ++j)
      csv_file << "," << solver.x_init(j);
    csv_file << "," << f_err << "\n";

    if (f_err < tolerance) {
      is_reached = true;
      break;
    }

    if (i % 10 == 0) {
      std::cout << "Iter: " << i << " | J1: " << solver.x_init(0)
                << " J2: " << solver.x_init(1) << " J3: " << solver.x_init(2)
                << " | Err: " << f_err << " | Elapsed: " << total_elapsed
                << "\n";
    }
  }

  csv_file.close();

  std::cout << "====================================\n";
  std::cout << "Success: " << is_reached << "\t Iters: " << i
            << "\t Total Time: " << total_elapsed << " seconds"
            << "\t Final Err: " << f_err << "\n";

  return 0;
}
