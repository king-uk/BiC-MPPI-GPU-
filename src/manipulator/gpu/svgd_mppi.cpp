#include <svgd_mppi_gpu.cuh>
#include <manipulator.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

int main() {
  std::cout << "Starting Manipulator GPU SVGD-MPPI..." << std::endl;

  // GPU collision checker는 2D 전용 (manipulator 장애물 회피는 stage cost FK 기반으로 처리)
  CollisionChecker collision_checker = CollisionChecker();
  auto model = Manipulator();

  using Solver = SVGDMPPI_GPU;
  using SolverParam = SVGDMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = 40;
  param.Tb = 40;
  param.x_init.resize(model.dim_x);
  param.x_init.setZero();
  param.x_target.resize(model.dim_x);
  param.x_target.setZero();
  param.x_target(0) = M_PI / 2.0;

  param.Nf = 200;
  param.Nb = 200;
  param.Ns = 10;
  param.istep = 5;
  param.Nr = 2000;
  param.gamma_u = 0.1;

  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.5;
  param.psi = 0.3;

  int maxiter = 500;
  double tolerance = 0.15;

  Solver solver(model);
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
  solver.init(param);
  solver.setCollisionChecker(&collision_checker);

  std::ofstream csv_file("result_manipulator_gpu_svgd_mppi.csv");
  csv_file << "Iter";
  for (int j = 0; j < model.dim_x; ++j) csv_file << ",q_" << j;
  csv_file << ",err,elapsed\n";

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
    for (int j = 0; j < model.dim_x; ++j) csv_file << "," << solver.x_init(j);
    csv_file << "," << f_err << "," << solver.elapsed << "\n";

    if (f_err < tolerance) { is_reached = true; break; }

    if (i % 50 == 0)
      std::cout << "Iter: " << i << " | q0: " << solver.x_init(0)
                << " | Err: " << f_err << " | Elapsed: " << total_elapsed << "\n";
  }

  csv_file.close();
  std::cout << "====================================\n"
            << "Success: " << is_reached << "\t Iters: " << i
            << "\t Total Time: " << total_elapsed
            << "\t Final Err: " << f_err << "\n";
  return 0;
}
