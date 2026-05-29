#include <cluster_mppi_gpu.cuh>
#include <manipulator.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

int main() {
  std::cout << "Starting Manipulator GPU Cluster-MPPI..." << std::endl;

  // GPU collision checker는 2D 전용 (manipulator 장애물 회피는 stage cost FK 기반으로 처리)
  CollisionChecker collision_checker = CollisionChecker();
  auto model = Manipulator();

  using Solver = ClusterMPPI_GPU;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.T = 80;
  param.x_init.resize(model.dim_x);
  param.x_init.setZero();
  param.x_target.resize(model.dim_x);
  param.x_target.setZero();
  param.x_target(0) = M_PI / 2.0;
  param.N = 5000;
  param.gamma_u = 0.1;

  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;
  param.sigma_u = sigma_u.asDiagonal();

  int maxiter = 500;
  double tolerance = 0.15;

  Solver solver(model);
  solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
  solver.init(param);
  solver.setCollisionChecker(&collision_checker);

  std::ofstream csv_file("result_manipulator_gpu_cluster_mppi.csv");
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
