#include "manipulator.h"
#include "mppi_3D.h"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

int main() {
  std::cout << "Starting Manipulator MPPI (Velocity Control)..." << std::endl;
  CollisionChecker collision_checker = CollisionChecker();
  collision_checker.addRectangle(-0.60, -0.60, -1.00, 0.20, 0.20, 1.5);

  auto model = Manipulator();
  for (const auto &rect : collision_checker.rectangles) {
    model.addObstacle(rect[0], rect[2], rect[4], rect[1] - rect[0],
                      rect[3] - rect[2], rect[5] - rect[4]);
  }

  using Solver = MPPI3D;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.T = 80;
  param.x_init.resize(model.dim_x);
  param.x_init.setZero();
  param.x_target.resize(model.dim_x);
  param.x_target.setZero();
  param.x_target(0) = M_PI / 2.0;
  param.N = 3000;
  param.gamma_u = 0.1;

  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;
  param.sigma_u = sigma_u.asDiagonal();

  int maxiter = 500;
  double tolerance = 0.15;

  Solver solver(model);
  solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);

  std::cout << "Obstacle: Set dynamically from CollisionChecker\n";

  solver.init(param);
  solver.setCollisionChecker(&collision_checker);

  std::ofstream csv_file("x_init_mppi_log.csv");
  if (!csv_file.is_open()) {
    std::cerr << "Error: Could not open log csv" << std::endl;
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
  for (int j = 0; j < model.dim_x; ++j) {
    csv_file << ",q_" << j;
  }
  csv_file << ",err\n";

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
    for (int j = 0; j < model.dim_x; ++j) {
      csv_file << "," << solver.x_init(j);
    }
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
