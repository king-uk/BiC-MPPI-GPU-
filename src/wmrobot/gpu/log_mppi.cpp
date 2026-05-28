#include <log_mppi_gpu.cuh>
#include <wmrobot_map.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = WMRobotMap();

  using Solver = LogMPPI_GPU;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.T = 100;
  param.x_init.resize(model.dim_x);
  param.x_init << 2.5, 0.0, M_PI_2;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, M_PI_2;
  param.N = 6000;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.6, 0.6;
  param.sigma_u = sigma_u.asDiagonal();

  int maxiter = 200;

  std::ofstream csv("result_log_mppi.csv");
  csv << "s,map,is_success,iter,elapsed,elapsed_rollout,elapsed_clustering,"
         "elapsed_connection,elapsed_guide,f_err\n";

  for (int s = 0; s < 2; ++s) {
    switch (s) {
    case 0:
      param.x_init(0) = 0.5;
      break;
    case 1:
      param.x_init(0) = 2.5;
      break;
    case 2:
      param.x_init(0) = 1.5;
      break;
    default:
      break;
    }
    for (int map = 299; map >= 0; --map) {
      CollisionChecker collision_checker = CollisionChecker();
      collision_checker.loadMap("../BARN_dataset/txt_files/output_" +
                                    std::to_string(map) + ".txt",
                                0.1);
      Solver solver(model);
      solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
      solver.init(param);
      solver.setCollisionChecker(&collision_checker);

      bool is_success = false;
      bool is_collision = false;
      int i = 0;
      double total_elapsed = 0.0;
      double total_rollout = 0.0;
      double total_clustering = 0.0;
      double total_connection = 0.0;
      double total_guide = 0.0;
      double f_err = 0.0;
      for (i = 0; i < maxiter; ++i) {
        solver.solve();
        solver.move();

        total_elapsed += solver.elapsed;
        total_rollout += solver.elapsed_rollout;
        total_clustering += solver.elapsed_clustering;
        total_connection += solver.elapsed_connection;
        total_guide += solver.elapsed_guide;

        if (collision_checker.getCollisionGrid(solver.x_init)) {
          is_collision = true;
          break;
        } else {
          f_err = (solver.x_init - param.x_target).norm();
          if (f_err < 0.1) {
            is_success = true;
            break;
          }
        }
      }
      std::cout << s << '\t' << map << '\t' << is_success << '\t' << i << '\t'
                << total_elapsed << std::endl;
      csv << s << ',' << map << ',' << is_success << ',' << i << ','
          << total_elapsed << ',' << total_rollout << ',' << total_clustering
          << ',' << total_connection << ',' << total_guide << ',' << f_err
          << '\n';
    }
  }

  csv.close();
  return 0;
}
