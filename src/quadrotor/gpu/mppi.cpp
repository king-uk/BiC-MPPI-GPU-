#include <mppi_gpu.cuh>
#include <quadrotor.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = Quadrotor();

  using Solver = MPPI_GPU;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.T = 100;
  param.x_init.resize(model.dim_x);
  param.x_init << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;
  param.N = 6000;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 1.5, 1.5, 1.5;
  param.sigma_u = sigma_u.asDiagonal();

  int maxiter = 200;

  std::ofstream csv("result_quadrotor_mppi.csv");
  csv << "map,is_failed,is_landed,iter,elapsed,elapsed_rollout,elapsed_"
         "clustering,"
         "elapsed_connection,elapsed_guide,f_err\n";

  for (int map = 299; map >= 0; --map) {
    CollisionChecker collision_checker = CollisionChecker();
    collision_checker.loadMap("../BARN_dataset/txt_files/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);
    Solver solver(model);
    solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
    solver.U_0.row(2).array() += model.g;
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);

    bool is_landed = false;
    bool is_failed = true;
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
        break;
      } else {
        f_err = (solver.x_init.head(2) - param.x_target.head(2)).norm();
        if (solver.x_init(2) < 0) {
          is_landed = true;
          is_failed = false;
          break;
        }
      }
    }
    std::cout << map << '\t' << is_failed << '\t' << is_landed << '\t' << i
              << '\t' << total_elapsed << std::endl;
    csv << map << ',' << is_failed << ',' << is_landed << ',' << i << ','
        << total_elapsed << ',' << total_rollout << ',' << total_clustering
        << ',' << total_connection << ',' << total_guide << ',' << f_err
        << '\n';
  }

  csv.close();
  return 0;
}
