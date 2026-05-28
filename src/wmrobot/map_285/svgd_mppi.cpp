#include <mppi_vis_logger.h>
#include <svgd_mppi_gpu.cuh>
#include <wmrobot_map.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = WMRobotMap();

  using Solver = SVGDMPPI_GPU;
  using SolverParam = SVGDMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = 50;
  param.Tb = 50;
  param.x_init.resize(model.dim_x);
  param.x_init << 2.5, 0.0, M_PI_2;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, M_PI_2;

  param.Nf = 700;
  param.Nb = 700;
  param.Ns = 10;
  param.istep = 5;

  param.Nr = 3000;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.6, 0.6;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;

  int maxiter = 200;

  std::ofstream csv("result_svgd_mppi.csv");
  csv << "s,map,is_success,iter,elapsed,elapsed_rollout,elapsed_clustering,"
         "elapsed_connection,elapsed_guide,f_err\n";

  param.x_init(0) = 0.5;

  int map = 285;
  CollisionChecker collision_checker = CollisionChecker();
  collision_checker.loadMap(
      "../BARN_dataset/txt_files/output_" + std::to_string(map) + ".txt", 0.1);

  // ── Visualization logger ──
  MPPIVisLogger vis_logger;
  vis_logger.enabled = true;
  vis_logger.init("svgd_mppi", model.dim_x, param.Tf,
                  collision_checker.resolution, collision_checker.map,
                  param.x_init, param.x_target);

  Solver solver(model);
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
  solver.init(param);
  solver.setCollisionChecker(&collision_checker);
  solver.setVisLogger(&vis_logger);

  bool is_success = false;
  bool is_collision = false;
  int i = 0;
  int s = 0;
  double total_elapsed = 0.0;
  double total_rollout = 0.0;
  double total_clustering = 0.0;
  double total_connection = 0.0;
  double total_guide = 0.0;
  double f_err = 0.0;

  for (i = 0; i < maxiter; ++i) {
    vis_logger.beginStep(i);

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
      << total_elapsed << ',' << total_rollout << ',' << total_clustering << ','
      << total_connection << ',' << total_guide << ',' << f_err << '\n';

  csv.close();
  return 0;
}
