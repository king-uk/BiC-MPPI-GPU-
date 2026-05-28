#include <bi_mppi.h>
#include <bicycle.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = Bicycle();

  using Solver = BiMPPI;
  using SolverParam = BiMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = 50;
  param.Tb = 50;

  param.x_init.resize(model.dim_x);
  param.x_init << 0.5, 0.0, M_PI_2, 0;

  param.x_target.resize(model.dim_x);
  param.x_target << 2.7, 0.4, M_PI, 0;

  param.Nf = 6000;
  param.Nb = 6000;
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

  std::ofstream csv("result_bicycle_bi_mppi.csv");
  csv << "is_failed,is_success,iter,elapsed,f_err\n";
  std::ofstream path_csv("path_bicycle_bi_mppi.csv");
  path_csv << "iter,x,y,theta,v\n";

  CollisionChecker collision_checker = CollisionChecker();
  collision_checker.loadMap("../parking.txt", 0.1);

  Solver solver(model);
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
  solver.init(param);
  solver.setCollisionChecker(&collision_checker);

  bool is_success = false;
  bool is_failed = false;
  int i = 0;
  double total_elapsed = 0.0;
  double f_err = 0.0;

  for (i = 0; i < maxiter; ++i) {
    solver.solve();
    solver.move();

    path_csv << i << "," << solver.x_init(0) << "," << solver.x_init(1) << ","
             << solver.x_init(2) << "," << solver.x_init(3) << "\n";

    total_elapsed += solver.elapsed;

    if (collision_checker.getCollisionGrid(solver.x_init)) {
      is_failed = true;
      break;
    } else {
      f_err = (solver.x_init - param.x_target).norm();
      if (f_err < 0.1) {
        is_success = true;
        break;
      }
    }
  }
  std::cout << is_failed << '\t' << is_success << '\t' << i << '\t'
            << total_elapsed << '\t' << f_err << std::endl;
  csv << is_failed << ',' << is_success << ',' << i << ',' << total_elapsed
      << ',' << f_err << '\n';

  csv.close();
  path_csv.close();
  return 0;
}
