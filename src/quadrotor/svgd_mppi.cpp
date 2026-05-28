#include <quadrotor.h>
#include <svgd_mppi.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = Quadrotor();

  using Solver = SVGDMPPI;
  using SolverParam = SVGDMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = 50;
  param.Tb = 50;
  param.x_init.resize(model.dim_x);
  param.x_init << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;

  param.Nf = 120;
  param.Nb = 120;
  param.Nr = 3000;
  param.Ns = 10;
  param.istep = 5;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 1.5, 1.5, 1.5;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;

  int maxiter = 200;

  std::ofstream csv("result_quadrotor_svgd_mppi.csv");
  csv << "map,is_failed,is_landed,iter,elapsed,f_err\n";

  for (int map = 299; map >= 0; --map) {
    CollisionChecker collision_checker = CollisionChecker();
    collision_checker.loadMap("../BARN_dataset/txt_files/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);

    Solver solver(model);
    solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
    solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
    // quadrotor: gravity compensation on thrust channel
    solver.U_f0.row(2).array() += model.g;
    solver.U_b0.row(2).array() += model.g;
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);

    bool is_landed = false;
    bool is_failed = true;
    int i = 0;
    double total_elapsed = 0.0;
    double f_err = 0.0;

    for (i = 0; i < maxiter; ++i) {
      solver.solve();
      solver.move();
      total_elapsed += solver.elapsed;

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
              << '\t' << total_elapsed << '\t' << f_err << std::endl;
    csv << map << ',' << is_failed << ',' << is_landed << ',' << i << ','
        << total_elapsed << ',' << f_err << '\n';
  }

  csv.close();
  return 0;
}
