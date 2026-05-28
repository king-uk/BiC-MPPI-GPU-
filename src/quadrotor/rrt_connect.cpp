// ============================================================
//  src/quadrotor/rrt_connect.cpp
//  RRT-Connect 플래너 – Quadrotor 모델, BiC-MPPI 와 비교용
//
//  출력 형식 (stdout):
//    map  is_failed  is_landed  iter  plan_elapsed  total_elapsed  err
// ============================================================
#include <quadrotor.h>
#include <rrt_connect.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = Quadrotor();

  // ---- 솔버 파라미터 ----
  RRTConnectParam param;
  param.dt = 0.05;
  param.max_iter = 8000;
  param.step_size = 0.5;
  param.goal_tol = 0.3;
  param.control_steps = 5;
  param.num_controls = 30;

  // ---- 각도 랩핑 및 거리 가중치 설정 ----
  param.state_weights.resize(model.dim_x);
  param.state_weights << 1.0, 1.0, 1.0, 0.2, 0.2,
      0.2; // x,y,z는 크게, 속도는 작게

  // 상태 공간:  x∈[0,3], y∈[-0.5,6], z∈[-1,6], vx/vy/vz∈[-5,5]
  param.x_min.resize(model.dim_x);
  param.x_min << 0.0, -0.5, -1.0, -5.0, -5.0, -5.0;
  param.x_max.resize(model.dim_x);
  param.x_max << 3.0, 6.0, 6.0, 5.0, 5.0, 5.0;

  // 제어:  ax/ay∈[-5,5], az∈[-5+g, 5+g]
  param.u_min.resize(model.dim_u);
  param.u_min << -5.0, -5.0, model.g - 5.0;
  param.u_max.resize(model.dim_u);
  param.u_max << 5.0, 5.0, model.g + 5.0;

  // 출발 / 목표  (bi_mppi.cpp 와 동일)
  param.x_init.resize(model.dim_x);
  param.x_init << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;

  int maxiter = 8000; // 최대 step

  std::ofstream result_file("rrt_quadrotor_results.csv");
  result_file << "map,failed,landed,iter,plan_elapsed,total_elapsed,f_err\n";

  std::ofstream path_file("rrt_quadrotor_paths.csv");
  path_file << "map,iter,x,y,z,vx,vy,vz,err\n";

  for (int map = 299; map >= 0; --map) {
    CollisionChecker collision_checker;
    collision_checker.loadMap("../BARN_dataset/txt_files/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);

    RRTConnect solver(model);
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);

    bool is_landed = false;
    bool is_failed = false;
    int i = 0;
    double total_elapsed = 0.0;
    double plan_elapsed = 0.0;
    double f_err = 0.0;

    // 첫 플랜
    solver.solve();
    plan_elapsed = solver.elapsed;
    total_elapsed = plan_elapsed;

    for (i = 0; i < maxiter; ++i) {
      // 경로 한 스텝 전진
      solver.move();

      path_file << map << "," << i << "," << solver.x_init(0) << ","
                << solver.x_init(1) << "," << solver.x_init(2) << ","
                << solver.x_init(3) << "," << solver.x_init(4) << ","
                << solver.x_init(5) << "," << f_err << "\n";

      // Quadrotor: 현재 상태 위치로 충돌 체크 (x, y 사용)
      if (collision_checker.getCollisionGrid(solver.x_init)) {
        is_failed = true;
        break;
      }

      f_err = (solver.x_init.head(2) - param.x_target.head(2)).norm();
      if (solver.x_init(2) < 0) {
        is_landed = true;
        is_failed = false;
        break;
      }

      // 재플래닝
      if (solver.isPathFinished() && i < maxiter - 1) {
        if (solver.getPathSize() <= 1)
          break;
        RRTConnectParam reparam = param;
        reparam.x_init = solver.x_init;
        solver.init(reparam);
        solver.setCollisionChecker(&collision_checker);
        solver.solve();
        total_elapsed += solver.elapsed;
      }
    }

    std::cout << map << '\t' << is_failed << '\t' << is_landed << '\t' << i
              << '\t' << plan_elapsed << '\t' << total_elapsed << '\t' << f_err
              << std::endl;

    result_file << map << "," << is_failed << "," << is_landed << "," << i
                << "," << plan_elapsed << "," << total_elapsed << "," << f_err
                << "\n";
  }

  result_file.close();
  path_file.close();
  return 0;
}
