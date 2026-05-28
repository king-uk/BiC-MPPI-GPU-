// ============================================================
//  src/wmrobot/rrt_connect.cpp
//  RRT-Connect 플래너 – WMRobot(차동 구동) 모델, BiC-MPPI 와 비교용
//
//  출력 형식 (stdout):
//    s  map  success  iter  plan_elapsed  total_elapsed
// ============================================================
#include <rrt_connect.h>
#include <wmrobot_map.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = WMRobotMap();

  // ---- 솔버 파라미터 ----
  RRTConnectParam param;
  param.dt = 0.05;
  param.max_iter = 50000;
  param.step_size = 0.4;
  param.goal_tol = 0.2;
  param.control_steps = 5;
  param.num_controls = 20;

  // ---- 각도 랩핑 및 거리 가중치 설정 ----
  param.angle_idx = {2}; // theta 인덱스
  param.state_weights.resize(model.dim_x);
  param.state_weights << 1.0, 1.0, 1.0; // x, y, theta 가중치

  // 상태 공간:  x∈[0,3], y∈[-0.5,6], theta∈[-π,π]
  param.x_min.resize(model.dim_x);
  param.x_min << 0.0, -0.5, -M_PI;
  param.x_max.resize(model.dim_x);
  param.x_max << 3.0, 6.0, M_PI;

  // 제어:  v∈[0,1], omega∈[-π/2, π/2]
  param.u_min.resize(model.dim_u);
  param.u_min << 0.0, -M_PI_2;
  param.u_max.resize(model.dim_u);
  param.u_max << 1.0, M_PI_2;

  // 출발 / 목표  (bi_mppi.cpp 와 동일)
  param.x_init.resize(model.dim_x);
  param.x_init << 2.5, 0.0, M_PI_2;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, M_PI_2;

  int maxiter = 2000;

  std::ofstream result_file("rrt_wmrobot_results.csv");
  result_file << "s,map,success,iter,plan_elapsed,total_elapsed\n";

  for (int s = 0; s < 2; ++s) {
    switch (s) {
    case 0:
      param.x_init(0) = 0.5;
      break;
    case 1:
      param.x_init(0) = 2.5;
      break;
    default:
      break;
    }

    // for (int map = 299; map >= 0; --map) {
    int map = 276;
    CollisionChecker collision_checker;
    collision_checker.loadMap("../BARN_dataset/txt_files/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);

    std::ofstream path_file("rrt_wmrobot_paths" + std::to_string(s) + "_" +
                            std::to_string(map) + ".csv");
    path_file << "x,y,theta\n";

    RRTConnect solver(model);
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);

    bool is_success = false;
    bool is_collision = false;
    int i = 0;
    double total_elapsed = 0.0;
    double plan_elapsed = 0.0;
    double f_err = 0.0;

    // 첫 플랜
    solver.solve();
    plan_elapsed = solver.elapsed;
    total_elapsed = plan_elapsed;

    for (i = 0; i < maxiter; ++i) {
      solver.move();

      path_file << solver.x_init(0) << "," << solver.x_init(1) << ","
                << solver.x_init(2) << "\n";

      if (collision_checker.getCollisionGrid(solver.x_init)) {
        is_collision = true;
        break;
      }

      f_err = (solver.x_init - param.x_target).norm();
      if (f_err < 0.15) {
        is_success = true;
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

    std::cout << s << '\t' << map << '\t' << is_success << '\t' << i << '\t'
              << plan_elapsed << '\t' << total_elapsed << std::endl;

    result_file << s << "," << map << "," << is_success << "," << i << ","
                << plan_elapsed << "," << total_elapsed << "\n";
    path_file.close();
  }
  // }

  result_file.close();

  return 0;
}
