// ============================================================
//  src/bicycle/rrt_connect.cpp
//  RRT-Connect 플래너 – Bicycle 모델, BiC-MPPI 와 비교용
//
//  출력 형식 (stdout):
//    s  map  success  iter  elapsed_plan  total_elapsed  err
//  CSV (rrt_paths.csv):
//    s,map,iter,x,y,theta,v,err
// ============================================================
#include <bicycle.h>
#include <rrt_connect.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = Bicycle();

  // ---- 솔버 파라미터 설정 ----
  RRTConnectParam param;
  param.dt = 0.05;         // 적분 시간 간격 (bi_mppi 의 dt=0.1 보다 작게 설정)
  param.max_iter = 5000;   // 최대 RRT 반복
  param.step_size = 0.4;   // 한 steer 에서 허용되는 최대 2D 이동 거리 [m]
  param.goal_tol = 0.2;    // 목표 도달 허용 오차 [m]
  param.control_steps = 5; // 한 번 steer 시 dt × control_steps 만큼 적분
  param.num_controls = 20; // 랜덤 제어 후보 수

  // ---- 각도 랩핑 및 거리 가중치 설정 ----
  param.angle_idx = {2}; // theta 인덱스는 2번
  param.state_weights.resize(model.dim_x);
  param.state_weights << 1.0, 1.0, 1.0, 0.1; // x, y, theta, v 가중치

  // 상태 공간 범위:  x∈[0,3], y∈[-0.5,6], theta∈[-π,π], v∈[-2,2]
  param.x_min.resize(model.dim_x);
  param.x_min << 0.0, -0.5, -M_PI, -2.0;
  param.x_max.resize(model.dim_x);
  param.x_max << 3.0, 6.0, M_PI, 2.0;

  // 제어 입력 범위:  a∈[-3,3], delta∈[-π/4, π/4]
  param.u_min.resize(model.dim_u);
  param.u_min << -3.0, -M_PI / 4.0;
  param.u_max.resize(model.dim_u);
  param.u_max << 3.0, M_PI / 4.0;

  // 출발 / 목표 (bi_mppi.cpp 와 동일)
  param.x_init.resize(model.dim_x);
  param.x_init << 2.5, 0.0, M_PI_2, 0.0;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, M_PI_2, 0.0;

  int maxiter = 200; // move() 최대 호출 횟수

  // ---- CSV 초기화 ----
  std::ofstream csv_file("rrt_paths.csv");
  csv_file << "s,map,iter,x,y,theta,v,err\n";

  std::ofstream result_file("rrt_bicycle_results.csv");
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

    for (int map = 299; map >= 0; --map) {
      CollisionChecker collision_checker;
      collision_checker.loadMap("../BARN_dataset/txt_files/output_" +
                                    std::to_string(map) + ".txt",
                                0.1);

      // ---- 솔버 생성 & 초기화 ----
      RRTConnect solver(model);
      solver.init(param);
      solver.setCollisionChecker(&collision_checker);

      bool is_success = false;
      bool is_collision = false;
      int i = 0;
      double total_elapsed = 0.0;
      double plan_elapsed = 0.0;
      double f_err = 0.0;

      // ---- 첫 번째 플랜 실행 ----
      solver.solve();
      plan_elapsed = solver.elapsed;
      total_elapsed = plan_elapsed;

      for (i = 0; i < maxiter; ++i) {
        // 경로 한 스텝 실행
        solver.move();

        // CSV 기록
        csv_file << s << "," << map << "," << i << "," << solver.x_init(0)
                 << "," << solver.x_init(1) << "," << solver.x_init(2) << ","
                 << solver.x_init(3) << "," << f_err << "\n";

        if (collision_checker.getCollisionGrid(solver.x_init)) {
          is_collision = true;
          break;
        }

        f_err = (solver.x_init - param.x_target).norm();
        if (f_err < 0.15) {
          is_success = true;
          break;
        }

        // 경로 끝에 도달했어도 아직 목표에 도달하지 못했으면 재플래닝
        if (solver.isPathFinished() && i < maxiter - 1) {
          // 탐색에 완전히 실패해서 시작점 그대로인 경우, 계속 재탐색하면
          // 프로그램이 멈추므로 중단
          if (solver.getPathSize() <= 1) {
            break;
          }
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
    }
  }

  csv_file.close();
  result_file.close();
  return 0;
}
