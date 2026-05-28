#include "bi_mppi_3D.h"
#include "manipulator.h"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

int main() {
  std::cout << "Starting Manipulator Bi-MPPI (Velocity Control)..."
            << std::endl;
  // ==========================================================================
  // 환경(Environment) 설정
  // ==========================================================================
  CollisionChecker collision_checker = CollisionChecker();

  // 이곳에서 장애물을 한 번만 정의하면 됩니다.
  // 인터페이스: addRectangle(x_min, y_min, z_min, width, height, depth)
  collision_checker.addRectangle(-0.60, -0.60, -1.00, 0.20, 0.20, 1.5);
  // collision_checker.addRectangle(-10, -10, -0.20, 20, 20, 0.1);

  // 모델 생성 및 장애물 동기화 (CollisionChecker의 장애물을 Model로 복사)
  auto model = Manipulator();
  for (const auto &rect : collision_checker.rectangles) {
    // rect = {xmin, xmax, ymin, ymax, zmin, zmax}
    model.addObstacle(rect[0], rect[2], rect[4], rect[1] - rect[0],
                      rect[3] - rect[2], rect[5] - rect[4]);
  }

  using Solver = BiMPPI;
  using SolverParam = BiMPPIParam;

  SolverParam param;

  // ==========================================================================
  // OCP 파라미터 설정 (One-step MPPI 논문 참고)
  // ==========================================================================

  // [시간 스텝]
  param.dt = 0.1;

  // [Horizon] 거대한 장애물을 넘기 위해서는 깊은 미래 예측이 필수
  // 60 샘플 * 0.05s = 3.0초 예측 (BiC 양방향 결합시 6.0초 경로)
  param.Tf = 40;
  param.Tb = 40;

  // [초기 상태] q = 0 (모든 관절 원점)
  param.x_init.resize(model.dim_x);
  param.x_init.setZero();

  // [목표 상태] 단일 고정 타겟 (J1 = 90도)
  param.x_target.resize(model.dim_x);
  param.x_target.setZero();
  param.x_target(0) = M_PI / 2.0;

  // [샘플 수] 장애물 회피를 위한 대규모 샘플링
  param.Nf = 3000;
  param.Nb = 3000;
  param.Nr = 2000;

  // [gamma_u]
  param.gamma_u = 0.1;

  // [sigma_u] 장애물을 '위로' 뛰어넘기 위해선 수직 상승(J2)에 대해 거대한 탐색
  // 노이즈가 필요함
  Eigen::VectorXd sigma_u(model.dim_u);
  // J2에 3.0, J3에 2.0의 강력한 탐색 노이즈 부여하여 거대한 박스를 넘는 경로를
  // 찾도록 유돟
  sigma_u << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;
  param.sigma_u = sigma_u.asDiagonal();

  // [DBSCAN] 제어 샘플 공간 클러스터링 파라미터
  // sigma=1.0 rad/s 수준에서 epsilon=0.5이면 적절한 클러스터링
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 3;
  param.epsilon = 0.5;
  param.psi = 0.6;

  // ==========================================================================
  // Solver 초기화
  // ==========================================================================
  int maxiter = 500;
  double tolerance = 0.15; // 장애물 회피 시 허용 오차 완화 [rad] (~8.6도)

  Solver solver(model);
  // 초기 제어 시퀀스 (영 입력)
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);

  std::cout << "Obstacle: Set dynamically from CollisionChecker\n";

  solver.init(param);
  solver.setCollisionChecker(&collision_checker);

  // ==========================================================================
  // CSV 로깅 (환경 정보 포함)
  // ==========================================================================
  std::ofstream csv_file("x_init_bi_mppi_log.csv");
  if (!csv_file.is_open()) {
    std::cerr << "Error: Could not open x_init_bi_mppi_log.csv" << std::endl;
    return -1;
  }

  // 환경 정보 헤더 추가 (# 으로 시작하여 파이썬에서 파싱)
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

  // ==========================================================================
  // Main Loop
  // ==========================================================================
  bool is_reached = false;
  double total_elapsed = 0.0;
  double f_err = 0.0;
  int i = 0;

  for (i = 0; i < maxiter; ++i) {
    solver.solve();
    solver.move();
    total_elapsed += solver.elapsed;

    // 최종 목적지까지의 오차
    f_err = (solver.x_init - param.x_target).norm();

    // CSV 기록 (최종 오차 기준)
    csv_file << i;
    for (int j = 0; j < model.dim_x; ++j) {
      csv_file << "," << solver.x_init(j);
    }
    csv_file << "," << f_err << "\n";

    // 충돌 체크
    // if (collision_checker.getCollisionGrid(solver.x_init)) {
    //   std::cout << "Collision detected!\n";
    //   break;
    // }

    // 수렴 체크
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