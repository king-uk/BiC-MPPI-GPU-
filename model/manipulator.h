#pragma once

#include "model_base.h"
#include <Eigen/Dense>
#include <cmath>
#include <eigen3/Eigen/src/Core/Diagonal.h>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>

#include <array>
#include <vector>

// =============================================================================
// Manipulator Model - Velocity Control (1st-order Kinematic Model)
//
// [One-step MPPI 논문 방식을 참고한 재설계]
//
// 핵심 변경:
//   - 제어 입력: 토크(τ) → 관절 속도(q̇) [velocity control]
//   - 상태:      [q, q̇] (12-dim) → [q] (6-dim) [1차 시스템]
//   - 동역학:    M*q̈ = τ-Cg (복잡, gravity 불안정) → q_next = q + dt*u (단순,
//   안정)
//
// 장점:
//   1. Gravity 항 없음 → 발산 없음
//   2. 상태 공간이 절반 → MPPI rollout이 2배 빠름
//   3. 비용 함수를 rad 단위로 통일 가능
//   4. 실제 velocity-controlled 로봇과 일치 (ROS MoveIt velocity controller 등)
// =============================================================================

class Manipulator : public ModelBase {
public:
  Manipulator();
  ~Manipulator();

  // Physical Constants
  const int dof = 6;

  // Limits
  Eigen::VectorXd q_min;     // Joint position lower limit [rad]
  Eigen::VectorXd q_max;     // Joint position upper limit [rad]
  Eigen::VectorXd q_dot_max; // Joint velocity limit [rad/s]

  // Obstacle bounds for stage cost repulsion: {xmin, xmax, ymin, ymax, zmin,
  // zmax}
  std::vector<std::array<double, 6>> obstacles;
  void addObstacle(double x, double y, double z, double w, double h, double d) {
    obstacles.push_back({x, x + w, y, y + h, z, z + d});
  }

  // DH Parameters (RB5 추정치)
  // a: Link length, d: Link offset, alpha: Link twist
  const double L2 = 0.427;
  const double L3 = 0.357;
  const double dh_a[6] = {0.0, -L2, -L3, 0.0, 0.0, 0.0};
  const double dh_d[6] = {0.15, 0.0, 0.0, 0.11, 0.09, 0.09};
  const double dh_alpha[6] = {M_PI / 2, 0.0, 0.0, M_PI / 2, -M_PI / 2, 0.0};

  // 각 관절의 SE(3) 상태 반환
  std::array<Eigen::Matrix4d, 6>
  computeForwardKinematics(const Eigen::VectorXd &q) const;

  // 각 관절의 position 반환
  // std::vector<Eigen::VectorXd>
  // return_joint_pose(const Eigen::VectorXd &q) const;

  // 각 관절의 position 반환
  virtual std::vector<Eigen::Vector3d>
  getAllJointPositions(const Eigen::VectorXd &x) const override;
};

Manipulator::Manipulator() {
  // ==========================================================================
  // 1. Dimensions
  // ==========================================================================
  // [변경] x = q (6-dim joint angles), u = q_dot (6-dim joint velocities)
  dim_x = dof; // 관절 위치만 (속도는 제어 입력이므로 상태에서 제거)
  dim_u = dof; // 관절 속도 명령

  // ==========================================================================
  // 2. Physical Limits
  // [One-step MPPI 논문] 관절 한계를 ±q 범위로 명확히 설정
  // ==========================================================================
  // RB5 approximate joint limits [rad]
  q_min = Eigen::VectorXd(dof);
  q_min << -2.0 * M_PI, -2.0 * M_PI, -165.0 * M_PI / 180.0, -2.0 * M_PI,
      -2.0 * M_PI, -2.0 * M_PI;

  q_max = Eigen::VectorXd(dof);
  q_max << 2.0 * M_PI, 2.0 * M_PI, 165.0 * M_PI / 180.0, 2.0 * M_PI, 2.0 * M_PI,
      2.0 * M_PI;

  // 속도 한계 [rad/s]: 각 관절별 최대 속도
  // One-step MPPI에서는 u = q_dot이므로, q_dot_max가 제어 한계
  q_dot_max = Eigen::VectorXd(dof);
  q_dot_max << 1.5, 1.5, 1.5, 2.0, 2.0, 2.0; // [rad/s]

  // ==========================================================================
  // 3. Kinematic Dynamics: x_{t+1} = x_t + dt * u
  //
  // [핵심 변경] 2차 동역학(토크 → 가속도 → 속도 → 위치)을
  //            1차 운동학(속도 명령 → 위치)으로 대체
  //
  // 장점:
  //   - Gravity 보상 불필요 → 발산 없음
  //   - 적분 오차 최소 (선형 시스템)
  //   - 실제 velocity controller와 동일한 인터페이스
  // ==========================================================================
  f = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &u) -> Eigen::VectorXd {
    // x = q (joint positions), u = q_dot (velocity command)
    // x_dot = u (단순 적분기)
    return u;
  };

  // ==========================================================================
  // 4. Stage Cost: l(x, u)
  //
  // [One-step MPPI 논문 방식]
  // 각 time step마다 목표로 향하는 gradient를 제공:
  //   - control_effort: 속도 크기 페널티 (smooth motion)
  //   - joint_limit:    관절 한계 소프트 페널티 (barrier에 접근 시 비용 증가)
  //
  // 주의: stage cost에서 x_target은 접근 불가(함수 시그니처 제약).
  //       따라서 joint limit 위반과 control effort만 penalize.
  //       Goal reaching은 terminal cost p()가 담당.
  // ==========================================================================

  // stage cost
  q = [this](const Eigen::VectorXd &x, const Eigen::VectorXd &u) -> double {
    // 1) Control effort
    double control_cost = 1 * u.norm();

    // 2) Joint limit soft barrier (80% 초과 시 페널티)
    double joint_limit_cost = 0.0;
    for (int i = 0; i < dof; ++i) {
      double q_range = (q_max(i) - q_min(i)) * 0.5;
      double q_center = (q_max(i) + q_min(i)) * 0.5;
      double normalized = (x(i) - q_center) / (q_range * 0.8);
      if (std::abs(normalized) > 1.0) {
        double excess = std::abs(normalized) - 1.0;
        joint_limit_cost += 5.0 * excess * excess; // 5*(excess)^2
      }
    }

    // 3) Obstacle repulsion: FK로 각 관절 위치 계산 후 장애물 반발력 부여
    // 장애물 중심으로부터 safe_r 이내의 관절에 이차 페널티
    // → MPPI rollout에 장애물 회피 gradient 제공 (핵심!)
    double obs_cost = 0.0;
    if (!obstacles.empty()) {
      auto T_fk = computeForwardKinematics(x);
      for (const auto &obs : obstacles) {
        // obs = {xmin, xmax, ymin, ymax, zmin, zmax}
        double cx = (obs[0] + obs[1]) * 0.5;
        double cy = (obs[2] + obs[3]) * 0.5;
        const double safe_r = 0.30;
        for (int k = 1; k < dof; ++k) {
          Eigen::Vector3d p1 = T_fk[k - 1].block<3, 1>(0, 3);
          Eigen::Vector3d p2 = T_fk[k].block<3, 1>(0, 3);

          // 조인트 자체와 링크 중간점(Mid-point) 모두 검사
          std::vector<Eigen::Vector3d> points_to_check = {p2, (p1 + p2) * 0.5};

          for (const auto &p : points_to_check) {
            // --- 바닥(z < 0) 충돌 방지 ---
            if (p.z() < 0.0) {
              obs_cost += 50000.0 * p.z() * p.z();
            }

            // 장애물 위(z > obs_z_max)로 통과 시 페널티 없음
            if (p.z() > obs[5] + 0.03)
              continue;

            // AABB(Bounding Box) 표면으로부터의 최단 거리 계산
            double dx = std::max({0.0, obs[0] - p.x(), p.x() - obs[1]});
            double dy = std::max({0.0, obs[2] - p.y(), p.y() - obs[3]});
            double d = std::sqrt(dx * dx + dy * dy);

            // Two-tier 장애물 회피 구역
            const double death_margin = 0.08; // 8cm 접근 시 완전 사망 처리
            const double warn_margin =
                0.15; // 15cm부터 경고 (타겟 지점 d=0.161은 0 코스트 유지)

            if (d < death_margin) {
              // 장애물 충돌 또는 극히 위험한 구역: 무조건 상수 형태의 절망적
              // 페널티 (뚫고 가기 불가능)
              obs_cost += 100000.0;
            } else if (d < warn_margin) {
              // 경고 구역: 다가갈수록 급격히 증가하는 2차 함수 언덕
              obs_cost += 50000.0 * (warn_margin - d) * (warn_margin - d);
            }

            // Z축 상승 보조 그라디언트: 경고 구역 이내에 있을 경우 위로
            // 도망치는 것을 엄청난 이득으로 설정
            if (d < warn_margin) {
              double target_z = obs[5] + 0.15;
              if (p.z() < target_z) {
                obs_cost += 200.0 * (target_z - p.z()); // 낮을수록 가중
              }
            }
          }
        }
      }
    }

    // 4) 스테이지 레벨 방향 제약 (앞/뒤 도망 극단적 차단)
    double backward_penalty = 0.0;
    if (x(0) < 0.0) {
      backward_penalty = 100000.0 * x(0) * x(0); // 절대 뒤로 가지 못함
    }

    // MPPI 특성상 장애물을 넘자마자 질주하여 Overshoot 하는 것을 차단 (Forward
    // Wall)
    double forward_penalty = 0.0;
    if (x(0) > M_PI / 2.0 + 0.1) {
      double excess = x(0) - (M_PI / 2.0 + 0.1);
      forward_penalty = 100000.0 * excess * excess;
    }

    // return control_cost + joint_limit_cost + obs_cost + backward_penalty +
    //        forward_penalty;
    return control_cost;
  };

  // ==========================================================================
  // 5. Terminal Cost: p(x, x_target)
  //
  // [One-step MPPI 논문 방식]
  // 목표 도달을 위한 비용. 단위를 rad로 통일:
  //   - config_error: 관절 공간 오차 (primary, 직접적)
  //   - task_error:   EE 위치 오차 (secondary, 세밀한 위치 조절용)
  //   - velocity_stop: 목표에서 zero velocity 유도 (해당 없음 - 제어 입력이
  //   속도이므로
  //                    stage cost의 control_cost가 이미 속도를 억제)
  //
  // 가중치 원칙 (One-step MPPI):
  //   - config_error와 task_error는 같은 단위(rad, m)이 아니므로
  //     스케일을 맞춰 설정. 1 rad ≈ 0.4m (L2 링크)이므로
  //     config:task = 1:0.1 정도가 적절 (config가 훨씬 직접적)
  // ==========================================================================
  p = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &x_target) -> double {
    Eigen::VectorXd q_state = x;
    Eigen::VectorXd q_target = x_target.head(dof);

    // -- Configuration Space Error (관절공간 오차) --
    // 제곱 norm 사용: 원점 근처에서 gradient가 명확하고 연속적
    // 단일 타겟 모드에서 장애물을 부드럽게 우회한 후에도, 타겟에 강력하게
    // 끌리도록 10000 부여
    Eigen::VectorXd q_err = q_state - q_target;

    Eigen::MatrixXd q_err_mat = Eigen::MatrixXd::Identity(dof, dof);
    double config_error =
        1000.0 * (q_err.transpose() * q_err_mat * q_err).trace();

    // -- Task Space Error (End-Effector 위치 오차) --
    std::array<Eigen::Matrix4d, 6> T_fk = computeForwardKinematics(q_state);
    std::array<Eigen::Matrix4d, 6> T_tgt = computeForwardKinematics(q_target);
    Eigen::Vector3d ee_pos = T_fk[5].block<3, 1>(0, 3);
    Eigen::Vector3d target_pos = T_tgt[5].block<3, 1>(0, 3);
    double task_error = 500.0 * (ee_pos - target_pos).squaredNorm();

    // -- J1 방향 제약: 반드시 양의 방향(0→π/2)으로 이동 --
    // J1 < 0이면 강하게 음수 방향 탐색 차단
    // J1 > π/2 + 0.3 이면 overshoot 페널티를 주어 2.0 rad 등에서 머물지 않게 핣
    double direction_cost = 0.0;
    if (q_state(0) < 0.0) {
      // 뒤로 도망가는 것을 obstacle_cost(10000)보다 훨씬 비싸게 만들어 원천
      // 차단
      direction_cost = 50000.0 * q_state(0) * q_state(0);
    } else if (q_state(0) > M_PI / 2.0 + 0.3) {
      double excess = q_state(0) - (M_PI / 2.0 + 0.3);
      direction_cost = 500.0 * excess * excess;
    }

    // return config_error + task_error + direction_cost;
    return config_error;
  };

  // ==========================================================================
  // 6. Input Constraint: h(U) - 속도 명령을 q_dot_max 이내로 clipping
  // ==========================================================================
  h = [&](Eigen::Ref<Eigen::MatrixXd> U) -> void {
    for (int i = 0; i < U.cols(); ++i) {
      U.col(i) = U.col(i).cwiseMin(q_dot_max).cwiseMax(-q_dot_max);
    }
  };
}

Manipulator::~Manipulator() {}

// joint space를 SE(3)로 변환
std::array<Eigen::Matrix4d, 6>
Manipulator::computeForwardKinematics(const Eigen::VectorXd &q) const {
  std::array<Eigen::Matrix4d, 6> T_global;
  Eigen::Matrix4d T_prev = Eigen::Matrix4d::Identity();

  for (int i = 0; i < dof; ++i) {
    double theta = q(i);
    double a = dh_a[i];
    double d = dh_d[i];
    double alpha = dh_alpha[i];

    double c_theta = std::cos(theta);
    double s_theta = std::sin(theta);
    double c_alpha = std::cos(alpha);
    double s_alpha = std::sin(alpha);

    Eigen::Matrix4d T_local;
    T_local << c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta,
        s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta, 0.0,
        s_alpha, c_alpha, d, 0.0, 0.0, 0.0, 1.0;

    T_global[i] = T_prev * T_local;
    T_prev = T_global[i];
  }
  return T_global;
}

// std::vector<Eigen::VectorXd>
// Manipulator::return_joint_pose(const Eigen::VectorXd &q) const {
//   std::array<Eigen::Matrix4d, 6> T_global = computeForwardKinematics(q);
//   std::vector<Eigen::VectorXd> joint_pose(6);
//   for (int i = 0; i < 6; ++i) {
//     joint_pose[i] = T_global[i].block<3, 1>(0, 3);
//   }
//   return joint_pose;
// }

std::vector<Eigen::Vector3d>
Manipulator::getAllJointPositions(const Eigen::VectorXd &x) const {
  // x = q (dim_x = dof)
  Eigen::VectorXd q = x.head(dof);
  std::array<Eigen::Matrix4d, 6> T_global = computeForwardKinematics(q);
  std::vector<Eigen::Vector3d> joint_positions(6);
  for (int i = 0; i < dof; ++i) {
    joint_positions[i] = T_global[i].block<3, 1>(0, 3);
  }
  return joint_positions;
}
