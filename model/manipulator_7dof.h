#pragma once

#include "model_base.h"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

class Manipulator : public ModelBase {
public:
  Manipulator();
  ~Manipulator();

  // Physical Constants
  const double g_acc = 9.81;
  const int dof = 7; // 6에서 7로 변경

  // 7-DOF Approximate Parameters (Franka Panda 스타일 추정치)
  const double L12 = 0.333;
  const double L34 = 0.316;
  const double L56 = 0.384;
  const double L7e = 0.107;

  // Limits
  Eigen::VectorXd q_max;
  Eigen::VectorXd q_dot_max;
  Eigen::VectorXd tau_max;

  // Simplified Dynamics Parameters
  Eigen::VectorXd inertia_diag;

  // DH Parameters (7-DOF Panda 스타일 추정치)
  // a: Link length, d: Link offset, alpha: Link twist
  const double dh_a[7] = {0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088};
  const double dh_d[7] = {0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107};
  const double dh_alpha[7] = {-M_PI / 2, M_PI / 2, M_PI / 2, M_PI / 2,
                              -M_PI / 2, M_PI / 2, 0.0};

  std::array<Eigen::Matrix4d, 7>
  computeForwardKinematics(const Eigen::VectorXd &q) const;
  std::vector<Eigen::VectorXd>
  return_joint_pose(const Eigen::VectorXd &q) const;
  virtual std::vector<Eigen::Vector3d>
  getAllJointPositions(const Eigen::VectorXd &x) const override;
};

Manipulator::Manipulator() {
  // 1. Dimensions Settings
  dim_x = 2 * dof; // 14 (7 angles + 7 velocities)
  dim_u = dof;     // 7 joint torques

  // 2. Physical Limits & Constraints Initialization
  q_max = Eigen::VectorXd::Constant(dof, 2.89); // Panda joint limits approx
  q_dot_max = Eigen::VectorXd::Constant(dof, 150.0 * M_PI / 180.0);

  // Torque limits (7-DOF arm class)
  tau_max = Eigen::VectorXd::Zero(dof);
  tau_max << 87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0;

  // Diagonal Mass Matrix (Inertia) - 7축에 맞춰 확장
  inertia_diag = Eigen::VectorXd(dof);
  inertia_diag << 5.0, 5.0, 3.0, 3.0, 2.0, 1.0, 0.5;

  // 3. Continuous Time System: x_dot = f(x, u)
  f = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &u) -> Eigen::VectorXd {
    Eigen::VectorXd q = x.head(dof);
    Eigen::VectorXd q_dot = x.tail(dof);
    Eigen::VectorXd tau = u;

    // Compute Gravity term (7-DOF 간소화 모델)
    // 실제 환경에서는 고수준 다이나믹스 라이브러리(Pinocchio 등) 권장
    Eigen::VectorXd Cg = Eigen::VectorXd::Zero(dof);

    // 간소화된 중력 계산 (J2, J4 등 주요 굽힘 관절에 가중치)
    Cg(1) = 15.0 * std::cos(q(1));
    Cg(3) = 10.0 * std::cos(q(1) + q(3));

    // M * q_ddot = tau - Cg -> q_ddot 계산
    Eigen::VectorXd q_ddot = (tau - Cg).cwiseQuotient(inertia_diag);

    Eigen::VectorXd x_dot(dim_x);
    x_dot << q_dot, q_ddot;
    return x_dot;
  };

  // 4. Stage Cost Function
  q = [this](const Eigen::VectorXd &x, const Eigen::VectorXd &u) -> double {
    double control_cost = 0.0001 * u.squaredNorm();
    double velocity_cost = 0.01 * x.tail(dof).squaredNorm();
    return control_cost + velocity_cost;
  };

  // 5. Terminal Cost Function
  p = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &x_target) -> double {
    Eigen::VectorXd q_state = x.head(dof);
    Eigen::VectorXd q_target = x_target.head(dof);

    double config_error = (q_state - q_target).norm();

    // 7축 FK 기반 End-effector 위치 오차 계산
    auto T_global = computeForwardKinematics(q_state);
    Eigen::Vector3d ee_pos = T_global[6].block<3, 1>(0, 3);

    auto T_target = computeForwardKinematics(q_target);
    Eigen::Vector3d target_pos = T_target[6].block<3, 1>(0, 3);

    double task_error = (ee_pos - target_pos).norm();

    return config_error * 10.0 + task_error * 100.0;
  };

  // 6. Input Constraint Function
  h = [&](Eigen::Ref<Eigen::MatrixXd> U) -> void {
    for (int i = 0; i < U.cols(); ++i) {
      U.col(i) = U.col(i).cwiseMin(tau_max).cwiseMax(-tau_max);
    }
  };
}

Manipulator::~Manipulator() {}

std::array<Eigen::Matrix4d, 7>
Manipulator::computeForwardKinematics(const Eigen::VectorXd &q) const {
  std::array<Eigen::Matrix4d, 7> T_global;
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

std::vector<Eigen::VectorXd>
Manipulator::return_joint_pose(const Eigen::VectorXd &q) const {
  auto T_global = computeForwardKinematics(q);
  std::vector<Eigen::VectorXd> joint_pose(7);
  for (int i = 0; i < 7; ++i) {
    joint_pose[i] = T_global[i].block<3, 1>(0, 3);
  }
  return joint_pose;
}

std::vector<Eigen::Vector3d>
Manipulator::getAllJointPositions(const Eigen::VectorXd &x) const {
  Eigen::VectorXd q = x.head(dof);
  auto T_global = computeForwardKinematics(q);
  std::vector<Eigen::Vector3d> joint_positions(7);
  for (int i = 0; i < 7; ++i) {
    joint_positions[i] = T_global[i].block<3, 1>(0, 3);
  }
  return joint_positions;
}