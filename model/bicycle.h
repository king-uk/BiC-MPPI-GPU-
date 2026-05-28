#pragma once

#include "model_base.h"
#include <cmath>

class Bicycle : public ModelBase {
public:
  Bicycle();
  ~Bicycle();

  double L; // 휠베이스 (앞바퀴와 뒷바퀴 축 사이의 거리)
};

Bicycle::Bicycle() {
  // Dimensions
  // x = [x, y, theta, v]
  dim_x = 4;

  // u = [a, delta] (가속도, 조향각)
  dim_u = 2;

  // 파라미터 초기화
  L = 0.2; // 예시 휠베이스 (m)

  // Continuous Time System[cite: 1]
  f = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &u) -> Eigen::MatrixXd {
    Eigen::VectorXd x_dot(x.rows());
    x_dot(0) = x(3) * cos(x(2));     // x_dot = v * cos(theta)
    x_dot(1) = x(3) * sin(x(2));     // y_dot = v * sin(theta)
    x_dot(2) = x(3) * tan(u(1)) / L; // theta_dot = (v / L) * tan(delta)
    x_dot(3) = u(0);                 // v_dot = a
    return x_dot;
  };

  // Stage Cost Function[cite: 1]
  q = [this](const Eigen::VectorXd &x, const Eigen::VectorXd &u) -> double {
    // 제어 입력(가속도, 조향각)의 크기를 최소화[cite: 2, 3]
    double t = 0.8;
    return t * u(0) * u(0) + (1 - t) * u(1) * u(1);
  };

  // Terminal Cost Function[cite: 1]
  p = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &x_target) -> double {
    // 목표 위치(x, y)에 도달하도록 페널티 부여[cite: 2, 3]
    return (x.head(2) - x_target.head(2)).norm();
  };

  // Input Clamping[cite: 1, 3]
  h = [&](Eigen::Ref<Eigen::MatrixXd> U) -> void {
    // U.row(0): 가속도(a) 제한 (예: -3.0 ~ 3.0 m/s^2)
    U.row(0) = U.row(0).cwiseMax(-3.0).cwiseMin(3.0);

    // U.row(1): 조향각(delta) 제한 (예: -pi/4 ~ pi/4 rad)
    double max_steer = M_PI / 4.0;
    U.row(1) = U.row(1).cwiseMax(-max_steer).cwiseMin(max_steer);
    return;
  };
}

Bicycle::~Bicycle() {}