#pragma once

#include "model_base.h"

class Velo : public ModelBase {
public:
  Velo();
  ~Velo();
};

Velo::Velo() {
  // Dimensions
  dim_x = 4;
  dim_u = 2;

  // Continous Time System
  f = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &u) -> Eigen::MatrixXd {
    Eigen::VectorXd x_dot(x.rows());
    x_dot(0) = x(2);
    x_dot(1) = x(3);
    x_dot(2) = u(0);
    x_dot(3) = u(1);
    return x_dot;
  };

  // Stage Cost Function
  q = [this](const Eigen::VectorXd &x, const Eigen::VectorXd &u) -> double {
    return u.norm();
  };

  // Terminal Cost Function
  p = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &x_target) -> double {
    return (x.head(2) - x_target.head(2)).norm();
  };

  // Input Clamping
  h = [&](Eigen::Ref<Eigen::MatrixXd> U) -> void {
    // U.row(0) = U.row(0).cwiseMax(velocity).cwiseMin(velocity);
    U.row(0) = U.row(0).cwiseMax(0.0).cwiseMin(1.0);
    // U.row(0) = U.row(0).cwiseMax(0.0).cwiseMin(1.0);
    // U.row(1) = U.row(1).cwiseMax(-0.5).cwiseMin(0.5);
    U.row(1) = U.row(1).cwiseMax(0).cwiseMin(1.0);
    return;
  };
}

Velo::~Velo() {}