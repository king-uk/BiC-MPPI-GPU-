#pragma once

#include "mppi_3D.h"

class LogMPPI3D : public MPPI3D {
public:
    template<typename ModelClass>
    LogMPPI3D(ModelClass model);
    ~LogMPPI3D();

    Eigen::Rand::LognormalGen<double> log_norm_gen{0.0, 1.0};

    Eigen::MatrixXd getNoise(const int &T_) override {
        Eigen::MatrixXd log_distribution = log_norm_gen.template generate<Eigen::MatrixXd>(dim_u, T_, urng);
        return (sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T_, urng)).array() * log_distribution.array();
    }
};

template<typename ModelClass>
LogMPPI3D::LogMPPI3D(ModelClass model) : MPPI3D(model) {
}

LogMPPI3D::~LogMPPI3D() {
}
