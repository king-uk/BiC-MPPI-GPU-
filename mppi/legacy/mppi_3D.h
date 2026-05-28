#pragma once
#include "matplotlibcpp.h"

#include <EigenRand/EigenRand>

#include "mppi_param.h"
#include "collision_checker_3D.h"
#include "model_base.h"

#include <ctime>
#include <vector>
#include <chrono>
#include <iostream>

#include <omp.h>

class MPPI3D {
public:
    template<typename ModelClass>
    MPPI3D(ModelClass model);
    ~MPPI3D();

    void init(MPPIParam mppi_param);
    void setCollisionChecker(CollisionChecker *collision_checker);
    virtual Eigen::MatrixXd getNoise(const int &T);
    void move();
    virtual void solve();
    void show();
    void showTraj();

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    std::chrono::duration<double> elapsed_1;
    double elapsed;

    Eigen::MatrixXd U_0;
    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    Eigen::MatrixXd Uo;
    Eigen::MatrixXd Xo;

protected:
    int dim_x;
    int dim_u;

    // Discrete Time System
    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
    // Projection
    std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;
    // Joint Positions for Collision Checking
    std::function<std::vector<Eigen::Vector3d>(Eigen::VectorXd)> get_all_joint_positions;

    std::mt19937_64 urng{static_cast<std::uint_fast64_t>(std::time(nullptr))};
    Eigen::Rand::NormalGen<double> norm_gen{0.0, 1.0};

    // Parameters
    float dt;
    int T;
    int N;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
    
    CollisionChecker *collision_checker;

    Eigen::VectorXd u0;

    std::vector<Eigen::MatrixXd> visual_traj;

    bool checkCollision(const Eigen::VectorXd &x);
    Eigen::VectorXd rk4Step(const Eigen::VectorXd &x, const Eigen::VectorXd &u);
};

template<typename ModelClass>
MPPI3D::MPPI3D(ModelClass model) {
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;

    this->f = model.f;
    this->q = model.q;
    this->p = model.p;
    this->h = model.h;
    this->get_all_joint_positions = [model](const Eigen::VectorXd &x) {
        return model.getAllJointPositions(x);
    };
}

MPPI3D::~MPPI3D() {}

void MPPI3D::init(MPPIParam mppi_param) {
    this->dt = mppi_param.dt;
    this->T = mppi_param.T;
    this->x_init = mppi_param.x_init;
    this->x_target = mppi_param.x_target;
    this->N = mppi_param.N;
    this->gamma_u = mppi_param.gamma_u;
    this->sigma_u = mppi_param.sigma_u;

    u0 = Eigen::VectorXd::Zero(dim_u);
    Xo = Eigen::MatrixXd::Zero(dim_x, T+1);
    U_0 = Eigen::MatrixXd::Zero(dim_u, T);
}

void MPPI3D::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

Eigen::MatrixXd MPPI3D::getNoise(const int &T_) {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T_, urng);
}

bool MPPI3D::checkCollision(const Eigen::VectorXd &x) {
    auto joint_positions = get_all_joint_positions(x);
    if (joint_positions.empty()) {
        return collision_checker->getCollisionGrid(x);
    }

    const int n = static_cast<int>(joint_positions.size());

    for (const auto &pos : joint_positions) {
        Eigen::VectorXd p(3);
        p << pos(0), pos(1), pos(2);
        if (collision_checker->getCollisionGrid(p)) {
            return true;
        }
    }

    Eigen::Vector3d prev(0.0, 0.0, 0.0);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d curr(joint_positions[i](0), joint_positions[i](1), joint_positions[i](2));
        if (collision_checker->checkSegmentCollision(prev, curr)) {
            return true;
        }
        prev = curr;
    }

    return false;
}

Eigen::VectorXd MPPI3D::rk4Step(const Eigen::VectorXd &x, const Eigen::VectorXd &u) {
    Eigen::VectorXd k1 = f(x, u);
    Eigen::VectorXd k2 = f(x + 0.5 * dt * k1, u);
    Eigen::VectorXd k3 = f(x + 0.5 * dt * k2, u);
    Eigen::VectorXd k4 = f(x + dt * k3, u);
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

void MPPI3D::move() {
    x_init = rk4Step(x_init, u0);
    U_0.leftCols(T-1) = Uo.rightCols(T-1);
}

void MPPI3D::solve() {
    start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd Ui = U_0.replicate(N, 1);
    Eigen::VectorXd costs(N);
    Eigen::VectorXd weights(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Eigen::MatrixXd Xi(dim_x, T+1);
        Eigen::MatrixXd noise = getNoise(T);
        Ui.middleRows(i * dim_u, dim_u) += noise;
        h(Ui.middleRows(i * dim_u, dim_u));

        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < T; ++j) {
            cost += p(Xi.col(j), x_target);
            // cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
            Xi.col(j+1) = rk4Step(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
        }

        cost += p(Xi.col(T), x_target);
        for (int j = 1; j < T+1; ++j) {
            if (checkCollision(Xi.col(j))) {
                cost = 1e8;
                break;
            }
        }
        costs(i) = cost;
    }

    double min_cost = costs.minCoeff();
    weights = (-gamma_u * (costs.array() - min_cost)).exp();
    double total_weight =  weights.sum();
    weights /= total_weight;

    Uo = Eigen::MatrixXd::Zero(dim_u, T);
    for (int i = 0; i < N; ++i) {
        Uo += Ui.middleRows(i * dim_u, dim_u) * weights(i);
    }
    h(Uo);

    finish = std::chrono::high_resolution_clock::now();
    elapsed_1 = finish - start;
    elapsed = elapsed_1.count();

    u0 = Uo.col(0);

    Xo.col(0) = x_init;
    for (int j = 0; j < T; ++j) {
        Xo.col(j+1) = rk4Step(Xo.col(j), Uo.col(j));
    }

    visual_traj.push_back(x_init);
}

void MPPI3D::show() {}
void MPPI3D::showTraj() {}
