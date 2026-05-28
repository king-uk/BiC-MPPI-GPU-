#pragma once
#include "matplotlibcpp.h"

#include <EigenRand/EigenRand>

#include "mppi_param.h"
#include "collision_checker_3D.h"
#include "model_base.h"

#include <ctime>
#include <vector>
#include <deque>
#include <map>
#include <chrono>
#include <iostream>

#include <omp.h>

#include <cmath>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <atomic>

// ============================================================
// SVGDMPPI3D:
//   - svgd_mppi.h의 SVGDMPPI와 동일한 알고리즘
//   - collision_checker_3D.h 사용 (manipulator용)
//   - RK4 적분 사용 (forward rollout, guideMPPI, move)
//   - 3D full-body 충돌 검사 (joint positions + link segments)
// ============================================================

class SVGDMPPI3D {
public:
    template<typename ModelClass>
    SVGDMPPI3D(ModelClass model);
    ~SVGDMPPI3D();

    void init(SVGDMPPIParam svgd_mppi_param);

    void setCollisionChecker(CollisionChecker *collision_checker);
    Eigen::MatrixXd getNoise(const int &T);
    void move();

    void solve();
    void backwardRollout();
    void forwardRollout();

    void SVGD();

    void selectConnection();
    void concatenate();
    void guideMPPI();
    void partitioningControl();

    void savePathToCSV(const std::string& filename) const;

    void dbscan(std::vector<std::vector<int>> &clusters, const Eigen::MatrixXd &Di,
                const Eigen::VectorXd &costs, const int &N, const int &T);
    void calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters,
                    const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T);
    void show();
    void showTraj();

    double pathLengthXY(double eps_ignore = 0.0) const;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    std::chrono::duration<double> elapsed_1;
    std::chrono::duration<double> elapsed_2;
    std::chrono::duration<double> elapsed_3;
    std::chrono::duration<double> elapsed_4;
    double elapsed;
    std::vector<Eigen::VectorXd> visual_traj;

    Eigen::MatrixXd U_f0;
    Eigen::MatrixXd U_b0;

    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    Eigen::VectorXd dummy_u;

    Eigen::MatrixXd Uo;
    Eigen::MatrixXd Xo;
    Eigen::VectorXd u0;

private:
    int dim_x;
    int dim_u;

    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
    std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;
    // Joint positions for 3D collision checking
    std::function<std::vector<Eigen::Vector3d>(Eigen::VectorXd)> get_all_joint_positions;

    std::mt19937_64 urng{static_cast<std::uint_fast64_t>(std::time(nullptr))};
    Eigen::Rand::NormalGen<double> norm_gen{0.0, 1.0};

    float dt;
    int Tf;
    int Tb;

    int Nf;
    int Nb;

    int Ns;      // surrogate grad 샘플 수
    int istep;   // SVGD i-step iteration 수

    int Nr;

    double gamma_u;
    Eigen::MatrixXd sigma_u;

    double deviation_mu;
    double cost_mu;
    int minpts;
    double epsilon;
    double psi;

    CollisionChecker *collision_checker{nullptr};

    // 3D full-body collision check
    bool checkCollision(const Eigen::VectorXd &x);
    // RK4 한 스텝 적분 (forward용)
    Eigen::VectorXd rk4Step(const Eigen::VectorXd &x, const Eigen::VectorXd &u);

    std::vector<std::vector<int>> clusters_f;
    std::vector<int> full_cluster_f;
    Eigen::MatrixXd Uf;
    Eigen::MatrixXd Xf;

    std::vector<std::vector<int>> clusters_b;
    std::vector<int> full_cluster_b;
    Eigen::MatrixXd Ub;
    Eigen::MatrixXd Xb;

    std::vector<std::vector<int>> joints;
    std::vector<Eigen::MatrixXd> Xc;
    std::vector<Eigen::MatrixXd> Uc;

    std::vector<Eigen::MatrixXd> Ur;
    std::vector<double> Cr;
    std::vector<Eigen::MatrixXd> Xr;
};

// ============================================================
// Constructor
// ============================================================
template<typename ModelClass>
SVGDMPPI3D::SVGDMPPI3D(ModelClass model) {
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

SVGDMPPI3D::~SVGDMPPI3D() {}

// ============================================================
// init
// ============================================================
void SVGDMPPI3D::init(SVGDMPPIParam svgd_mppi_param) {
    this->dt = svgd_mppi_param.dt;
    this->Tf = svgd_mppi_param.Tf;
    this->Tb = svgd_mppi_param.Tb;
    this->x_init = svgd_mppi_param.x_init;
    this->x_target = svgd_mppi_param.x_target;

    this->Nf = svgd_mppi_param.Nf;
    this->Nb = svgd_mppi_param.Nb;

    this->Ns = svgd_mppi_param.Ns;
    this->istep = svgd_mppi_param.istep;

    this->Nr = svgd_mppi_param.Nr;
    this->gamma_u = svgd_mppi_param.gamma_u;
    this->sigma_u = svgd_mppi_param.sigma_u;
    this->deviation_mu = svgd_mppi_param.deviation_mu;
    this->cost_mu = svgd_mppi_param.cost_mu;
    this->minpts = svgd_mppi_param.minpts;
    this->epsilon = svgd_mppi_param.epsilon;
    this->psi = svgd_mppi_param.psi;

    full_cluster_f.resize(Nf);
    std::iota(full_cluster_f.begin(), full_cluster_f.end(), 0);
    full_cluster_b.resize(Nb);
    std::iota(full_cluster_b.begin(), full_cluster_b.end(), 0);

    u0 = Eigen::VectorXd::Zero(dim_u);
    dummy_u = Eigen::VectorXd::Zero(dim_u);
}

void SVGDMPPI3D::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

Eigen::MatrixXd SVGDMPPI3D::getNoise(const int &T) {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
}

// ============================================================
// 3D collision check: joint points + link segments
// ============================================================
bool SVGDMPPI3D::checkCollision(const Eigen::VectorXd &x) {
    auto joint_positions = get_all_joint_positions(x);
    if (joint_positions.empty()) {
        return collision_checker->getCollisionGrid(x);
    }
    const int n = static_cast<int>(joint_positions.size());

    for (const auto &pos : joint_positions) {
        Eigen::VectorXd pt(3);
        pt << pos(0), pos(1), pos(2);
        if (collision_checker->getCollisionGrid(pt)) return true;
    }

    Eigen::Vector3d prev(0.0, 0.0, 0.0);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d curr(joint_positions[i](0), joint_positions[i](1), joint_positions[i](2));
        if (collision_checker->checkSegmentCollision(prev, curr)) return true;
        prev = curr;
    }
    return false;
}

// ============================================================
// RK4 적분 (forward rollout 정확도 향상)
// ============================================================
Eigen::VectorXd SVGDMPPI3D::rk4Step(const Eigen::VectorXd &x, const Eigen::VectorXd &u) {
    Eigen::VectorXd k1 = f(x, u);
    Eigen::VectorXd k2 = f(x + 0.5 * dt * k1, u);
    Eigen::VectorXd k3 = f(x + 0.5 * dt * k2, u);
    Eigen::VectorXd k4 = f(x + dt * k3, u);
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// ============================================================
// move: RK4 적분으로 실제 로봇 이동
// ============================================================
void SVGDMPPI3D::move() {
    x_init = rk4Step(x_init, u0);
    U_f0.leftCols(U_f0.cols() - 1) = U_f0.rightCols(U_f0.cols() - 1);
}

// ============================================================
// solve
// ============================================================
void SVGDMPPI3D::solve() {
    omp_set_nested(1);

    start = std::chrono::high_resolution_clock::now();
    backwardRollout();
    forwardRollout();
    finish = std::chrono::high_resolution_clock::now();
    elapsed_1 = finish - start;

    start = std::chrono::high_resolution_clock::now();
    selectConnection();
    concatenate();
    finish = std::chrono::high_resolution_clock::now();
    elapsed_2 = finish - start;

    start = std::chrono::high_resolution_clock::now();
    guideMPPI();
    finish = std::chrono::high_resolution_clock::now();
    elapsed_3 = finish - start;

    partitioningControl();

    elapsed = elapsed_1.count() + elapsed_2.count() + elapsed_3.count();
    visual_traj.push_back(x_init);
}

// ============================================================
// backwardRollout: SVGD i-step + covariance-adapted MPPI refine
// backward는 Euler 역적분 (bi_mppi_3D.h 패턴 유지)
// ============================================================
void SVGDMPPI3D::backwardRollout() {
    Eigen::MatrixXd Ui = U_b0.replicate(Nb, 1);
    Eigen::MatrixXd Di(dim_u, Nb);
    Eigen::VectorXd costs(Nb);

    const Eigen::MatrixXd Sigma = sigma_u * sigma_u.transpose();
    const Eigen::LDLT<Eigen::MatrixXd> Sigma_inv(Sigma);

    std::vector<Eigen::MatrixXd> cov_move_particles(Nb, Eigen::MatrixXd::Zero(dim_u, dim_u));
    std::vector<double> cov_move_counts(Nb, 0.0);

    std::vector<std::uint64_t> seeds(Nb);
    for (int i = 0; i < Nb; ++i) seeds[i] = urng();

    // backward rollout cost (Euler 역적분)
    auto rollout_cost_backward = [&](const Eigen::MatrixXd& U) -> double {
        Eigen::VectorXd x = x_target;
        double c = 0.0;
        if (checkCollision(x)) return 1e8;
        for (int j = Tb - 1; j >= 0; --j) {
            const Eigen::VectorXd u_use = (j == Tb - 1) ? U.col(j) : U.col(j + 1);
            x = x - (dt * f(x, u_use));
            c += q(x, U.col(j)) * dt;
            if (checkCollision(x)) return 1e8;
        }
        c += p(x, x_init);
        return c;
    };

    // (0) 초기 Nb 샘플 생성
    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        std::mt19937_64 rng(seeds[i] ^ (0x9e3779b97f4a7c15ULL + (std::uint64_t)i));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);
        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);
        U.noalias() += sigma_u * normal.generate<Eigen::MatrixXd>(dim_u, Tb, rng);
        h(U);
    }

    // (1) SVGD i-step 이동 + 이동 공분산 누적
    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        std::mt19937_64 rng(seeds[i] ^ (0xD1B54A32D192ED03ULL + (std::uint64_t)i));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);
        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);

        Eigen::MatrixXd cov_acc = Eigen::MatrixXd::Zero(dim_u, dim_u);
        double cov_cnt = 0.0;

        for (int it = 0; it < istep; ++it) {
            std::vector<Eigen::MatrixXd> Vset;
            Vset.reserve(Ns);
            std::vector<double> Jset(Ns, 0.0);
            double Jmin = std::numeric_limits<double>::infinity();
            for (int s = 0; s < Ns; ++s) {
                Eigen::MatrixXd V = U + sigma_u * normal.generate<Eigen::MatrixXd>(dim_u, Tb, rng);
                h(V);
                double J = rollout_cost_backward(V);
                Vset.push_back(std::move(V));
                Jset[s] = J;
                if (J < Jmin) Jmin = J;
            }
            std::vector<double> w(Ns, 0.0);
            double sumw = 0.0;
            for (int s = 0; s < Ns; ++s) {
                double ws = std::exp(-cost_mu * (Jset[s] - Jmin));
                w[s] = ws; sumw += ws;
            }
            if (!(sumw > 0.0) || !std::isfinite(sumw)) { sumw = (double)Ns; std::fill(w.begin(), w.end(), 1.0); }
            Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(dim_u, Tb);
            for (int s = 0; s < Ns; ++s) {
                grad.noalias() += w[s] * Sigma_inv.solve(Vset[s] - U);
            }
            grad /= sumw;
            Eigen::MatrixXd dU = psi * grad;
            U.noalias() += dU;
            h(U);
            for (int t = 0; t < Tb; ++t) {
                Eigen::VectorXd du = dU.col(t);
                cov_acc.noalias() += du * du.transpose();
            }
            cov_cnt += (double)Tb;
        }
        cov_move_particles[i] = cov_acc;
        cov_move_counts[i] = cov_cnt;
    }

    // (2) SVGD 후 particle cost/Di 계산
    std::atomic_bool all_feasible(true);
    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);
        double J = rollout_cost_backward(U);
        if (J > 1e7) all_feasible.store(false, std::memory_order_relaxed);
        costs(i) = J;
        Di.col(i) = (U - U_b0).rowwise().mean();
    }

    // (3) clustering
    if (!all_feasible.load(std::memory_order_relaxed)) { dbscan(clusters_b, Di, costs, Nb, Tb); }
    else { clusters_b.clear(); }
    if (clusters_b.empty()) { clusters_b.push_back(full_cluster_b); }

    // (4) cluster mean
    calculateU(Ub, clusters_b, costs, Ui, Tb);

    // (5) cluster별 covariance-adapted MPPI refine
    for (int c = 0; c < (int)clusters_b.size(); ++c) {
        const auto& cl = clusters_b[c];
        int best_i = cl[0];
        double best_J = costs(best_i);
        for (int idx : cl) { if (costs(idx) < best_J) { best_J = costs(idx); best_i = idx; } }
        Eigen::MatrixXd U_nom = Ui.middleRows(best_i * dim_u, dim_u);

        Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(dim_u, dim_u);
        double wsum = 0.0;
        double cmin = std::numeric_limits<double>::infinity();
        for (int idx : cl) cmin = std::min(cmin, costs(idx));
        for (int idx : cl) {
            double wi = std::exp(-gamma_u * (costs(idx) - cmin));
            wsum += wi;
            double cnt = std::max(1.0, cov_move_counts[idx]);
            Cov.noalias() += wi * (cov_move_particles[idx] / cnt);
        }
        if (!(wsum > 0.0)) wsum = 1.0;
        Cov /= wsum;
        const double reg = 1e-9;
        Cov.diagonal().array() += reg;

        Eigen::LLT<Eigen::MatrixXd> llt(Cov);
        Eigen::MatrixXd L;
        if (llt.info() == Eigen::Success) {
            L = llt.matrixL();
        } else {
            Eigen::LLT<Eigen::MatrixXd> llt2(Sigma + reg * Eigen::MatrixXd::Identity(dim_u, dim_u));
            L = (llt2.info() == Eigen::Success) ? llt2.matrixL() : sigma_u;
        }

        const int K = std::max(1, Ns);
        std::vector<double> Jm(K, 0.0);
        std::vector<Eigen::MatrixXd> Um(K);
        std::mt19937_64 rng(urng() ^ (0x94D049BB133111EBULL + (std::uint64_t)c));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);
        double Jmin = std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k) {
            Eigen::MatrixXd noise = L * normal.generate<Eigen::MatrixXd>(dim_u, Tb, rng);
            Um[k] = U_nom + noise;
            h(Um[k]);
            double Jk = rollout_cost_backward(Um[k]);
            Jm[k] = Jk;
            if (Jk < Jmin) Jmin = Jk;
        }
        Eigen::VectorXd w(K);
        for (int k = 0; k < K; ++k) w(k) = std::exp(-gamma_u * (Jm[k] - Jmin));
        double wtot = w.sum();
        if (!(wtot > 0.0) || !std::isfinite(wtot)) { w.setOnes(); wtot = (double)K; }
        w /= wtot;
        Eigen::MatrixXd U_upd = Eigen::MatrixXd::Zero(dim_u, Tb);
        for (int k = 0; k < K; ++k) U_upd.noalias() += w(k) * Um[k];
        h(U_upd);
        Ub.middleRows(c * dim_u, dim_u) = U_upd;
    }

    // (6) Xb 재생성 (Euler 역적분)
    Xb.resize((int)clusters_b.size() * dim_x, Tb + 1);
    for (int i = 0; i < (int)clusters_b.size(); ++i) {
        Xb.block(i * dim_x, Tb, dim_x, 1) = x_target;
        for (int t = Tb - 1; t >= 0; --t) {
            if (t == Tb - 1) {
                Xb.block(i * dim_x, t, dim_x, 1) =
                    Xb.block(i * dim_x, t + 1, dim_x, 1) -
                    (dt * f(Xb.block(i * dim_x, t + 1, dim_x, 1), Ub.block(i * dim_u, t, dim_u, 1)));
            } else {
                Xb.block(i * dim_x, t, dim_x, 1) =
                    Xb.block(i * dim_x, t + 1, dim_x, 1) -
                    (dt * f(Xb.block(i * dim_x, t + 1, dim_x, 1), Ub.block(i * dim_u, t + 1, dim_u, 1)));
            }
        }
    }
}

// ============================================================
// forwardRollout: SVGD i-step + covariance-adapted MPPI refine
// forward는 RK4 적분 (manipulator 정확도 향상)
// ============================================================
void SVGDMPPI3D::forwardRollout() {
    Eigen::MatrixXd Ui = U_f0.replicate(Nf, 1);
    Eigen::MatrixXd Di(dim_u, Nf);
    Eigen::VectorXd costs(Nf);

    const Eigen::MatrixXd Sigma = sigma_u * sigma_u.transpose();
    const Eigen::LDLT<Eigen::MatrixXd> Sigma_inv(Sigma);

    std::vector<Eigen::MatrixXd> cov_move_particles(Nf, Eigen::MatrixXd::Zero(dim_u, dim_u));
    std::vector<double> cov_move_counts(Nf, 0.0);

    std::vector<std::uint64_t> seeds(Nf);
    for (int i = 0; i < Nf; ++i) seeds[i] = urng();

    // forward rollout cost (RK4 적분)
    auto rollout_cost_forward = [&](const Eigen::MatrixXd& U) -> double {
        Eigen::VectorXd x = x_init;
        double c = 0.0;
        if (checkCollision(x)) return 1e8;
        for (int j = 0; j < Tf; ++j) {
            c += q(x, U.col(j)) * dt;
            x = rk4Step(x, U.col(j));
            if (checkCollision(x)) return 1e8;
        }
        c += p(x, x_target);
        return c;
    };

    // (0) 초기 Nf 샘플 생성
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        std::mt19937_64 rng(seeds[i] ^ (0x9e3779b97f4a7c15ULL + (std::uint64_t)i));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);
        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);
        U.noalias() += sigma_u * normal.generate<Eigen::MatrixXd>(dim_u, Tf, rng);
        h(U);
    }

    // (1) SVGD i-step 이동 + 이동 공분산 누적
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        std::mt19937_64 rng(seeds[i] ^ (0xD1B54A32D192ED03ULL + (std::uint64_t)i));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);
        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);

        Eigen::MatrixXd cov_acc = Eigen::MatrixXd::Zero(dim_u, dim_u);
        double cov_cnt = 0.0;

        for (int it = 0; it < istep; ++it) {
            std::vector<Eigen::MatrixXd> Vset;
            Vset.reserve(Ns);
            std::vector<double> Jset(Ns, 0.0);
            double Jmin = std::numeric_limits<double>::infinity();
            for (int s = 0; s < Ns; ++s) {
                Eigen::MatrixXd V = U + sigma_u * normal.generate<Eigen::MatrixXd>(dim_u, Tf, rng);
                h(V);
                double J = rollout_cost_forward(V);
                Vset.push_back(std::move(V));
                Jset[s] = J;
                if (J < Jmin) Jmin = J;
            }
            std::vector<double> w(Ns, 0.0);
            double sumw = 0.0;
            for (int s = 0; s < Ns; ++s) {
                double ws = std::exp(-cost_mu * (Jset[s] - Jmin));
                w[s] = ws; sumw += ws;
            }
            if (!(sumw > 0.0) || !std::isfinite(sumw)) { sumw = (double)Ns; std::fill(w.begin(), w.end(), 1.0); }
            Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(dim_u, Tf);
            for (int s = 0; s < Ns; ++s) {
                grad.noalias() += w[s] * Sigma_inv.solve(Vset[s] - U);
            }
            grad /= sumw;
            Eigen::MatrixXd dU = psi * grad;
            U.noalias() += dU;
            h(U);
            for (int t = 0; t < Tf; ++t) {
                Eigen::VectorXd du = dU.col(t);
                cov_acc.noalias() += du * du.transpose();
            }
            cov_cnt += (double)Tf;
        }
        cov_move_particles[i] = cov_acc;
        cov_move_counts[i] = cov_cnt;
    }

    // (2) SVGD 후 particle cost/Di 계산
    std::atomic_bool all_feasible(true);
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);
        double J = rollout_cost_forward(U);
        if (J > 1e7) all_feasible.store(false, std::memory_order_relaxed);
        costs(i) = J;
        Di.col(i) = (U - U_f0).rowwise().mean();
    }

    // (3) clustering
    if (!all_feasible.load(std::memory_order_relaxed)) { dbscan(clusters_f, Di, costs, Nf, Tf); }
    else { clusters_f.clear(); }
    if (clusters_f.empty()) { clusters_f.push_back(full_cluster_f); }

    // (4) cluster mean
    calculateU(Uf, clusters_f, costs, Ui, Tf);

    // (5) cluster별 covariance-adapted MPPI refine
    for (int c = 0; c < (int)clusters_f.size(); ++c) {
        const auto& cl = clusters_f[c];
        int best_i = cl[0];
        double best_J = costs(best_i);
        for (int idx : cl) { if (costs(idx) < best_J) { best_J = costs(idx); best_i = idx; } }
        Eigen::MatrixXd U_nom = Ui.middleRows(best_i * dim_u, dim_u);

        Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(dim_u, dim_u);
        double wsum = 0.0;
        double cmin = std::numeric_limits<double>::infinity();
        for (int idx : cl) cmin = std::min(cmin, costs(idx));
        for (int idx : cl) {
            double wi = std::exp(-gamma_u * (costs(idx) - cmin));
            wsum += wi;
            double cnt = std::max(1.0, cov_move_counts[idx]);
            Cov.noalias() += wi * (cov_move_particles[idx] / cnt);
        }
        if (!(wsum > 0.0)) wsum = 1.0;
        Cov /= wsum;
        const double reg = 1e-9;
        Cov.diagonal().array() += reg;

        Eigen::LLT<Eigen::MatrixXd> llt(Cov);
        Eigen::MatrixXd L;
        if (llt.info() == Eigen::Success) {
            L = llt.matrixL();
        } else {
            Eigen::LLT<Eigen::MatrixXd> llt2(Sigma + reg * Eigen::MatrixXd::Identity(dim_u, dim_u));
            L = (llt2.info() == Eigen::Success) ? llt2.matrixL() : sigma_u;
        }

        const int K = std::max(1, Ns);
        std::vector<double> Jm(K, 0.0);
        std::vector<Eigen::MatrixXd> Um(K);
        std::mt19937_64 rng(urng() ^ (0x94D049BB133111EBULL + (std::uint64_t)c));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);
        double Jmin = std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k) {
            Eigen::MatrixXd noise = L * normal.generate<Eigen::MatrixXd>(dim_u, Tf, rng);
            Um[k] = U_nom + noise;
            h(Um[k]);
            double Jk = rollout_cost_forward(Um[k]);
            Jm[k] = Jk;
            if (Jk < Jmin) Jmin = Jk;
        }
        Eigen::VectorXd w(K);
        for (int k = 0; k < K; ++k) w(k) = std::exp(-gamma_u * (Jm[k] - Jmin));
        double wtot = w.sum();
        if (!(wtot > 0.0) || !std::isfinite(wtot)) { w.setOnes(); wtot = (double)K; }
        w /= wtot;
        Eigen::MatrixXd U_upd = Eigen::MatrixXd::Zero(dim_u, Tf);
        for (int k = 0; k < K; ++k) U_upd.noalias() += w(k) * Um[k];
        h(U_upd);
        Uf.middleRows(c * dim_u, dim_u) = U_upd;
    }

    // (6) Xf 재생성 (RK4 적분)
    Xf.resize((int)clusters_f.size() * dim_x, Tf + 1);
    for (int i = 0; i < (int)clusters_f.size(); ++i) {
        Xf.block(i * dim_x, 0, dim_x, 1) = x_init;
        for (int t = 0; t < Tf; ++t) {
            Xf.block(i * dim_x, t + 1, dim_x, 1) =
                rk4Step(Xf.block(i * dim_x, t, dim_x, 1), Uf.block(i * dim_u, t, dim_u, 1));
        }
    }
}

void SVGDMPPI3D::SVGD() {
    // forward/backwardRollout 내부에서 stein 형식 pipeline을 수행하도록 통합함
}

// ============================================================
// selectConnection / concatenate / guideMPPI / partitioningControl
// bi_mppi_3D.h 패턴과 동일
// ============================================================
void SVGDMPPI3D::selectConnection() {
    joints.clear();
    for (int cf = 0; cf < (int)clusters_f.size(); ++cf) {
        double min_norm = std::numeric_limits<double>::max();
        int cb = 0, df = 0, db = 0;
        for (int cb_ = 0; cb_ < (int)clusters_b.size(); ++cb_) {
            for (int df_ = 0; df_ <= Tf; ++df_) {
                for (int db_ = 0; db_ <= Tb; ++db_) {
                    double norm = (Xf.block(cf * dim_x, df_, dim_x, 1) -
                                   Xb.block(cb_ * dim_x, db_, dim_x, 1)).norm();
                    if (norm < min_norm) { min_norm = norm; cb = cb_; df = df_; db = db_; }
                }
            }
        }
        joints.push_back({cf, cb, df, db});
    }
}

void SVGDMPPI3D::concatenate() {
    Uc.clear(); Xc.clear();
    for (std::vector<int> joint : joints) {
        int cf = joint[0], cb = joint[1], df = joint[2], db = joint[3];
        Eigen::MatrixXd U(dim_u, std::max(Tf, df + (Tb - db)));
        Eigen::MatrixXd X(dim_x, std::max(Tf, df + (Tb - db)) + 1);
        if (df == 0) { X.leftCols(df + 1) = Xf.block(cf * dim_x, 0, dim_x, df + 1); }
        else { U.leftCols(df) = Uf.block(cf * dim_u, 0, dim_u, df); X.leftCols(df + 1) = Xf.block(cf * dim_x, 0, dim_x, df + 1); }
        if (db != Tb) {
            U.middleCols(df + 1, Tb - db - 1) = Ub.block(cb * dim_u, db + 1, dim_u, Tb - db - 1);
            X.middleCols(df + 2, Tb - db - 1) = Xb.block(cb * dim_x, db + 1, dim_x, Tb - db - 1);
        }
        if (df + (Tb - db) < Tf) {
            U.rightCols(Tf - (df + (Tb - db))).colwise() = dummy_u;
            X.rightCols(Tf - (df + (Tb - db))).colwise() = x_target;
        }
        Uc.push_back(U); Xc.push_back(X);
    }
}

void SVGDMPPI3D::guideMPPI() {
    Ur.clear(); Cr.clear(); Xr.clear();
    for (int r = 0; r < (int)joints.size(); ++r) {
        Eigen::MatrixXd Ui = Uc[r].replicate(Nr, 1);
        Eigen::MatrixXd X_ref = Xc[r];
        int Tr = Uc[r].cols();
        Eigen::VectorXd costs(Nr);
        Eigen::VectorXd weights(Nr);

        #pragma omp parallel for
        for (int i = 0; i < Nr; ++i) {
            Eigen::MatrixXd Xi(dim_x, Tr + 1);
            Eigen::MatrixXd noise = getNoise(Tr);
            Ui.middleRows(i * dim_u, dim_u) += noise;
            h(Ui.middleRows(i * dim_u, dim_u));

            Xi.col(0) = x_init;
            double cost = 0.0;
            for (int j = 0; j < Tr; ++j) {
                cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)) * dt;
                Xi.col(j + 1) = rk4Step(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
            }
            cost += p(Xi.col(Tr), x_target);
            cost += (Xi - X_ref).colwise().norm().sum();
            for (int j = 0; j < Tr + 1; ++j) {
                if (checkCollision(Xi.col(j))) { cost = 1e8; break; }
            }
            costs(i) = cost;
        }

        double min_cost = costs.minCoeff();
        weights = (-gamma_u * (costs.array() - min_cost)).exp();
        double total_weight = weights.sum();
        weights /= total_weight;

        Eigen::MatrixXd Ures = Eigen::MatrixXd::Zero(dim_u, Tr);
        for (int i = 0; i < Nr; ++i) Ures += Ui.middleRows(i * dim_u, dim_u) * weights(i);
        h(Ures);

        Eigen::MatrixXd Xi(dim_x, Tr + 1);
        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < Tr; ++j) {
            cost += q(Xi.col(j), Ures.col(j)) * dt;
            Xi.col(j + 1) = rk4Step(Xi.col(j), Ures.col(j));
        }
        cost += p(Xi.col(Tr), x_target);
        for (int j = 0; j < Tr + 1; ++j) {
            if (checkCollision(Xi.col(j))) { cost = 1e8; break; }
        }
        Ur.push_back(Ures); Cr.push_back(cost); Xr.push_back(Xi);
    }

    double min_cost = std::numeric_limits<double>::max();
    int index = 0;
    for (int r = 0; r < (int)joints.size(); ++r) {
        if (Cr[r] < min_cost) { min_cost = Cr[r]; index = r; }
    }
    Uo = Ur[index]; Xo = Xr[index]; u0 = Uo.col(0);
}

void SVGDMPPI3D::partitioningControl() {
    U_f0 = Uo.leftCols(Tf);
    U_b0 = Eigen::MatrixXd::Zero(dim_u, Tb);
}

// ============================================================
// dbscan / calculateU  (동일 로직)
// ============================================================
void SVGDMPPI3D::dbscan(std::vector<std::vector<int>> &clusters, const Eigen::MatrixXd &Di,
                         const Eigen::VectorXd &costs, const int &N, const int &T) {
    clusters.clear();
    std::vector<bool> core_points(N, false);
    std::map<int, std::vector<int>> core_tree;

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (costs(i) > 1E7) continue;
        for (int j = i + 1; j < N; ++j) {
            if (costs(j) > 1E7) continue;
            if (deviation_mu * (Di.col(i) - Di.col(j)).norm() < epsilon) {
                #pragma omp critical
                { core_tree[i].push_back(j); core_tree[j].push_back(i); }
            }
        }
    }
    for (int i = 0; i < N; ++i) { if (minpts < (int)core_tree[i].size()) core_points[i] = true; }

    std::vector<bool> visited(N, false);
    for (int i = 0; i < N; ++i) {
        if (!core_points[i] || visited[i]) continue;
        std::deque<int> branch;
        std::vector<int> cluster;
        branch.push_back(i); cluster.push_back(i); visited[i] = true;
        while (!branch.empty()) {
            int now = branch.front();
            for (int next : core_tree[now]) {
                if (visited[next]) continue;
                visited[next] = true;
                cluster.push_back(next);
                if (core_points[next]) branch.push_back(next);
            }
            branch.pop_front();
        }
        clusters.push_back(cluster);
    }
}

void SVGDMPPI3D::calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters,
                              const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T) {
    U = Eigen::MatrixXd::Zero(clusters.size() * dim_u, T);
    #pragma omp parallel for
    for (int index = 0; index < (int)clusters.size(); ++index) {
        int pts = clusters[index].size();
        std::vector<double> weights(pts);
        double min_cost = std::numeric_limits<double>::max();
        for (int k : clusters[index]) min_cost = std::min(min_cost, costs(k));
        for (size_t i = 0; i < (size_t)pts; ++i)
            weights[i] = std::exp(-gamma_u * (costs(clusters[index][i]) - min_cost));
        double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (size_t i = 0; i < (size_t)pts; ++i)
            U.middleRows(index * dim_u, dim_u) +=
                (weights[i] / total_weight) * Ui.middleRows(clusters[index][i] * dim_u, dim_u);
        h(U.middleRows(index * dim_u, dim_u));
    }
}

void SVGDMPPI3D::show() {}

void SVGDMPPI3D::showTraj() {}

inline double SVGDMPPI3D::pathLengthXY(double eps_ignore) const {
    double L = 0.0;
    const size_t n = visual_traj.size();
    if (n < 2 || visual_traj[0].rows() < 2) return 0.0;
    auto finite = [](double v) { return std::isfinite(v); };
    double px = visual_traj[0](0), py = visual_traj[0](1);
    for (size_t i = 1; i < n; ++i) {
        if (visual_traj[i].rows() < 2) continue;
        double cx = visual_traj[i](0), cy = visual_traj[i](1);
        if (!finite(px) || !finite(py) || !finite(cx) || !finite(cy)) { px = cx; py = cy; continue; }
        double d = std::hypot(cx - px, cy - py);
        if (d > eps_ignore) L += d;
        px = cx; py = cy;
    }
    return L;
}

inline void SVGDMPPI3D::savePathToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);
    for (const auto& pt : visual_traj) {
        if (pt.size() >= 2 && std::isfinite(pt(0)) && std::isfinite(pt(1)))
            file << pt(0) << "," << pt(1) << "\n";
    }
}
