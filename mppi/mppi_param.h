#pragma once
#include <Eigen/Dense>

struct MPPIParam {
    float dt;
    int T;
    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    int N;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
};

struct SmoothMPPIParam{
    double dt;
    double lambda;
    Eigen::MatrixXd w;
};

struct BiMPPIParam {
    float dt;
    int Tf;
    int Tb;
    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    int Nf;
    int Nb;
    int Nr;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
    double deviation_mu;
    double cost_mu;
    int minpts;
    double epsilon;
    double psi;
};

struct SVGDMPPIParam {
    float dt;
    int Tf;
    int Tb;
    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    int Nf;           // forward particle count
    int Nb;           // backward particle count
    int Ns;           // surrogate sample count per SVGD step
    int istep;        // number of SVGD inner iterations
    int Nr;           // guide MPPI sample count
    double gamma_u;
    Eigen::MatrixXd sigma_u;
    double deviation_mu;
    double cost_mu;   // softmax temperature for SVGD weights
    int minpts;
    double epsilon;
    double psi;       // SVGD step size
};

struct RRTConnectParam {
    double dt;                // 적분 시간 간격 (동역학 전파용)
    Eigen::VectorXd x_init;  // 시작 상태
    Eigen::VectorXd x_target;// 목표 상태
    int max_iter;             // 최대 반복 횟수
    double step_size;         // 하나의 RRT 확장 스텝 크기 (상태 공간 거리)
    double goal_tol;          // 목표 도달 허용 오차
    Eigen::VectorXd x_min;   // 상태 공간 하한
    Eigen::VectorXd x_max;   // 상태 공간 상한
    Eigen::VectorXd u_min;   // 제어 입력 하한
    Eigen::VectorXd u_max;   // 제어 입력 상한
    int control_steps;        // 확장 시 적용하는 제어 스텝 수
    int num_controls;         // 랜덤 제어 후보 수
    Eigen::VectorXd state_weights; // 상태 거리 계산 시 사용할 가중치
    std::vector<int> angle_idx;   // 각도(wrapping) 처리가 필요한 상태 인덱스
};