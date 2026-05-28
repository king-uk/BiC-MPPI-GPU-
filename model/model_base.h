#pragma once

#include <Eigen/Dense>
#include <vector>

// Maximum dimensions for GPU kernels (to avoid Variable Length Arrays)
#define GPU_MAX_DIM_X 14
#define GPU_MAX_DIM_U 7
#define GPU_MAX_T 200
#define GPU_MAX_NS 1024

class ModelBase {
public:
    ModelBase();
    ~ModelBase();

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

    // 가상 함수 추가: 매니퓰레이터가 아닌 모델은 빈 벡터를 반환하도록 기본 구현
    virtual std::vector<Eigen::Vector3d> getAllJointPositions(const Eigen::VectorXd& x) const {
        return {}; 
    }
};

inline ModelBase::ModelBase() {
};

inline ModelBase::~ModelBase() {
};



