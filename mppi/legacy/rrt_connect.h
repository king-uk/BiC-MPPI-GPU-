#pragma once

#include "collision_checker.h"
#include "model_base.h"
#include "mppi_param.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// ============================================================
// RRTConnect : bi_mppi.h 와 동일한 사용 패턴을 유지하는 솔버
//
//  - template constructor : BiMPPI 와 동일하게 ModelClass 를 받음
//  - init()                : RRTConnectParam 으로 파라미터 설정
//  - setCollisionChecker() : CollisionChecker 등록
//  - solve()               : 한 번의 플랜 실행 (전체 경로 생성)
//  - move()                : x_init 을 경로를 따라 한 스텝 전진
//  - x_init / x_target / Xo / Uo / elapsed : 공개 인터페이스
// ============================================================

class RRTConnect {
public:
  // ---- Node structure for the RRT tree ----
  struct Node {
    Eigen::VectorXd x; // 상태
    Eigen::VectorXd u; // 이 노드에 도달하기 위해 사용한 제어 입력
    int parent;        // 부모 노드 인덱스 (-1 이면 root)
    int steps_taken;   // 이 노드에 도달하기 위해 u를 적용한 횟수
  };

  // ---- Public constructor (same pattern as BiMPPI) ----
  template <typename ModelClass> RRTConnect(ModelClass model);
  ~RRTConnect();

  // ---- Interface (mirroring BiMPPI) ----
  void init(RRTConnectParam param);
  void setCollisionChecker(CollisionChecker *cc);
  void solve();
  void move();

  // ---- Public state (same names as BiMPPI for easy comparison) ----
  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::MatrixXd Xo; // 최적 상태 trajectory  (dim_x × steps+1)
  Eigen::MatrixXd Uo; // 최적 제어 trajectory  (dim_u × steps)

  double elapsed; // 마지막 solve() 에 걸린 시간 [초]

  std::vector<Eigen::VectorXd> visual_traj; // move() 호출마다 기록

  // 경로 도달 검사기
  bool isPathFinished() const {
    return path_x.empty() || path_index >= static_cast<int>(path_x.size()) - 1;
  }

  int getPathSize() const { return path_x.size(); }

private:
  // ---- Dynamics / cost from ModelBase ----
  int dim_x;
  int dim_u;
  std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;

  // ---- Parameters ----
  double dt;
  int max_iter;
  double step_size;
  double goal_tol;
  int control_steps; // 한 번 steer 할 때 적용하는 적분 스텝 수
  int num_controls;  // 무작위 제어 후보 수

  Eigen::VectorXd x_min, x_max;
  Eigen::VectorXd u_min, u_max;

  Eigen::VectorXd state_weights; // 거리 가중치
  std::vector<int> angle_idx;    // 각도 wrap 처리를 할 인덱스

  // ---- Collision checker ----
  CollisionChecker *collision_checker;

  // ---- Internal trees ----
  std::vector<Node> tree_f; // forward  tree (시작 → 목표 방향)

  // ---- Path tracking for move() ----
  std::vector<Eigen::VectorXd> path_x; // 경로 상태 (순서대로)
  std::vector<Eigen::VectorXd> path_u; // 경로 제어 입력
  int path_index;                      // 현재 실행 중인 경로 인덱스

  // ---- Random engine ----
  std::mt19937 rng;

  // ---- Helper functions ----
  Eigen::VectorXd sampleRandom();
  int nearestNeighbor(const std::vector<Node> &tree,
                      const Eigen::VectorXd &x_rand);
  // steer: near 에서 x_rand 방향으로 한 스텝 확장.
  // 성공하면 새 Node 를 반환하고 true, 충돌이면 false.
  bool steer(const std::vector<Node> &tree, int near_idx,
             const Eigen::VectorXd &x_rand, Node &new_node);



  // 단방향이므로 tree_b는 삭제됨. 추출 방식도 단방향에 맞춤.
  void extractPath(int last_idx);

  // 상태 공간 거리 (가중치 및 각도 보정)
  double stateDist(const Eigen::VectorXd &a, const Eigen::VectorXd &b);

  // 단일 RK4 적분 스텝
  Eigen::VectorXd integrate(const Eigen::VectorXd &x,
                            const Eigen::VectorXd &u) const;

  // 경로 충돌 여부
  bool pathCollision(const Eigen::VectorXd &x_from,
                     const Eigen::VectorXd &x_to) const;
};

// ============================================================
//  Template constructor
// ============================================================
template <typename ModelClass> RRTConnect::RRTConnect(ModelClass model) {
  dim_x = model.dim_x;
  dim_u = model.dim_u;
  f = model.f;

  rng.seed(static_cast<unsigned>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count()));

  collision_checker = nullptr;
  path_index = 0;
  elapsed = 0.0;
}

RRTConnect::~RRTConnect() {}

// ============================================================
//  init
// ============================================================
void RRTConnect::init(RRTConnectParam param) {
  dt = param.dt;
  x_init = param.x_init;
  x_target = param.x_target;
  max_iter = param.max_iter;
  step_size = param.step_size;
  goal_tol = param.goal_tol;
  x_min = param.x_min;
  x_max = param.x_max;
  u_min = param.u_min;
  u_max = param.u_max;
  control_steps = param.control_steps;
  num_controls = param.num_controls;
  state_weights = param.state_weights;
  angle_idx = param.angle_idx;

  path_x.clear();
  path_u.clear();
  path_index = 0;
}

// ============================================================
//  setCollisionChecker
// ============================================================
void RRTConnect::setCollisionChecker(CollisionChecker *cc) {
  collision_checker = cc;
}

// ============================================================
//  Helpers
// ============================================================
double RRTConnect::stateDist(const Eigen::VectorXd &a,
                             const Eigen::VectorXd &b) {
  Eigen::VectorXd diff = a - b;
  // 각도 인덱스에 대해 랩핑 수행
  for (int idx : angle_idx) {
    if (idx >= 0 && idx < diff.size()) {
      diff(idx) = std::atan2(std::sin(diff(idx)), std::cos(diff(idx)));
    }
  }
  // 가중치 적용
  if (state_weights.size() == a.size()) {
    diff = diff.cwiseProduct(state_weights);
  }
  return diff.norm();
}

Eigen::VectorXd RRTConnect::integrate(const Eigen::VectorXd &x,
                                      const Eigen::VectorXd &u) const {
  // RK4
  Eigen::VectorXd k1 = f(x, u);
  Eigen::VectorXd k2 = f(x + 0.5 * dt * k1, u);
  Eigen::VectorXd k3 = f(x + 0.5 * dt * k2, u);
  Eigen::VectorXd k4 = f(x + dt * k3, u);
  return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

Eigen::VectorXd RRTConnect::sampleRandom() {
  // 10% 확률로 목표 상태 직접 샘플 (goal bias)
  std::uniform_real_distribution<double> bias(0.0, 1.0);
  if (bias(rng) < 0.10) {
    return x_target;
  }
  Eigen::VectorXd x_rand(dim_x);
  for (int i = 0; i < dim_x; ++i) {
    std::uniform_real_distribution<double> dist(x_min(i), x_max(i));
    x_rand(i) = dist(rng);
  }
  return x_rand;
}

int RRTConnect::nearestNeighbor(const std::vector<Node> &tree,
                                const Eigen::VectorXd &x_rand) {
  int best_idx = 0;
  double best_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(tree.size()); ++i) {
    double d = stateDist(tree[i].x, x_rand);
    if (d < best_dist) {
      best_dist = d;
      best_idx = i;
    }
  }
  return best_idx;
}

bool RRTConnect::steer(const std::vector<Node> &tree, int near_idx,
                       const Eigen::VectorXd &x_rand, Node &new_node) {
  const Eigen::VectorXd &x_near = tree[near_idx].x;

  // --- 랜덤 제어 입력 후보 중 x_rand 에 가장 가깝게 도달하는 것 선택 ---
  double best_dist = std::numeric_limits<double>::max();
  Eigen::VectorXd best_x = x_near;
  Eigen::VectorXd best_u = Eigen::VectorXd::Zero(dim_u);
  int best_steps = 0; // 최고 효율이었던 적분 스텝 수

  std::uniform_real_distribution<double> zero_one(0.0, 1.0);

  for (int k = 0; k < num_controls; ++k) {
    // 랜덤 제어 입력 샘플
    Eigen::VectorXd u_rand(dim_u);
    for (int j = 0; j < dim_u; ++j) {
      std::uniform_real_distribution<double> ud(u_min(j), u_max(j));
      u_rand(j) = ud(rng);
    }

    // control_steps 번 적분
    Eigen::VectorXd x_cur = x_near;
    double min_dist_to_rand = std::numeric_limits<double>::max();
    Eigen::VectorXd best_x_for_this_u = x_near;
    int best_steps_for_this_u = 0;

    for (int s = 0; s < control_steps; ++s) {
      x_cur = integrate(x_cur, u_rand);
      if (collision_checker && collision_checker->getCollisionGrid(x_cur)) {
        break; // 충돌 시 적분 중단
      }

      // step_size 제약을 벗어나면 그 즉시 잘라냄 (선형 보간 X)
      if (stateDist(x_near, x_cur) > step_size) {
        break;
      }

      double d = stateDist(x_cur, x_rand);
      if (d < min_dist_to_rand) {
        min_dist_to_rand = d;
        best_x_for_this_u = x_cur;
        best_steps_for_this_u = s + 1; // 1번 적분 시 1 스텝
      }
    }

    // 초기 위치보다 이동했고 안전한 점검이 이뤄졌다면 후보로 등록
    if (min_dist_to_rand != std::numeric_limits<double>::max() &&
        min_dist_to_rand < best_dist) {
      if (stateDist(x_near, best_x_for_this_u) >
          1e-3) { // 너무 적은 이동(micro step) 무시
        best_dist = min_dist_to_rand;
        best_x = best_x_for_this_u;
        best_u = u_rand;
        best_steps = best_steps_for_this_u;
      }
    }
  }

  // 실제 이동하지 못했으면 실패
  if (best_dist == std::numeric_limits<double>::max()) {
    return false;
  }

  new_node.x = best_x;
  new_node.u = best_u;
  new_node.parent = near_idx;
  new_node.steps_taken = best_steps;
  return true;
}



// extractPath: 단방향 트리(tree_f)에서 촘촘한(Dense) 경로 추출
void RRTConnect::extractPath(int last_idx) {
  std::vector<int> fwd_indices;
  int cur = last_idx;
  while (cur != -1) {
    fwd_indices.push_back(cur);
    cur = tree_f[cur].parent;
  }
  std::reverse(fwd_indices.begin(), fwd_indices.end());

  path_x.clear();
  path_u.clear();

  if (fwd_indices.empty()) {
    return;
  }

  // 시작점 추가
  path_x.push_back(tree_f[fwd_indices[0]].x);

  for (size_t i = 1; i < fwd_indices.size(); ++i) {
    const Node &parent_node = tree_f[fwd_indices[i - 1]];
    const Node &child_node = tree_f[fwd_indices[i]];

    Eigen::VectorXd x_cur = parent_node.x;
    for (int s = 0; s < child_node.steps_taken; ++s) {
      x_cur = integrate(x_cur, child_node.u);
      path_x.push_back(x_cur);
      path_u.push_back(child_node.u); // Dense U
    }
  }

  path_index = 0;
}

// ============================================================
//  solve : RRT(단방향) 플래닝 실행
// ============================================================
void RRTConnect::solve() {
  auto t_start = std::chrono::high_resolution_clock::now();

  // 트리 초기화
  tree_f.clear();

  Node root_f;
  root_f.x = x_init;
  root_f.u = Eigen::VectorXd::Zero(dim_u);
  root_f.parent = -1;
  root_f.steps_taken = 0;
  tree_f.push_back(root_f);

  bool found = false;
  int last_f = 0;

  for (int iter = 0; iter < max_iter && !found; ++iter) {
    // --- 1. 랜덤 샘플 ---
    Eigen::VectorXd x_rand = sampleRandom();

    // --- 2. tree 확장 (steer) ---
    int near_f = nearestNeighbor(tree_f, x_rand);
    Node new_f;
    if (!steer(tree_f, near_f, x_rand, new_f)) {
      continue;
    }
    tree_f.push_back(new_f);
    last_f = static_cast<int>(tree_f.size()) - 1;

    if (stateDist(new_f.x, x_target) < goal_tol) {
      found = true;
      break;
    }
  }

  if (found) {
    extractPath(last_f);
  } else {
    // 경로 미발견 시 현재 위치 유지
    if (path_x.empty()) {
      path_x.push_back(x_init);
    }
    std::cerr << "[RRTConnect] Warning: path not found within max_iter="
              << max_iter << std::endl;
  }

  // Xo / Uo 구성
  int steps = static_cast<int>(path_x.size());
  Xo = Eigen::MatrixXd(dim_x, steps);
  for (int i = 0; i < steps; ++i) {
    Xo.col(i) = path_x[i];
  }
  int u_steps = static_cast<int>(path_u.size());
  if (u_steps > 0) {
    Uo = Eigen::MatrixXd(dim_u, u_steps);
    for (int i = 0; i < u_steps; ++i) {
      Uo.col(i) = path_u[i];
    }
  } else {
    Uo = Eigen::MatrixXd::Zero(dim_u, 1);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration<double>(t_end - t_start).count();

  // 매 solve 마다 경로를 갱신하므로 인덱스 리셋
  path_index = 0;

  visual_traj.push_back(x_init);
}

// ============================================================
//  move : x_init 을 경로 한 스텝 전진
// ============================================================
void RRTConnect::move() {
  if (path_x.empty()) {
    return;
  }

  int next = path_index + 1;
  if (next < static_cast<int>(path_x.size())) {
    x_init = path_x[next];
    path_index = next;
  } else {
    // 경로 끝에 도달 → 마지막 위치 유지 (x_target으로 무조건 점프하는 텔레포트
    // 버그 수정)
    x_init = path_x.back();
  }
  visual_traj.push_back(x_init);
}
