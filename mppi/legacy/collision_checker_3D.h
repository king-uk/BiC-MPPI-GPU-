#pragma once

#include "model_base.h"

#include <eigen3/Eigen/Dense>

#include <array>
#include <fstream>
#include <vector>

class CollisionChecker {
public:
  CollisionChecker();
  ~CollisionChecker();

  //   void loadMap(const std::string &file_path, double resolution);
  //   void set3D(double max_hei);

  // x, y, z, r
  std::vector<std::array<double, 4>> circles;

  // x, y, z, w, h, d
  std::vector<std::array<double, 6>> rectangles;

  void addCircle(double x, double y, double z, double r);
  void addRectangle(double x, double y, double z, double w, double h, double d);

  bool getCollisionGrid(const Eigen::VectorXd &x);
  bool getCollisionCircle(const Eigen::VectorXd &z);
  bool getCollisionGrid_polygon(const Eigen::VectorXd &x);
  bool getCollisionCircle_polygon(const Eigen::VectorXd &z);
  bool getCollisionGrid_map(const Eigen::VectorXd &x);
  bool getCollisionCircle_map(const Eigen::VectorXd &z);

  bool check_collision_circle(const Eigen::VectorXd &i);
  bool check_collision_rectangle(const Eigen::VectorXd &j);

  // [새로 추가] 선분-AABB 충돌 체크 (Slab method)
  // p0, p1: 선분의 시작/끝 좌표 (3D)
  // 멈브, 링크, 세그먼트 전체에 대해 통과 여부를 판단
  bool checkSegmentCollision(const Eigen::Vector3d &p0,
                             const Eigen::Vector3d &p1);

  std::vector<std::vector<std::vector<double>>> map; // 3차원

private:
  bool is_3d;
  int max_hei;
  bool with_map;
  double resolution;
  int max_row;
  int max_col;
};

CollisionChecker::CollisionChecker() {
  map.clear();
  circles.clear();
  rectangles.clear();
  with_map = false;
  is_3d = true;
}

CollisionChecker::~CollisionChecker() {}

// void CollisionChecker::loadMap(const std::string &file_path,
//                                double resolution) {
//   map.clear();
//   std::ifstream file(file_path);
//   std::string line;

//   while (std::getline(file, line)) {
//     std::vector<double> row;
//     std::string::size_type sz = 0;

//     if (!file.is_open()) {
//       throw std::runtime_error("Error opening file: " + file_path);
//     }

//     while (sz < line.length()) {
//       double value = std::stod(line, &sz);
//       row.push_back(value);
//       line = line.substr(sz);
//     }

//     map.push_back(row);
//   }

//   file.close();

//   with_map = true;
//   this->resolution = resolution;
//   max_row = map.size();
//   max_col = map[0].size();
// }

// void CollisionChecker::set3D(double max_hei) {
//   this->is_3d = true;
//   this->max_hei = max_hei / resolution;
// }

void CollisionChecker::addCircle(double x, double y, double z, double r) {
  circles.push_back({x, y, z, r});
}

void CollisionChecker::addRectangle(double x, double y, double z, double w,
                                    double h, double d) {
  rectangles.push_back({x, x + w, y, y + h, z, z + d});
}

bool CollisionChecker::getCollisionGrid(const Eigen::VectorXd &x) {
  if (with_map) {
    return getCollisionGrid_map(x);
  } else {
    return getCollisionGrid_polygon(x);
  }
}

// bool CollisionChecker::getCollisionCircle(const Eigen::VectorXd &x) {
//   if (with_map) {
//     return getCollisionCircle_map(x);
//   } else {
//     return getCollisionCircle_polygon(x);
//   }
// }

bool CollisionChecker::getCollisionGrid_polygon(
    const Eigen::VectorXd &x) { // VectorXd<VectorXd> 가 입력으로 들어온다

  // Circle
  if (check_collision_circle(x)) {
    return true;
  }
  // Rectangle
  if (check_collision_rectangle(x)) {
    return true;
  }
  return false;
}

bool CollisionChecker::getCollisionCircle_polygon(const Eigen::VectorXd &z) {
  // Circle
  if (check_collision_circle(z)) {
    return true;
  }
  // Rectangle
  if (check_collision_rectangle(z)) {
    return true;
  }
  return false;
}

bool CollisionChecker::check_collision_circle(
    const Eigen::VectorXd &i) { // i=x,y,z
  double dx, dy, dz, distance_2, dc;
  for (int j = 0; j < circles.size(); ++j) {
    dx = circles[j][0] - i(0);
    dy = circles[j][1] - i(1);
    dz = circles[j][2] - i(2);
    distance_2 = (dx * dx) + (dy * dy) + (dz * dz);
    dc = circles[j][3];
    if (distance_2 <= (dc * dc)) {
      return true;
    }
  }
  return false;
}

bool CollisionChecker::check_collision_rectangle(
    const Eigen::VectorXd &j) { // j=x,y,z,
  for (int i = 0; i < rectangles.size(); ++i) {
    if ((j(0)) < rectangles[i][0]) {
      continue;
    } else if (rectangles[i][1] < (j(0))) {
      continue;
    } else if ((j(1)) < rectangles[i][2]) {
      continue;
    } else if (rectangles[i][3] < (j(1))) {
      continue;
    } else if ((j(2)) < rectangles[i][4]) {
      continue;
    } else if (rectangles[i][5] < (j(2))) {
      continue;
    } else {
      return true;
    }
  }
  return false;
}

// ============================================================
// 선분-AABB 충돌 체크: Slab Method (분리축 정리)
// p0 → p1 선분이 저장된 직육면체(AABB) 중 하나라도 교차하면 true
// ============================================================
bool CollisionChecker::checkSegmentCollision(const Eigen::Vector3d &p0,
                                             const Eigen::Vector3d &p1) {
  for (const auto &rect : rectangles) {
    // rect = {xmin, xmax, ymin, ymax, zmin, zmax}
    double bmin[3] = {rect[0], rect[2], rect[4]};
    double bmax[3] = {rect[1], rect[3], rect[5]};

    // 로봇 충돌 체크 시 micro-margin 포함 (회전 중 얼마리)
    const double margin = 0.03; // 3cm margin
    for (int k = 0; k < 3; ++k) {
      bmin[k] -= margin;
      bmax[k] += margin;
    }

    // Slab method: t_enter, t_exit 계산
    double t_enter = 0.0, t_exit = 1.0;
    bool hit = true;

    for (int k = 0; k < 3; ++k) {
      double dk = p1(k) - p0(k); // 방향 벡터 성분
      if (std::abs(dk) < 1e-10) {
        // 해당 축에 평행 → 해당 축 slab 밖이면 no-hit
        if (p0(k) < bmin[k] || p0(k) > bmax[k]) {
          hit = false;
          break;
        }
      } else {
        double t1 = (bmin[k] - p0(k)) / dk;
        double t2 = (bmax[k] - p0(k)) / dk;
        if (t1 > t2)
          std::swap(t1, t2);
        t_enter = std::max(t_enter, t1);
        t_exit = std::min(t_exit, t2);
        if (t_enter > t_exit) {
          hit = false;
          break;
        }
      }
    }
    if (hit)
      return true;
  }
  return false;
}

bool CollisionChecker::getCollisionGrid_map(const Eigen::VectorXd &x) {
  // Need to check comparison double
  int nx = round(x(0) / resolution);
  int ny = round(x(1) / resolution);
  int nz = round(x(2) / resolution);
  if (nx < 0 || max_row <= nx) {
    return true;
  }
  if (ny < 0 || max_col <= ny) {
    return true;
  }
  if (nz < 0 || max_hei <= nz) {
    return true;
  }
  if (map[nx][ny][nz] == 10) {
    return true;
  }
  return false;
}

bool CollisionChecker::getCollisionCircle_map(const Eigen::VectorXd &z) {
  // Need to check comparison double
  int cx = round(z(0) / resolution);
  int cy = round(z(1) / resolution);
  int cz = round(z(2) / resolution);
  double r = z(3) / resolution + 0.5;
  if (cx < 0 || max_row <= cx) {
    return true;
  }
  if (cy < 0 || max_col <= cy) {
    return true;
  }
  if (cz < 0 || max_hei <= cz) {
    return true;
  }
  if (map[cx][cy][cz] == 10) {
    return true;
  }
  if (map[cx][cy][cz] == 5) {
    return false;
  }
  if (map[cx][cy][cz] < r) {
    return true;
  }
  return false;
}