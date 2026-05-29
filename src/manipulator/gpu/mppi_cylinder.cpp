/**
 * mppi_cylinder.cpp — Manipulator GPU MPPI with 3D Cylinder Avoidance
 *
 * Scenario:
 *   - RB5 6-DOF manipulator (velocity control, dim_x=6, dim_u=6)
 *   - Start: q = [0,0,0,0,0,0]
 *   - Goal:  q = [pi/2, 0,0,0,0,0]  (rotate J1 by 90 deg)
 *   - Obstacle: vertical cylinder (cx,cy,r,z_min,z_max) read from txt
 *
 * Avoidance strategy:
 *   - GPU rollout: uses joint-space dynamics + p_manipulator terminal cost
 *   - CPU stage cost q(): FK → cylinder distance penalty for each joint
 *   - Trajectory saved per iteration to CSV for 3D visualization
 */

#include <mppi_gpu.cuh>
#include <manipulator.h>
#include <collision_checker.h>

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ── Cylinder definition ─────────────────────────────────────────
struct Cylinder {
  double cx, cy, r, z_min, z_max;
};

std::vector<Cylinder> loadCylinders(const std::string &path) {
  std::vector<Cylinder> cyls;
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "Warning: cannot open " << path << "\n";
    return cyls;
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    Cylinder c;
    if (ss >> c.cx >> c.cy >> c.r >> c.z_min >> c.z_max)
      cyls.push_back(c);
  }
  return cyls;
}

// ── Cylinder avoidance cost (CPU, using FK) ─────────────────────
// Returns penalty for a 3D point being near/inside cylinders
double cylinderCost(double px, double py, double pz,
                    const std::vector<Cylinder> &cyls) {
  double cost = 0.0;
  for (const auto &c : cyls) {
    // Only penalize within z range
    if (pz < c.z_min - 0.1 || pz > c.z_max + 0.1) continue;
    double dx = px - c.cx, dy = py - c.cy;
    double dist = std::sqrt(dx*dx + dy*dy);
    const double death_r = c.r + 0.04;   // hard collision zone
    const double warn_r  = c.r + 0.15;   // soft warning zone
    if (dist < death_r) {
      cost += 200000.0;
    } else if (dist < warn_r) {
      double excess = warn_r - dist;
      cost += 80000.0 * excess * excess;
    }
  }
  return cost;
}

int main(int argc, char **argv) {
  std::cout << "=== Manipulator GPU MPPI — 3D Cylinder Avoidance ===\n";

  // ── Load cylinders ──────────────────────────────────────────
  std::string cyl_file = "../obstacles/cylinder_test.txt";
  if (argc >= 2) cyl_file = argv[1];
  auto cyls = loadCylinders(cyl_file);
  std::cout << "Loaded " << cyls.size() << " cylinder(s) from " << cyl_file << "\n";
  for (size_t i = 0; i < cyls.size(); ++i)
    std::cout << "  Cyl " << i << ": center=(" << cyls[i].cx << ","
              << cyls[i].cy << ") r=" << cyls[i].r
              << " z=[" << cyls[i].z_min << "," << cyls[i].z_max << "]\n";

  // ── Build Manipulator with cylinder-aware stage cost ────────
  auto model = Manipulator();

  // Override stage cost to include cylinder avoidance via FK
  model.q = [&cyls, &model](const Eigen::VectorXd &x,
                              const Eigen::VectorXd &u) -> double {
    // 1) Control effort
    double ctrl = u.norm();

    // 2) Joint limit soft barrier
    double jlim = 0.0;
    for (int i = 0; i < model.dof; ++i) {
      double qr = (model.q_max(i) - model.q_min(i)) * 0.5;
      double qc = (model.q_max(i) + model.q_min(i)) * 0.5;
      double norm = (x(i) - qc) / (qr * 0.8);
      if (std::abs(norm) > 1.0)
        jlim += 5.0 * (std::abs(norm) - 1.0) * (std::abs(norm) - 1.0);
    }

    // 3) Cylinder avoidance via FK
    double obs = 0.0;
    if (!cyls.empty()) {
      auto T_fk = model.computeForwardKinematics(x);
      // Check each joint position and link midpoint
      for (int k = 0; k < model.dof; ++k) {
        Eigen::Vector3d p = T_fk[k].block<3,1>(0,3);
        obs += cylinderCost(p.x(), p.y(), p.z(), cyls);
        // Also check midpoint of link
        if (k > 0) {
          Eigen::Vector3d p_prev = T_fk[k-1].block<3,1>(0,3);
          Eigen::Vector3d mid = (p + p_prev) * 0.5;
          obs += cylinderCost(mid.x(), mid.y(), mid.z(), cyls);
        }
      }
    }

    return ctrl + jlim + obs;
  };

  // ── Collision checker (empty GPU collision, FK handles it) ──
  CollisionChecker collision_checker;
  for (const auto &c : cyls)
    collision_checker.addCylinder(c.cx, c.cy, c.r, c.z_min, c.z_max);

  // ── Solver setup ────────────────────────────────────────────
  using Solver = MPPI_GPU;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.T  = 80;
  param.x_init.resize(model.dim_x);
  param.x_init.setZero();
  param.x_target.resize(model.dim_x);
  param.x_target.setZero();
  param.x_target(0) = M_PI / 2.0;   // J1 = 90 deg

  param.N       = 5000;
  param.gamma_u = 0.1;

  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.3, 0.2, 0.2, 0.2, 0.2, 0.2;
  param.sigma_u = sigma_u.asDiagonal();

  int maxiter   = 300;
  double tol    = 0.15;

  Solver solver(model);
  solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
  solver.init(param);
  solver.setCollisionChecker(&collision_checker);

  // ── CSV output ──────────────────────────────────────────────
  // Columns: iter, q0..q5, ee_x, ee_y, ee_z, err, elapsed_ms
  std::ofstream csv("trajectory_manipulator_cylinder.csv");
  csv << "iter,q0,q1,q2,q3,q4,q5,ee_x,ee_y,ee_z,err,elapsed_ms\n";

  // Also save cylinder info as header comment in a sidecar file
  {
    std::ofstream meta("trajectory_manipulator_cylinder_meta.txt");
    meta << "# x_init: 0 0 0 0 0 0\n";
    meta << "# x_target: " << M_PI/2 << " 0 0 0 0 0\n";
    meta << "# cylinders:\n";
    for (const auto &c : cyls)
      meta << "# CYL " << c.cx <<" "<< c.cy <<" "<< c.r
           <<" "<< c.z_min <<" "<< c.z_max <<"\n";
  }

  bool   is_reached = false;
  double total_ms   = 0.0;
  double f_err      = 0.0;
  int    i          = 0;

  for (i = 0; i < maxiter; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    solver.solve();
    solver.move();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    total_ms += ms;

    // FK for EE position
    auto T_fk = model.computeForwardKinematics(solver.x_init);
    Eigen::Vector3d ee = T_fk[5].block<3,1>(0,3);

    f_err = (solver.x_init - param.x_target).norm();

    csv << i;
    for (int j = 0; j < model.dim_x; ++j) csv << "," << solver.x_init(j);
    csv << "," << ee.x() << "," << ee.y() << "," << ee.z()
        << "," << f_err << "," << ms << "\n";

    if (f_err < tol) { is_reached = true; break; }

    if (i % 30 == 0)
      std::cout << "Iter " << i
                << " | q0=" << solver.x_init(0)
                << " | EE=(" << ee.x() << "," << ee.y() << "," << ee.z() << ")"
                << " | err=" << f_err
                << " | " << ms << " ms\n";
  }

  csv.close();
  std::cout << "\n=== Result ===\n"
            << "Success:    " << is_reached << "\n"
            << "Iterations: " << i << "\n"
            << "Total time: " << total_ms << " ms\n"
            << "Final err:  " << f_err << "\n"
            << "CSV saved:  trajectory_manipulator_cylinder.csv\n";

  return 0;
}
