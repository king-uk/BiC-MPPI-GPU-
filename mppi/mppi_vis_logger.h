#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

// ============================================================
// MPPIVisLogger — Generic trajectory data logger for MPPI
//
// Saves per-step rollout / cluster / optimal trajectory data
// as binary files. A companion Python script reads these and
// generates animated GIF visualizations.
//
// Binary format per file:
//   int32 num_trajectories
//   int32 dim  (rows, typically dim_x)
//   int32 len  (columns, typically T+1)
//   double[num_trajectories * dim * len]  (row-major per traj)
//
// Usage:
//   MPPIVisLogger logger;
//   logger.enabled = true;
//   logger.init("mppi", 3, 100, 0.1, cc->map, x_init, x_target);
//   // in solve loop:
//   logger.beginStep(i);
//   logger.saveTrajectory("optimal", Xo);
//   logger.saveTrajectories("rollouts", sample_trajs);
// ============================================================
class MPPIVisLogger {
public:
  bool enabled = false;

  void init(const std::string &solver_name, int dim_x, int T,
            double map_resolution,
            const std::vector<std::vector<double>> &map_data,
            const Eigen::VectorXd &x_init, const Eigen::VectorXd &x_target) {
    if (!enabled)
      return;
    this->base_dir = "vis_data/" + solver_name;
    this->dim_x = dim_x;
    this->T = T;

    mkdirp("vis_data");
    mkdirp(base_dir);

    // Save metadata
    std::ofstream meta(base_dir + "/meta.txt");
    meta << "solver=" << solver_name << "\n";
    meta << "dim_x=" << dim_x << "\n";
    meta << "T=" << T << "\n";
    meta << "map_resolution=" << map_resolution << "\n";
    int rows = (int)map_data.size();
    int cols = rows > 0 ? (int)map_data[0].size() : 0;
    meta << "map_rows=" << rows << "\n";
    meta << "map_cols=" << cols << "\n";
    // Write init/target as space-separated values
    meta << "x_init=";
    for (int i = 0; i < x_init.size(); ++i)
      meta << (i ? " " : "") << x_init(i);
    meta << "\n";
    meta << "x_target=";
    for (int i = 0; i < x_target.size(); ++i)
      meta << (i ? " " : "") << x_target(i);
    meta << "\n";
    meta.close();

    // Save map as binary: rows x cols doubles
    {
      std::ofstream f(base_dir + "/map.bin", std::ios::binary);
      int32_t r = rows, c = cols;
      f.write((const char *)&r, 4);
      f.write((const char *)&c, 4);
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
          double v = map_data[i][j];
          f.write((const char *)&v, 8);
        }
    }
    std::cout << "[VisLogger] Initialized: " << base_dir << std::endl;
  }

  void beginStep(int step) {
    if (!enabled)
      return;
    current_step = step;
    mkdirp(stepDir());
  }

  // Save a single trajectory (dim_x x cols)
  void saveTrajectory(const std::string &label, const Eigen::MatrixXd &traj) {
    if (!enabled)
      return;
    std::vector<Eigen::MatrixXd> v = {traj};
    saveBin(stepDir() + "/" + label + ".bin", v);
  }

  // Save multiple trajectories (each dim_x x cols, can differ in cols)
  void saveTrajectories(const std::string &label,
                        const std::vector<Eigen::MatrixXd> &trajs) {
    if (!enabled || trajs.empty())
      return;
    saveBin(stepDir() + "/" + label + ".bin", trajs);
  }

  // Save current position
  void savePosition(const Eigen::VectorXd &pos) {
    if (!enabled)
      return;
    std::ofstream f(stepDir() + "/x_pos.bin", std::ios::binary);
    int32_t n = 1, r = (int32_t)pos.size(), c = 1;
    f.write((const char *)&n, 4);
    f.write((const char *)&r, 4);
    f.write((const char *)&c, 4);
    for (int i = 0; i < pos.size(); ++i) {
      double v = pos(i);
      f.write((const char *)&v, 8);
    }
  }

  // Save cost values for rollout samples (for coloring)
  void saveCosts(const std::string &label, const std::vector<double> &costs) {
    if (!enabled)
      return;
    std::ofstream f(stepDir() + "/" + label + "_costs.bin", std::ios::binary);
    int32_t n = (int32_t)costs.size();
    f.write((const char *)&n, 4);
    for (double c : costs)
      f.write((const char *)&c, 8);
  }

private:
  std::string base_dir;
  int current_step = 0;
  int dim_x = 0;
  int T = 0;

  std::string pad(int n) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%04d", n);
    return std::string(buf);
  }

  std::string stepDir() { return base_dir + "/step_" + pad(current_step); }

  static void mkdirp(const std::string &path) {
    mkdir(path.c_str(), 0755);
  }

  void saveBin(const std::string &path,
               const std::vector<Eigen::MatrixXd> &trajs) {
    std::ofstream f(path, std::ios::binary);
    int32_t n = (int32_t)trajs.size();
    int32_t rows = n > 0 ? (int32_t)trajs[0].rows() : 0;
    f.write((const char *)&n, 4);
    f.write((const char *)&rows, 4);
    for (const auto &m : trajs) {
      int32_t cols = (int32_t)m.cols();
      f.write((const char *)&cols, 4);
      for (int r = 0; r < m.rows(); ++r)
        for (int c = 0; c < m.cols(); ++c) {
          double v = m(r, c);
          f.write((const char *)&v, 8);
        }
    }
  }
};
