#pragma once

#include "mppi_gpu.cuh"

// ============================================================
// LogMPPI_GPU — Log-MPPI solver with CUDA GPU rollout
//
// Inherits MPPI_GPU and overrides noise generation to use
// lognormal-scaled perturbations:
//   ε(d,t) = σ_d * η(d,t) * exp(ξ(d,t))
// where η ~ N(0,1), ξ ~ N(0,1) independently (lognormal factor).
//
// Interface mirrors LogMPPI (CPU). Replace:
//   #include <log_mppi.h>       →  #include <log_mppi_gpu.cuh>
//   using Solver = LogMPPI;     →  using Solver = LogMPPI_GPU;
// ============================================================
class LogMPPI_GPU : public MPPI_GPU {
public:
  template <typename ModelClass>
  LogMPPI_GPU(ModelClass model);
  ~LogMPPI_GPU();

  void solve() override;

protected:
  // Additional lognormal buffer: same shape as d_noise [N * dim_u * T]
  double *d_log_noise;  // raw N(0,1) used as log-normal exponent

  void generateLogNoise();  // fill d_noise (normal) and d_log_noise (lognormal)
};

// ---- Template constructor ----
template <typename ModelClass>
LogMPPI_GPU::LogMPPI_GPU(ModelClass model) : MPPI_GPU(model) {
  d_log_noise = nullptr;
}
