#include "bi_mppi_gpu.cuh"
#include "mppi_gpu.cuh"   // rollout_kernel 접근을 위해 포함
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>


// Local rollout kernel (ODR-safe: unique name per TU)
DEFINE_FORWARD_ROLLOUT_KERNEL(bi_rollout_kernel)

// backward_rollout_kernel과 guide_rollout_kernel만 정의 (forward는 매크로 인스턴스화 사용)


// ── Backward rollout kernel ─────────────────────────────────────
__global__ void backward_rollout_kernel(
    const double* __restrict__ d_Ub0,
    double* d_Ubi,
    const double* __restrict__ d_noise,
    const double* __restrict__ d_sigma,
    const double* __restrict__ d_x_init,
    const double* __restrict__ d_x_target,
    double* d_costs, double* d_Di,
    bool with_map, const double* d_map,
    int max_row, int max_col, double res,
    const double* d_circles, int n_circ,
    const double* d_rects,   int n_rect,
    int N, int dim_u, int dim_x, int T, double dt, double gamma_u,
    int model_type)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double* Ui_i = d_Ubi + i * (dim_u * T);

    for (int d = 0; d < dim_u; ++d)
        for (int t = 0; t < T; ++t) {
            double raw = d_Ub0[d*T+t] + d_sigma[d]*d_noise[i*(dim_u*T)+d*T+t];
            Ui_i[d*T+t] = raw;
        }
    if (model_type == 0) {
        h_wmrobot(Ui_i, T);
    } else if (model_type == 1) {
        h_quadrotor(Ui_i, T);
    } else if (model_type == 2) {
        h_velo(Ui_i, T);
    } else if (model_type == 3) {
        h_manipulator(Ui_i, dim_u, T);
    }

    double* Di_i = d_Di + i * dim_u;
    for (int d = 0; d < dim_u; ++d) {
        double acc = 0.0;
        for (int t = 0; t < T; ++t) acc += Ui_i[d*T+t] - d_Ub0[d*T+t];
        Di_i[d] = acc / T;
    }

    double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd_[GPU_MAX_DIM_X];
    for (int d = 0; d < dim_x; ++d) x[d] = d_x_target[d];

    double cost = 0.0;
    bool hit = false;

    for (int j = T-1; j >= 0; --j) {
        if (model_type == 0) {
            double v, omega;
            if (j == T-1) { v = Ui_i[0*T+j]; omega = Ui_i[1*T+j]; }
            else           { v = Ui_i[0*T+j+1]; omega = Ui_i[1*T+j+1]; }

            cost += p_wmrobot(x, d_x_init, dim_x);
            f_wmrobot(x, v, omega, xd_);
        } else if (model_type == 1) {
            double u_val[3];
            if (j == T-1) {
                u_val[0] = Ui_i[0*T+j]; u_val[1] = Ui_i[1*T+j]; u_val[2] = Ui_i[2*T+j];
            } else {
                u_val[0] = Ui_i[0*T+j+1]; u_val[1] = Ui_i[1*T+j+1]; u_val[2] = Ui_i[2*T+j+1];
            }

            cost += p_quadrotor(x, d_x_init, dim_x);
            f_quadrotor(x, u_val, xd_);
        } else if (model_type == 2) {
            double u_val[2];
            if (j == T-1) {
                u_val[0] = Ui_i[0*T+j]; u_val[1] = Ui_i[1*T+j];
            } else {
                u_val[0] = Ui_i[0*T+j+1]; u_val[1] = Ui_i[1*T+j+1];
            }

            cost += p_velo(x, d_x_init, dim_x);
            f_velo(x, u_val, xd_);
        } else if (model_type == 3) {
            double u_manip[GPU_MAX_DIM_U];
            for (int _d = 0; _d < dim_u; ++_d) {
                if (j == T-1) u_manip[_d] = Ui_i[_d*T+j];
                else          u_manip[_d] = Ui_i[_d*T+j+1];
            }
            cost += p_manipulator(x, d_x_init, dim_x);
            f_manipulator(x, u_manip, xd_, dim_u);
        }
        for (int d = 0; d < dim_x; ++d) xn[d] = x[d] - dt * xd_[d];

        if (!hit && check_collision(x, with_map, d_map, max_row, max_col, res,
                                    d_circles, n_circ, d_rects, n_rect)) {
            hit = true; cost = 1e8;
        }
        for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
    }
    if (!hit) {
        if (model_type == 0) {
            cost += p_wmrobot(x, d_x_init, dim_x);
        } else if (model_type == 1) {
            cost += p_quadrotor(x, d_x_init, dim_x);
        } else if (model_type == 2) {
            cost += p_velo(x, d_x_init, dim_x);
        } else if (model_type == 3) {
            cost += p_manipulator(x, d_x_init, dim_x);
        }
        if (check_collision(x, with_map, d_map, max_row, max_col, res,
                            d_circles, n_circ, d_rects, n_rect))
            cost = 1e8;
    }
    d_costs[i] = cost;
}

// ── Guide rollout kernel ────────────────────────────────────────
__global__ void guide_rollout_kernel(
    const double* __restrict__ d_Ur0,
    double* d_Uri,
    const double* __restrict__ d_noise,
    const double* __restrict__ d_sigma,
    const double* __restrict__ d_x_init,
    const double* __restrict__ d_x_target,
    const double* __restrict__ d_Xref,   // dim_x x (Tr+1)
    double* d_costs,
    bool with_map, const double* d_map,
    int max_row, int max_col, double res,
    const double* d_circles, int n_circ,
    const double* d_rects,   int n_rect,
    int N, int dim_u, int dim_x, int Tr, double dt, double gamma_u,
    int model_type)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double* Ui_i = d_Uri + i * (dim_u * Tr);
    for (int d = 0; d < dim_u; ++d)
        for (int t = 0; t < Tr; ++t) {
            Ui_i[d*Tr+t] = d_Ur0[d*Tr+t] +
                            d_sigma[d]*d_noise[i*(dim_u*Tr)+d*Tr+t];
        }
    if (model_type == 0) {
        h_wmrobot(Ui_i, Tr);
    } else if (model_type == 1) {
        h_quadrotor(Ui_i, Tr);
    } else if (model_type == 2) {
        h_velo(Ui_i, Tr);
    } else if (model_type == 3) {
        h_manipulator(Ui_i, dim_u, Tr);
    }

    double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd_[GPU_MAX_DIM_X];
    for (int d = 0; d < dim_x; ++d) x[d] = d_x_init[d];

    double cost = 0.0;
    bool hit = false;

    for (int t = 0; t < Tr; ++t) {
        if (model_type == 0) {
            double v = Ui_i[0*Tr+t], omega = Ui_i[1*Tr+t];
            cost += p_wmrobot(x, d_x_target, dim_x);
            // guide cost: distance to reference trajectory
            double gc = 0.0;
            for (int d = 0; d < dim_x; ++d) {
                double diff = x[d] - d_Xref[d*(Tr+1)+t];
                gc += diff*diff;
            }
            cost += sqrt(gc);

            f_wmrobot(x, v, omega, xd_);
        } else if (model_type == 1) {
            double u_val[3] = { Ui_i[0*Tr+t], Ui_i[1*Tr+t], Ui_i[2*Tr+t] };
            cost += p_quadrotor(x, d_x_target, dim_x);
            // guide cost: distance to reference trajectory
            double gc = 0.0;
            for (int d = 0; d < dim_x; ++d) {
                double diff = x[d] - d_Xref[d*(Tr+1)+t];
                gc += diff*diff;
            }
            cost += sqrt(gc);

            f_quadrotor(x, u_val, xd_);
        } else if (model_type == 2) {
            double u_val[2] = { Ui_i[0*Tr+t], Ui_i[1*Tr+t] };
            cost += p_velo(x, d_x_target, dim_x);
            // guide cost: distance to reference trajectory
            double gc = 0.0;
            for (int d = 0; d < dim_x; ++d) {
                double diff = x[d] - d_Xref[d*(Tr+1)+t];
                gc += diff*diff;
            }
            cost += sqrt(gc);

            f_velo(x, u_val, xd_);
        } else if (model_type == 3) {
            double u_manip[GPU_MAX_DIM_U];
            for (int _d = 0; _d < dim_u; ++_d) u_manip[_d] = Ui_i[_d*Tr+t];
            cost += p_manipulator(x, d_x_target, dim_x);
            double gc = 0.0;
            for (int d = 0; d < dim_x; ++d) {
                double diff = x[d] - d_Xref[d*(Tr+1)+t];
                gc += diff*diff;
            }
            cost += sqrt(gc);
            f_manipulator(x, u_manip, xd_, dim_u);
        }
        for (int d = 0; d < dim_x; ++d) xn[d] = x[d] + dt*xd_[d];

        if (!hit && check_collision(x, with_map, d_map, max_row, max_col, res,
                                    d_circles, n_circ, d_rects, n_rect)) {
            hit = true; cost = 1e8;
        }
        for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
    }
    if (!hit) {
        if (model_type == 0) {
            cost += p_wmrobot(x, d_x_target, dim_x);
            cost += p_wmrobot(x, d_x_target, dim_x);
        } else if (model_type == 1) {
            cost += p_quadrotor(x, d_x_target, dim_x);
            cost += p_quadrotor(x, d_x_target, dim_x);
        } else if (model_type == 2) {
            cost += p_velo(x, d_x_target, dim_x);
            cost += p_velo(x, d_x_target, dim_x);
        } else if (model_type == 3) {
            cost += p_manipulator(x, d_x_target, dim_x);
            cost += p_manipulator(x, d_x_target, dim_x);
        }
        if (check_collision(x, with_map, d_map, max_row, max_col, res,
                            d_circles, n_circ, d_rects, n_rect))
            cost = 1e8;
    }
    d_costs[i] = cost;
}

// ── Helper: safe malloc ─────────────────────────────────────────
static void safe_cuda_malloc(double** ptr, size_t sz) {
    if (*ptr) { cudaFree(*ptr); *ptr = nullptr; }
    if (sz > 0) CUDA_CHECK(cudaMalloc(ptr, sz));
}

// ── BiMPPI_GPU methods ──────────────────────────────────────────
BiMPPI_GPU::~BiMPPI_GPU() {
    freeForward(); freeBackward(); freeGuide(); freeCommon();
    curandDestroyGenerator(curand_gen);
}

void BiMPPI_GPU::freeForward() {
    safe_cuda_malloc(&d_Uf0, 0); safe_cuda_malloc(&d_Ufi, 0);
    safe_cuda_malloc(&d_noise_f, 0); safe_cuda_malloc(&d_costs_f, 0);
    safe_cuda_malloc(&d_Uf_out, 0); safe_cuda_malloc(&d_Di_f, 0);
}
void BiMPPI_GPU::freeBackward() {
    safe_cuda_malloc(&d_Ub0, 0); safe_cuda_malloc(&d_Ubi, 0);
    safe_cuda_malloc(&d_noise_b, 0); safe_cuda_malloc(&d_costs_b, 0);
    safe_cuda_malloc(&d_Ub_out, 0); safe_cuda_malloc(&d_Di_b, 0);
}
void BiMPPI_GPU::freeGuide() {
    safe_cuda_malloc(&d_Xref, 0);
    safe_cuda_malloc(&d_Ur0, 0); safe_cuda_malloc(&d_Uri, 0);
    safe_cuda_malloc(&d_noise_r, 0); safe_cuda_malloc(&d_costs_r, 0);
    safe_cuda_malloc(&d_Ur_out, 0);
}
void BiMPPI_GPU::freeCommon() {
    safe_cuda_malloc(&d_x_init, 0); safe_cuda_malloc(&d_x_target, 0);
    safe_cuda_malloc(&d_sigma, 0);
    safe_cuda_malloc(&d_map, 0); safe_cuda_malloc(&d_circles, 0);
    safe_cuda_malloc(&d_rects, 0);
}

void BiMPPI_GPU::allocForward() {
    safe_cuda_malloc(&d_Uf0,    (size_t)dim_u*Tf*sizeof(double));
    safe_cuda_malloc(&d_Ufi,    (size_t)Nf*dim_u*Tf*sizeof(double));
    safe_cuda_malloc(&d_noise_f,(size_t)Nf*dim_u*Tf*sizeof(double));
    safe_cuda_malloc(&d_costs_f,(size_t)Nf*sizeof(double));
    safe_cuda_malloc(&d_Uf_out, (size_t)dim_u*Tf*sizeof(double));
    safe_cuda_malloc(&d_Di_f,   (size_t)Nf*dim_u*sizeof(double));
}
void BiMPPI_GPU::allocBackward() {
    safe_cuda_malloc(&d_Ub0,    (size_t)dim_u*Tb*sizeof(double));
    safe_cuda_malloc(&d_Ubi,    (size_t)Nb*dim_u*Tb*sizeof(double));
    safe_cuda_malloc(&d_noise_b,(size_t)Nb*dim_u*Tb*sizeof(double));
    safe_cuda_malloc(&d_costs_b,(size_t)Nb*sizeof(double));
    safe_cuda_malloc(&d_Ub_out, (size_t)dim_u*Tb*sizeof(double));
    safe_cuda_malloc(&d_Di_b,   (size_t)Nb*dim_u*sizeof(double));
}

void BiMPPI_GPU::allocGuide() {
    int Tr_max = Tf + Tb;
    safe_cuda_malloc(&d_Xref, (size_t)dim_x*(Tr_max+1)*sizeof(double));
    safe_cuda_malloc(&d_Ur0,  (size_t)dim_u*Tr_max*sizeof(double));
    safe_cuda_malloc(&d_Uri,  (size_t)Nr*dim_u*Tr_max*sizeof(double));
    size_t ncnt = (size_t)Nr*dim_u*Tr_max; if (ncnt % 2) ncnt++;
    safe_cuda_malloc(&d_noise_r, ncnt*sizeof(double));
    safe_cuda_malloc(&d_costs_r, (size_t)Nr*sizeof(double));
    safe_cuda_malloc(&d_Ur_out,  (size_t)dim_u*Tr_max*sizeof(double));
}

void BiMPPI_GPU::init(BiMPPIParam p) {
    dt = p.dt; Tf = p.Tf; Tb = p.Tb;
    Nf = p.Nf; Nb = p.Nb; Nr = p.Nr;
    gamma_u = p.gamma_u;
    x_init = p.x_init; x_target = p.x_target;
    deviation_mu = p.deviation_mu; epsilon = p.epsilon;
    minpts = p.minpts; psi = p.psi;

    sigma_diag.resize(dim_u);
    for (int d = 0; d < dim_u; ++d) sigma_diag[d] = p.sigma_u(d,d);

    full_cluster_f.resize(Nf); std::iota(full_cluster_f.begin(), full_cluster_f.end(), 0);
    full_cluster_b.resize(Nb); std::iota(full_cluster_b.begin(), full_cluster_b.end(), 0);

    u0 = Eigen::VectorXd::Zero(dim_u);
    dummy_u = Eigen::VectorXd::Zero(dim_u);

    CUDA_CHECK(cudaMalloc(&d_x_init,   dim_x*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_target, dim_x*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sigma,    dim_u*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sigma, sigma_diag.data(), dim_u*sizeof(double), cudaMemcpyHostToDevice));

    allocForward(); allocBackward(); allocGuide();
    alloc_Nf=Nf; alloc_Nb=Nb; alloc_Tf=Tf; alloc_Tb=Tb;
}

void BiMPPI_GPU::setCollisionChecker(CollisionChecker* cc) {
    collision_checker = cc;
    uploadCollisionData();
}

void BiMPPI_GPU::uploadCollisionData() {
    map_max_row = (int)collision_checker->map.size();
    map_max_col = map_max_row>0?(int)collision_checker->map[0].size():0;
    map_resolution = collision_checker->resolution;
    with_map = collision_checker->with_map;

    if (with_map && map_max_row>0) {
        size_t sz = (size_t)map_max_row*map_max_col*sizeof(double);
        safe_cuda_malloc(&d_map, sz);
        std::vector<double> flat(map_max_row*map_max_col);
        for (int r=0;r<map_max_row;++r)
            for (int c=0;c<map_max_col;++c)
                flat[r*map_max_col+c]=collision_checker->map[r][c];
        CUDA_CHECK(cudaMemcpy(d_map,flat.data(),sz,cudaMemcpyHostToDevice));
    }
    n_circles=(int)collision_checker->circles.size();
    if (n_circles>0) {
        size_t sz=n_circles*4*sizeof(double);
        safe_cuda_malloc(&d_circles,sz);
        std::vector<double> buf(n_circles*4);
        for (int i=0;i<n_circles;++i) for(int j=0;j<4;++j) buf[i*4+j]=collision_checker->circles[i][j];
        CUDA_CHECK(cudaMemcpy(d_circles,buf.data(),sz,cudaMemcpyHostToDevice));
    }
    n_rects=(int)collision_checker->rectangles.size();
    if (n_rects>0) {
        size_t sz=n_rects*4*sizeof(double);
        safe_cuda_malloc(&d_rects,sz);
        std::vector<double> buf(n_rects*4);
        for (int i=0;i<n_rects;++i) for(int j=0;j<4;++j) buf[i*4+j]=collision_checker->rectangles[i][j];
        CUDA_CHECK(cudaMemcpy(d_rects,buf.data(),sz,cudaMemcpyHostToDevice));
    }
}

// ── Helper: Eigen → flat row-major ────────────────────────────────
static std::vector<double> eigen_to_flat(const Eigen::MatrixXd& M, int rows, int cols) {
    std::vector<double> v(rows*cols);
    for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) v[r*cols+c]=M(r,c);
    return v;
}
static Eigen::MatrixXd flat_Ui_to_eigen(const std::vector<double>& f, int N, int du, int T) {
    Eigen::MatrixXd M(N*du, T);
    for(int i=0;i<N;++i) for(int d=0;d<du;++d) for(int t=0;t<T;++t)
        M(i*du+d,t)=f[i*(du*T)+d*T+t];
    return M;
}
static Eigen::MatrixXd flat_Di_to_eigen_col(const std::vector<double>& f, int N, int du) {
    Eigen::MatrixXd M(du,N);
    for(int i=0;i<N;++i) for(int d=0;d<du;++d) M(d,i)=f[i*du+d];
    return M;
}

// ── DBSCAN (CPU) ──────────────────────────────────────────────────
void BiMPPI_GPU::dbscan(std::vector<std::vector<int>>& clusters,
                         const Eigen::MatrixXd& Di,
                         const Eigen::VectorXd& costs, int Ns) {
    clusters.clear();
    std::map<int,std::vector<int>> tree;
    std::vector<bool> core(Ns,false);
#pragma omp parallel for
    for(int i=0;i<Ns;++i){
        if(costs(i)>1e7) continue;
        for(int j=i+1;j<Ns;++j){
            if(costs(j)>1e7) continue;
            if(deviation_mu*(Di.col(i)-Di.col(j)).norm()<epsilon)
#pragma omp critical
            { tree[i].push_back(j); tree[j].push_back(i); }
        }
    }
    for(int i=0;i<Ns;++i) if((int)tree[i].size()>minpts) core[i]=true;
    std::vector<bool> vis(Ns,false);
    for(int i=0;i<Ns;++i){
        if(!core[i]||vis[i]) continue;
        std::deque<int> q; std::vector<int> cl;
        q.push_back(i); cl.push_back(i); vis[i]=true;
        while(!q.empty()){
            int n=q.front(); q.pop_front();
            for(int nb:tree[n]){ if(vis[nb]) continue; vis[nb]=true; cl.push_back(nb); if(core[nb]) q.push_back(nb); }
        }
        clusters.push_back(cl);
    }
}

// ── calculateU (CPU) ──────────────────────────────────────────────
void BiMPPI_GPU::calculateU(Eigen::MatrixXd& Uout,
                             const std::vector<std::vector<int>>& clusters,
                             const Eigen::VectorXd& costs,
                             const Eigen::MatrixXd& Ui_cpu, int T_steps) {
    int nc=(int)clusters.size();
    Uout=Eigen::MatrixXd::Zero(nc*dim_u, T_steps);
#pragma omp parallel for
    for(int idx=0;idx<nc;++idx){
        int pts=(int)clusters[idx].size();
        double mc=std::numeric_limits<double>::max();
        for(int k:clusters[idx]) mc=std::min(mc,costs(k));
        double tw=0.0; std::vector<double> wts(pts);
        for(int i=0;i<pts;++i){ wts[i]=std::exp(-gamma_u*(costs(clusters[idx][i])-mc)); tw+=wts[i]; }
        for(int i=0;i<pts;++i)
            Uout.middleRows(idx*dim_u,dim_u)+=(wts[i]/tw)*Ui_cpu.middleRows(clusters[idx][i]*dim_u,dim_u);
        Eigen::Ref<Eigen::MatrixXd> slice = Uout.middleRows(idx*dim_u, dim_u);
        h(slice);
    }
}

// ── forwardRollout ────────────────────────────────────────────────
void BiMPPI_GPU::forwardRollout() {
    auto t0=std::chrono::high_resolution_clock::now();
    auto ff=eigen_to_flat(U_f0,dim_u,Tf);
    CUDA_CHECK(cudaMemcpy(d_Uf0,ff.data(),dim_u*Tf*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_init,  x_init.data(),  dim_x*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_target,x_target.data(),dim_x*sizeof(double),cudaMemcpyHostToDevice));
    size_t nc=(size_t)Nf*dim_u*Tf; if(nc%2)nc++;
    CURAND_CHECK(curandGenerateNormalDouble(curand_gen,d_noise_f,nc,0.0,1.0));
    int B=256,G=(Nf+B-1)/B;
    bi_rollout_kernel<<<G,B>>>(d_Uf0,d_Ufi,d_noise_f,d_sigma,d_x_init,d_x_target,
        d_costs_f,d_Di_f,with_map,d_map,map_max_row,map_max_col,map_resolution,
        d_circles,n_circles,d_rects,n_rects,Nf,dim_u,dim_x,Tf,(double)dt,gamma_u,model_type);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    elapsed_rollout+=std::chrono::duration<double>(t1-t0).count();

    std::vector<double> hc(Nf),hUi((size_t)Nf*dim_u*Tf),hDi((size_t)Nf*dim_u);
    CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_f,Nf*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hUi.data(),d_Ufi,(size_t)Nf*dim_u*Tf*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hDi.data(),d_Di_f,(size_t)Nf*dim_u*sizeof(double),cudaMemcpyDeviceToHost));
    Eigen::VectorXd costs_f=Eigen::Map<Eigen::VectorXd>(hc.data(),Nf);
    Eigen::MatrixXd Ui_f=flat_Ui_to_eigen(hUi,Nf,dim_u,Tf);
    Eigen::MatrixXd Di_f=flat_Di_to_eigen_col(hDi,Nf,dim_u);
    bool ok=(costs_f.array()<1e7).all();
    clusters_f.clear();
    if(!ok) dbscan(clusters_f,Di_f,costs_f,Nf);
    if(clusters_f.empty()) clusters_f.push_back(full_cluster_f);
    calculateU(Uf,clusters_f,costs_f,Ui_f,Tf);
    Xf.resize(clusters_f.size()*dim_x,Tf+1);
    for(int ci=0;ci<(int)clusters_f.size();++ci){
        Xf.block(ci*dim_x,0,dim_x,1)=x_init;
        for(int t=0;t<Tf;++t){
            Xf.block(ci*dim_x,t+1,dim_x,1)=Xf.block(ci*dim_x,t,dim_x,1)+
                (double)dt*f(Xf.block(ci*dim_x,t,dim_x,1), Uf.block(ci*dim_u,t,dim_u,1));
        }
    }
    elapsed_clustering+=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t1).count();
}

// ── backwardRollout ───────────────────────────────────────────────
void BiMPPI_GPU::backwardRollout() {
    auto t0=std::chrono::high_resolution_clock::now();
    auto bf=eigen_to_flat(U_b0,dim_u,Tb);
    CUDA_CHECK(cudaMemcpy(d_Ub0,bf.data(),dim_u*Tb*sizeof(double),cudaMemcpyHostToDevice));
    size_t nc=(size_t)Nb*dim_u*Tb; if(nc%2)nc++;
    CURAND_CHECK(curandGenerateNormalDouble(curand_gen,d_noise_b,nc,0.0,1.0));
    int B=256,G=(Nb+B-1)/B;
    backward_rollout_kernel<<<G,B>>>(d_Ub0,d_Ubi,d_noise_b,d_sigma,d_x_init,d_x_target,
        d_costs_b,d_Di_b,with_map,d_map,map_max_row,map_max_col,map_resolution,
        d_circles,n_circles,d_rects,n_rects,Nb,dim_u,dim_x,Tb,(double)dt,gamma_u,model_type);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    elapsed_rollout+=std::chrono::duration<double>(t1-t0).count();

    std::vector<double> hc(Nb),hUi((size_t)Nb*dim_u*Tb),hDi((size_t)Nb*dim_u);
    CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_b,Nb*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hUi.data(),d_Ubi,(size_t)Nb*dim_u*Tb*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hDi.data(),d_Di_b,(size_t)Nb*dim_u*sizeof(double),cudaMemcpyDeviceToHost));
    Eigen::VectorXd costs_b=Eigen::Map<Eigen::VectorXd>(hc.data(),Nb);
    Eigen::MatrixXd Ui_b=flat_Ui_to_eigen(hUi,Nb,dim_u,Tb);
    Eigen::MatrixXd Di_b=flat_Di_to_eigen_col(hDi,Nb,dim_u);
    bool ok=(costs_b.array()<1e7).all();
    clusters_b.clear();
    if(!ok) dbscan(clusters_b,Di_b,costs_b,Nb);
    if(clusters_b.empty()) clusters_b.push_back(full_cluster_b);
    calculateU(Ub,clusters_b,costs_b,Ui_b,Tb);
    Xb.resize(clusters_b.size()*dim_x,Tb+1);
    for(int ci=0;ci<(int)clusters_b.size();++ci){
        Xb.block(ci*dim_x,Tb,dim_x,1)=x_target;
        for(int t=Tb-1;t>=0;--t){
            Eigen::VectorXd u_val(dim_u);
            if(t==Tb-1){ u_val=Ub.block(ci*dim_u,t,dim_u,1); }
            else        { u_val=Ub.block(ci*dim_u,t+1,dim_u,1); }
            Xb.block(ci*dim_x,t,dim_x,1)=Xb.block(ci*dim_x,t+1,dim_x,1)-(double)dt*f(Xb.block(ci*dim_x,t+1,dim_x,1), u_val);
        }
    }
    elapsed_clustering+=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t1).count();
}

// ── selectConnection ──────────────────────────────────────────────
void BiMPPI_GPU::selectConnection() {
    joints.clear();
    for(int cf=0;cf<(int)clusters_f.size();++cf){
        double mn=std::numeric_limits<double>::max(); int cb=0,df_=0,db_=0;
        for(int cb_=0;cb_<(int)clusters_b.size();++cb_)
            for(int df__=0;df__<=Tf;++df__)
                for(int db__=0;db__<=Tb;++db__){
                    double n=(Xf.block(cf*dim_x,df__,dim_x,1)-Xb.block(cb_*dim_x,db__,dim_x,1)).norm();
                    if(n<mn){mn=n;cb=cb_;df_=df__;db_=db__;}
                }
        joints.push_back({cf,cb,df_,db_});
    }
}

// ── concatenate ───────────────────────────────────────────────────
void BiMPPI_GPU::concatenate() {
    Uc.clear(); Xc.clear();
    for(auto& j:joints){
        int cf=j[0],cb=j[1],df=j[2],db=j[3],len=std::max(Tf,df+(Tb-db));
        Eigen::MatrixXd U(dim_u,len); U.setZero();
        Eigen::MatrixXd X(dim_x,len+1); X.setZero();
        if(df>0) U.leftCols(df)=Uf.block(cf*dim_u,0,dim_u,df);
        X.leftCols(df+1)=Xf.block(cf*dim_x,0,dim_x,df+1);
        if(db!=Tb){ U.middleCols(df,Tb-db)=Ub.block(cb*dim_u,db,dim_u,Tb-db);
                    X.middleCols(df+1,Tb-db)=Xb.block(cb*dim_x,db+1,dim_x,Tb-db); }
        if(df+(Tb-db)<Tf){ U.rightCols(Tf-(df+(Tb-db))).colwise()=dummy_u;
                           X.rightCols(Tf-(df+(Tb-db))).colwise()=x_target; }
        Uc.push_back(U); Xc.push_back(X);
    }
}

// ── guideMPPI ─────────────────────────────────────────────────────
void BiMPPI_GPU::guideMPPI() {
    Ur.clear(); Cr.clear(); Xr.clear();
    for(int r=0;r<(int)joints.size();++r){
        int Tr=Uc[r].cols();
        std::vector<double> xrf(dim_x*(Tr+1));
        for(int d=0;d<dim_x;++d) for(int t=0;t<=Tr;++t) xrf[d*(Tr+1)+t]=Xc[r](d,t);
        CUDA_CHECK(cudaMemcpy(d_Xref,xrf.data(),(size_t)dim_x*(Tr+1)*sizeof(double),cudaMemcpyHostToDevice));
        auto r0f=eigen_to_flat(Uc[r],dim_u,Tr);
        CUDA_CHECK(cudaMemcpy(d_Ur0,r0f.data(),(size_t)dim_u*Tr*sizeof(double),cudaMemcpyHostToDevice));
        size_t ncnt=(size_t)Nr*dim_u*Tr; if(ncnt%2)ncnt++;
        CURAND_CHECK(curandGenerateNormalDouble(curand_gen,d_noise_r,ncnt,0.0,1.0));
        int B=256,G=(Nr+B-1)/B;
        guide_rollout_kernel<<<G,B>>>(d_Ur0,d_Uri,d_noise_r,d_sigma,d_x_init,d_x_target,d_Xref,d_costs_r,
            with_map,d_map,map_max_row,map_max_col,map_resolution,d_circles,n_circles,d_rects,n_rects,
            Nr,dim_u,dim_x,Tr,(double)dt,gamma_u,model_type);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> hcr(Nr),hri((size_t)Nr*dim_u*Tr);
        CUDA_CHECK(cudaMemcpy(hcr.data(),d_costs_r,Nr*sizeof(double),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hri.data(),d_Uri,(size_t)Nr*dim_u*Tr*sizeof(double),cudaMemcpyDeviceToHost));
        Eigen::VectorXd costs_r=Eigen::Map<Eigen::VectorXd>(hcr.data(),Nr);
        double mc=costs_r.minCoeff();
        Eigen::VectorXd wts=(-gamma_u*(costs_r.array()-mc)).exp(); wts/=wts.sum();
        Eigen::MatrixXd Ures=Eigen::MatrixXd::Zero(dim_u,Tr);
        for(int i=0;i<Nr;++i) for(int d=0;d<dim_u;++d) for(int t=0;t<Tr;++t)
            Ures(d,t)+=wts(i)*hri[i*(dim_u*Tr)+d*Tr+t];
        h(Ures);
        Eigen::MatrixXd Xi(dim_x,Tr+1); Xi.col(0)=x_init; double cost=0.0;
        for(int t=0;t<Tr;++t){
            Xi.col(t+1)=Xi.col(t)+(double)dt*f(Xi.col(t), Ures.col(t));
            cost+=p(Xi.col(t), x_target);
        }
        cost+=p(Xi.col(Tr), x_target);
        if(collision_checker->getCollisionGrid(Xi.col(Tr))) cost=1e8;
        Ur.push_back(Ures); Cr.push_back(cost); Xr.push_back(Xi);
    }
    double mn=std::numeric_limits<double>::max(); int idx=0;
    for(int r=0;r<(int)joints.size();++r) if(Cr[r]<mn){mn=Cr[r];idx=r;}
    Uo=Ur[idx]; Xo=Xr[idx]; u0=Uo.col(0);
}

void BiMPPI_GPU::partitioningControl() {
    U_f0=Uo.leftCols(Tf);
    U_b0=Eigen::MatrixXd::Zero(dim_u,Tb);
}

void BiMPPI_GPU::solve() {
    elapsed_rollout=elapsed_clustering=0.0;
    start=std::chrono::high_resolution_clock::now();
    backwardRollout(); forwardRollout();
    auto t2=std::chrono::high_resolution_clock::now();
    selectConnection(); concatenate();
    auto t3=std::chrono::high_resolution_clock::now();
    elapsed_connection=std::chrono::duration<double>(t3-t2).count();
    guideMPPI();
    auto t4=std::chrono::high_resolution_clock::now();
    elapsed_guide=std::chrono::duration<double>(t4-t3).count();
    elapsed_1=t4-start;
    partitioningControl();
    elapsed=elapsed_rollout+elapsed_clustering+elapsed_connection+elapsed_guide;

    // ── Visualization data export ──
    if (vis_logger && vis_logger->enabled) {
      // Forward cluster trajectories (Xf: clusters_f.size() * dim_x rows)
      std::vector<Eigen::MatrixXd> fwd_trajs;
      for (int ci = 0; ci < (int)clusters_f.size(); ++ci)
        fwd_trajs.push_back(Xf.block(ci * dim_x, 0, dim_x, Tf + 1));
      vis_logger->saveTrajectories("forward_clusters", fwd_trajs);
      // Backward cluster trajectories
      std::vector<Eigen::MatrixXd> bwd_trajs;
      for (int ci = 0; ci < (int)clusters_b.size(); ++ci)
        bwd_trajs.push_back(Xb.block(ci * dim_x, 0, dim_x, Tb + 1));
      vis_logger->saveTrajectories("backward_clusters", bwd_trajs);
      // All guide candidate trajectories
      if (!Xr.empty())
        vis_logger->saveTrajectories("guide_candidates", Xr);
      // Optimal trajectory
      vis_logger->saveTrajectory("optimal", Xo);
      vis_logger->savePosition(x_init);
    }

    visual_traj.push_back(x_init);
}

void BiMPPI_GPU::move() {
    x_init=x_init+(double)dt*f(x_init, u0);
    U_f0.leftCols(U_f0.cols()-1)=U_f0.rightCols(U_f0.cols()-1);
}
