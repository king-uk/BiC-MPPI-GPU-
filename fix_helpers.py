with open("mppi/cuda/bi_mppi_gpu.cu", "r") as f:
    text = f.read()

helpers = """
// ── Helper ──────────────────────────────────────────────────────────
static std::vector<double> eigen_to_flat(const Eigen::MatrixXd& M, int d, int t) {
    std::vector<double> r(d*t);
    for(int i=0;i<d;++i) for(int j=0;j<t;++j) r[i*t+j]=M(i,j);
    return r;
}
static Eigen::MatrixXd flat_Ui_to_eigen(const std::vector<double>& f, int N, int du, int T) {
    Eigen::MatrixXd r(N*du,T);
    for(int n=0;n<N;++n) for(int i=0;i<du;++i) for(int t=0;t<T;++t)
        r(n*du+i,t)=f[n*(du*T)+i*T+t];
    return r;
}
static Eigen::MatrixXd flat_Di_to_eigen_col(const std::vector<double>& f, int N, int du) {
    Eigen::MatrixXd r(du,N);
    for(int n=0;n<N;++n) for(int d=0;d<du;++d) r(d,n)=f[n*du+d];
    return r;
}
"""

with open("mppi/cuda/svgd_mppi_gpu.cu", "r") as f:
    svgd = f.read()

# Insert after safe_cuda_malloc
marker = "static void safe_cuda_malloc"
end_marker = "}\n"
insert_pos = svgd.find(end_marker, svgd.find(marker)) + 2

svgd = svgd[:insert_pos] + helpers + svgd[insert_pos:]

with open("mppi/cuda/svgd_mppi_gpu.cu", "w") as f:
    f.write(svgd)

print("Injected helper functions into svgd_mppi_gpu.cu")
