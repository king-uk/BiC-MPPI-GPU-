cmake_minimum_required(VERSION 3.18)
project(mppi LANGUAGES CXX)

# ── Standard dependencies ─────────────────────────────────────────
find_package(PythonLibs REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

# ── Include directories ───────────────────────────────────────────
include_directories(
    mppi/
    model/
    include/EigenRand/
    ${EIGEN3_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    include/matplotlibcpp/
)

# ── Macros ────────────────────────────────────────────────────────
macro(add_cpu_target SRC)
    get_filename_component(NAME ${SRC} NAME_WE)
    add_executable(${NAME} ${SRC})
    target_compile_options(${NAME} PRIVATE -O3)
    target_link_libraries(${NAME} ${PYTHON_LIBRARIES} OpenMP::OpenMP_CXX)
endmacro()

# ===========================================================================
# 빌드할 소스 파일 선택 (사용할 예제의 주석을 해제하세요)
# ===========================================================================

# ---- WMRobot CPU 버전 ----
add_cpu_target(src/wmrobot/cpu/mppi.cpp)
add_cpu_target(src/wmrobot/cpu/cluster_mppi.cpp)
add_cpu_target(src/wmrobot/cpu/bi_mppi.cpp)
add_cpu_target(src/wmrobot/cpu/log_mppi.cpp)
add_cpu_target(src/wmrobot/cpu/svgd_mppi.cpp)
add_cpu_target(src/wmrobot/cpu/rrt_connect.cpp)

# ---- Clustering Parameter (CPU) ----
# add_cpu_target("src/wmrobot/clustering parameter/bi_mppi.cpp")
# add_cpu_target("src/wmrobot/clustering parameter/bi_mppi_epsilon_0.001.cpp")
# add_cpu_target("src/wmrobot/clustering parameter/bi_mppi_epsilon_0.005.cpp")
# add_cpu_target("src/wmrobot/clustering parameter/bi_mppi_epsilon_0.1.cpp")
# add_cpu_target("src/wmrobot/clustering parameter/bi_mppi_epsilon_0.5.cpp")
# add_cpu_target("src/wmrobot/clustering parameter/bi_mppi_epsilon_1.cpp")

# ---- Bicycle (CPU) ----
# add_cpu_target(src/bicycle/mppi.cpp)
# add_cpu_target(src/bicycle/bi_mppi.cpp)

# ---- Quadrotor (CPU) ----
# add_cpu_target(src/quadrotor/mppi.cpp)

# ---- Parking (CPU) ----
# add_cpu_target(src/parking/mppi.cpp)