#!/usr/bin/env bash
# ============================================================
# build_gpu.sh — BiC-MPPI GPU 버전 빌드 스크립트
#
# 시스템에 CUDA toolkit이 /usr/local/cuda에 설치되어 있으면
# 자동으로 감지합니다. 없으면 MATLAB 번들 nvcc를 사용합니다.
#
# 사용법:
#   bash build_gpu.sh          # 전체 빌드
#   bash build_gpu.sh mppi     # mppi만 빌드
# ============================================================
set -e

# ── 경로 설정 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NVCC=""
CUDA_INCLUDE=""
CUDA_LIBDIR=""
CURAND_LIB=""
CUDART_LIB=""

# 1) 시스템 CUDA toolkit (/usr/local/cuda*)
for d in /usr/local/cuda /usr/local/cuda-12.5 /usr/local/cuda-12.6 /usr/local/cuda-13; do
    if [ -f "$d/bin/nvcc" ]; then
        NVCC="$d/bin/nvcc"
        CUDA_INCLUDE="$d/include"
        CUDA_LIBDIR="$d/lib64"
        break
    fi
done

# 2) apt 등으로 설치된 전역 nvcc 확인
if [ -z "$NVCC" ] && command -v nvcc >/dev/null 2>&1; then
    NVCC=$(command -v nvcc)
    CUDA_INCLUDE="/usr/include"
    CUDA_LIBDIR="/usr/lib/x86_64-linux-gnu"
fi

if [ -z "$NVCC" ]; then
    echo "ERROR: nvcc를 찾을 수 없습니다. CUDA toolkit을 설치해 주세요."
    echo "  sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

# curand 탐색
CURAND_LIB=""
for lib in "$CUDA_LIBDIR/libcurand.so" "/usr/lib/x86_64-linux-gnu/libcurand.so"; do
    if [ -f "$lib" ]; then CURAND_LIB="$lib"; break; fi
done

# cudart 탐색
CUDART_LIB=""
for lib in "$CUDA_LIBDIR/libcudart.so" "/usr/lib/x86_64-linux-gnu/libcudart.so"; do
    if [ -f "$lib" ]; then CUDART_LIB="$lib"; break; fi
done

echo "=== GPU Build Configuration ==="
echo "  nvcc        : $NVCC"
echo "  CUDA include: $CUDA_INCLUDE"
echo "  cudart      : ${CUDART_LIB:-NOT FOUND}"
echo "  curand      : ${CURAND_LIB:-NOT FOUND}"
echo "================================"

# ── 컴파일 옵션 ───────────────────────────────────────────────────
ARCH="-arch=sm_86"          # RTX 5060 (sm_120)은 PTX JIT 사용
CXX=g++
CXXFLAGS="-O3 -std=c++17 -fopenmp"
# GCC 11 버그 우회 (nvcc 11.5 호환성을 위해 g++-10 사용 권장)
CCBIN_FLAG=""
if command -v g++-10 >/dev/null 2>&1; then
    CCBIN_FLAG="-ccbin g++-10"
fi

NVCCFLAGS="-O3 --std=c++14 $ARCH $CCBIN_FLAG --expt-relaxed-constexpr -Xcompiler -fopenmp"

INCLUDES="-I./mppi -I./mppi/cuda -I./model \
          -I./include/EigenRand -I./include/matplotlibcpp \
          -I$CUDA_INCLUDE \
          $(python3-config --includes) \
          $(pkg-config --cflags eigen3)"

EIGEN_INC=$(pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")

INCLUDES="-I./mppi -I./mppi/cuda -I./model \
          -I./include/EigenRand -I./include/matplotlibcpp \
          -I$CUDA_INCLUDE \
          $(python3-config --includes) \
          $EIGEN_INC"

LDFLAGS="-fopenmp"
[ -n "$CUDART_LIB" ] && LDFLAGS="$LDFLAGS $CUDART_LIB"
[ -n "$CURAND_LIB" ] && LDFLAGS="$LDFLAGS $CURAND_LIB"
LDFLAGS="$LDFLAGS $(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags)"

# ── 빌드 디렉토리 ─────────────────────────────────────────────────
mkdir -p build_gpu

# ── GPU solver 오브젝트 파일 컴파일 ──────────────────────────────
echo "[1/6] Compiling mppi_gpu.cu ..."
$NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/mppi_gpu.cu -o build_gpu/mppi_gpu.o

echo "[2/6] Compiling cluster_mppi_gpu.cu ..."
$NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/cluster_mppi_gpu.cu -o build_gpu/cluster_mppi_gpu.o

echo "[3/6] Compiling bi_mppi_gpu.cu ..."
$NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/bi_mppi_gpu.cu -o build_gpu/bi_mppi_gpu.o

echo "[4/6] Compiling svgd_mppi_gpu.cu ..."
$NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/svgd_mppi_gpu.cu -o build_gpu/svgd_mppi_gpu.o

echo "[5/6] Compiling log_mppi_gpu.cu ..."
$NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/log_mppi_gpu.cu -o build_gpu/log_mppi_gpu.o

# GPU 오브젝트들을 ar로 정적 라이브러리로 묶기
GPU_OBJS="build_gpu/mppi_gpu.o build_gpu/cluster_mppi_gpu.o build_gpu/bi_mppi_gpu.o build_gpu/svgd_mppi_gpu.o build_gpu/log_mppi_gpu.o"
ar rcs build_gpu/libmppi_gpu.a $GPU_OBJS
echo "  → build_gpu/libmppi_gpu.a 생성 완료"

# ── 실행 파일 링크 ────────────────────────────────────────────────
build_target() {
    local SRC="$1"
    local NAME=$(basename "${SRC%.cpp}")
    echo "[4/?] Building $NAME ..."
    $CXX $CXXFLAGS $INCLUDES "$SRC" \
        -Lbuild_gpu -lmppi_gpu \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "build_gpu/$NAME"
    echo "  → build_gpu/$NAME 완료"
}

# ===========================================================================
# 빌드할 소스 파일 선택 (사용할 예제의 주석을 해제하세요)
# ===========================================================================

# ---- WMRobot GPU 버전 ----
build_target src/wmrobot/gpu/mppi.cpp
build_target src/wmrobot/gpu/cluster_mppi.cpp
build_target src/wmrobot/gpu/bi_mppi.cpp
build_target src/wmrobot/gpu/svgd_mppi.cpp
# build_target src/wmrobot/gpu/log_mppi.cpp

# ---- Quadrotor GPU 버전 ----
# build_target src/quadrotor/gpu/mppi.cpp
# build_target src/quadrotor/gpu/cluster_mppi.cpp
# build_target src/quadrotor/gpu/bi_mppi.cpp
# build_target src/quadrotor/gpu/svgd_mppi.cpp
# build_target src/quadrotor/gpu/log_mppi.cpp

# ---- Manipulator GPU 버전 ----
# build_target src/manipulator/gpu/mppi.cpp
# build_target src/manipulator/gpu/cluster_mppi.cpp
# build_target src/manipulator/gpu/bi_mppi.cpp
# build_target src/manipulator/gpu/svgd_mppi.cpp
# build_target src/manipulator/gpu/mppi_cylinder.cpp

# ---- Velo GPU 버전 ----
# build_target src/velo/gpu/mppi.cpp
# build_target src/velo/gpu/cluster_mppi.cpp
# build_target src/velo/gpu/bi_mppi.cpp
# build_target src/velo/gpu/svgd_mppi.cpp
# build_target src/velo/gpu/log_mppi.cpp

# ---- WMRobot map_78 시각화 버전 ----
build_target_named() {
    local SRC="$1"
    local NAME="$2"
    echo "[vis] Building $NAME ..."
    $CXX $CXXFLAGS $INCLUDES "$SRC" \
        -Lbuild_gpu -lmppi_gpu \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "build_gpu/$NAME"
    echo "  → build_gpu/$NAME 완료"
}
# build_target_named src/wmrobot/map_285/mppi.cpp vis_mppi
# build_target_named src/wmrobot/map_285/cluster_mppi.cpp vis_cluster_mppi
# build_target_named src/wmrobot/map_285/bi_mppi.cpp vis_bi_mppi
# build_target_named src/wmrobot/map_285/svgd_mppi.cpp vis_svgd_mppi

echo ""
echo "=== 빌드 완료 ==="
echo "실행: cd build_gpu && ./mppi"
echo "      cd build_gpu && ./bi_mppi"
echo "      cd build_gpu && ./cluster_mppi"
echo ""
echo "시각화: cd build_gpu && ./vis_mppi && python3 ../build/visualize_mppi.py vis_data/mppi/"
echo "        cd build_gpu && ./vis_bi_mppi && python3 ../build/visualize_mppi.py vis_data/bi_mppi/"

