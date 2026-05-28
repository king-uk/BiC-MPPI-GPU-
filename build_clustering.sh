#!/usr/bin/env bash
# ============================================================
# clustering.sh — 빌드 스크립트 for Clustering Parameter cpp 파일들
#
# src/wmrobot/clustering parameter/ 안의 모든 cpp 파일들을 빌드하여
# build_gpu/ 디렉토리에 실행 파일로 저장합니다.
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

echo "=== GPU Build Configuration (Clustering) ==="
echo "  nvcc        : $NVCC"
echo "  CUDA include: $CUDA_INCLUDE"
echo "  cudart      : ${CUDART_LIB:-NOT FOUND}"
echo "  curand      : ${CURAND_LIB:-NOT FOUND}"
echo "============================================"

# ── 컴파일 옵션 ───────────────────────────────────────────────────
ARCH="-arch=sm_86"
CXX=g++
CXXFLAGS="-O3 -std=c++17 -fopenmp"
CCBIN_FLAG=""
if command -v g++-10 >/dev/null 2>&1; then
    CCBIN_FLAG="-ccbin g++-10"
fi

NVCCFLAGS="-O3 --std=c++14 $ARCH $CCBIN_FLAG --expt-relaxed-constexpr -Xcompiler -fopenmp"

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

# ── 빌드 디렉토리 생성 ─────────────────────────────────────────────
mkdir -p build_gpu

# ── GPU 라이브러리 검사 및 빌드 ───────────────────────────────────
if [ ! -f "build_gpu/libmppi_gpu.a" ]; then
    echo "libmppi_gpu.a가 없습니다. GPU 라이브러리를 먼저 빌드합니다..."
    echo "Compiling mppi_gpu.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/mppi_gpu.cu -o build_gpu/mppi_gpu.o

    echo "Compiling cluster_mppi_gpu.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/cluster_mppi_gpu.cu -o build_gpu/cluster_mppi_gpu.o

    echo "Compiling bi_mppi_gpu.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/bi_mppi_gpu.cu -o build_gpu/bi_mppi_gpu.o

    echo "Compiling svgd_mppi_gpu.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c mppi/cuda/svgd_mppi_gpu.cu -o build_gpu/svgd_mppi_gpu.o

    GPU_OBJS="build_gpu/mppi_gpu.o build_gpu/cluster_mppi_gpu.o build_gpu/bi_mppi_gpu.o build_gpu/svgd_mppi_gpu.o"
    ar rcs build_gpu/libmppi_gpu.a $GPU_OBJS
    echo "  → build_gpu/libmppi_gpu.a 생성 완료"
fi

# ── 실행 파일 링크 함수 ────────────────────────────────────────────
build_target() {
    local SRC="$1"
    local NAME=$(basename "${SRC%.cpp}")
    echo "Building $NAME ..."
    $CXX $CXXFLAGS $INCLUDES "$SRC" \
        -Lbuild_gpu -lmppi_gpu \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "build_gpu/$NAME"
    echo "  → build_gpu/$NAME 완료"
}

# ── Clustering Parameter 디렉토리 내 모든 cpp 빌드 ────────────────
CLUSTERING_DIR="src/wmrobot/clustering parameter"

if [ -d "$CLUSTERING_DIR" ]; then
    for cpp_file in "$CLUSTERING_DIR"/*.cpp; do
        if [ -f "$cpp_file" ]; then
            build_target "$cpp_file"
        fi
    done
else
    echo "ERROR: 디렉토리를 찾을 수 없습니다: $CLUSTERING_DIR"
    exit 1
fi

echo ""
echo "=== Clustering Parameter 빌드 완료 ==="
