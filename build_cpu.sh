#!/usr/bin/env bash
# ============================================================
# build_cpu.sh — BiC-MPPI CPU 버전 빌드 스크립트
#
# CUDA 의존성 없이 g++만으로 순수 CPU 예제들을 빌드합니다.
#
# 사용법:
#   bash build_cpu.sh
#
# 하단의 "빌드할 소스 파일 선택" 부분에서 원하는 예제의
# 주석을 해제하여 빌드 대상을 조절할 수 있습니다.
# ============================================================
set -e

# ── 경로 설정 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 컴파일 옵션 ───────────────────────────────────────────────────
CXX=g++
CXXFLAGS="-O3 -std=c++17 -fopenmp"

EIGEN_INC=$(pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")

INCLUDES="-I./mppi -I./model \
          -I./include/EigenRand -I./include/matplotlibcpp \
          $(python3-config --includes) \
          $EIGEN_INC"

LDFLAGS="-fopenmp $(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags)"

# ── 빌드 디렉토리 ─────────────────────────────────────────────────
mkdir -p build_cpu

# ── 실행 파일 링크 ────────────────────────────────────────────────
build_target() {
    local SRC="$1"
    local NAME=$(basename "${SRC%.cpp}")
    echo "Building $NAME ..."
    $CXX $CXXFLAGS $INCLUDES "$SRC" \
        $LDFLAGS \
        -o "build_cpu/$NAME"
    echo "  → build_cpu/$NAME 완료"
}

# ===========================================================================
# 빌드할 소스 파일 선택 (사용할 예제의 주석을 해제하세요)
# ===========================================================================

echo "=== CPU Build Configuration ==="
echo "  Compiler : $CXX"
echo "  Flags    : $CXXFLAGS"
echo "================================"

# ---- WMRobot CPU 버전 ----
build_target src/wmrobot/cpu/mppi.cpp
# build_target src/wmrobot/cpu/cluster_mppi.cpp
# build_target src/wmrobot/cpu/bi_mppi.cpp
# build_target src/wmrobot/cpu/log_mppi.cpp
# build_target src/wmrobot/cpu/svgd_mppi.cpp
# build_target src/wmrobot/cpu/rrt_connect.cpp

echo ""
echo "=== 빌드 완료 ==="
echo "실행: cd build_cpu && ./mppi"
