#!/bin/bash
# ============================================================
# run_all.sh  –  wmrobot 벤치마크 전체 실행 스크립트
# 프로젝트 루트에서 실행 가능: bash run_all.sh
#
# 주의: 각 실행 파일은 "../BARN_dataset/..." 상대경로로 맵을 로드하므로
#       working directory를 build/ 로 바꿔서 실행합니다.
#       결과 CSV도 build/ 안에 생성됩니다.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "[ERROR] build/ directory not found: $BUILD_DIR"
    exit 1
fi

SOLVERS=(mppi log_mppi cluster_mppi bi_mppi svgd_mppi)

# 순차 실행 여부 (1=순차, 0=병렬)
SEQUENTIAL=${SEQUENTIAL:-1}

echo "======================================"
echo "  WMRobot MPPI Benchmark Runner"
echo "  build dir : $BUILD_DIR"
echo "  mode      : $([ $SEQUENTIAL -eq 1 ] && echo sequential || echo parallel)"
echo "======================================"

run_solver() {
    local name=$1
    local exe="$BUILD_DIR/$name"

    if [ ! -f "$exe" ]; then
        echo "[SKIP] $name not built (not found: $exe)"
        return
    fi

    local log_file="$BUILD_DIR/log_${name}.txt"
    echo "[START] $name  →  build/result_${name}.csv"

    local t_start=$(date +%s%3N)

    # working directory를 build/로 변경 후 실행
    (cd "$BUILD_DIR" && ./"$name") 2>&1 | tee "$log_file"

    local t_end=$(date +%s%3N)
    echo "[DONE ] $name  – $(( t_end - t_start )) ms"
    echo "--------------------------------------"
}

if [ $SEQUENTIAL -eq 1 ]; then
    for s in "${SOLVERS[@]}"; do
        run_solver "$s"
    done
else
    PIDS=()
    for s in "${SOLVERS[@]}"; do
        run_solver "$s" &
        PIDS+=($!)
    done
    echo "Waiting for all solvers..."
    for pid in "${PIDS[@]}"; do wait "$pid"; done
fi

echo ""
echo "======================================"
echo "  Results:"
for s in "${SOLVERS[@]}"; do
    f="$BUILD_DIR/result_${s}.csv"
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  build/result_${s}.csv  ($((lines-1)) rows)"
    fi
done
echo "======================================"
