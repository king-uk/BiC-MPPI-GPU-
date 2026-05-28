#!/usr/bin/env bash
# ============================================================
# run_gpu_all.sh — 모든 GPU 예제 순차 실행 스크립트
#
# build_gpu 폴더 내의 실행 가능한 모든 GPU 예제를
# 자동으로 실행합니다.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/build_gpu" || { echo "build_gpu 폴더가 없습니다. 먼저 ./build_gpu.sh 를 실행하세요."; exit 1; }

echo "=== Running all GPU examples ==="

# 실행할 GPU 예제 목록
EXAMPLES=("mppi" "cluster_mppi" "bi_mppi" "svgd_mppi" "log_mppi")

for EX in "${EXAMPLES[@]}"; do
    if [ -x "./$EX" ]; then
        echo ""
        echo "--------------------------------------------------"
        echo " 🚀 실행 중: ./$EX"
        echo "--------------------------------------------------"
        ./$EX
    else
        echo ""
        echo "⚠️  실행 파일이 없습니다: ./$EX (빌드 시 주석 처리됨)"
    fi
done

echo ""
echo "=== 모든 GPU 예제 실행 완료 ==="
