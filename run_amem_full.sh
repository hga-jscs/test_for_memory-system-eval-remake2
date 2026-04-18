#!/bin/bash
set -e
cd /Users/bytedance/proj/AgeMem

echo "=== A-MEM 全量评测 (串行) ==="
echo "开始时间: $(date)"

echo ""
echo ">>> [1/3] AMemGym (~5.5h)"
PYTHONUNBUFFERED=1 python3 bench_amemgym_amem.py

echo ""
echo ">>> [2/3] memory-probe (~3.7h)"
PYTHONUNBUFFERED=1 python3 bench_memory_probe_amem.py

echo ""
echo ">>> [3/3] StructMemEval (~4.2h)"
PYTHONUNBUFFERED=1 python3 bench_structmemeval_amem.py

echo ""
echo "=== 全部完成: $(date) ==="
