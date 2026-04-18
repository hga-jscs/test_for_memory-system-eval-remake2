#!/bin/bash
set -e
cd /Users/bytedance/proj/AgeMem
echo "=== RAPTOR 全量评测 (串行) ==="
echo "开始时间: $(date)"
echo ">>> [1/3] AMemGym"
PYTHONUNBUFFERED=1 python3 bench_amemgym_raptor.py
echo ">>> [2/3] memory-probe"
PYTHONUNBUFFERED=1 python3 bench_memory_probe_raptor.py
echo ">>> [3/3] StructMemEval"
PYTHONUNBUFFERED=1 python3 bench_structmemeval_raptor.py
echo "=== 全部完成: $(date) ==="
