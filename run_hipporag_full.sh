#!/bin/bash
cd /Users/bytedance/proj/AgeMem

echo "[$(date)] AMemGym: 等待已有进程结束（已在 PID 90345 运行）..."
wait 90345 2>/dev/null || true

echo "[$(date)] AMemGym 完成，开始 memory-probe..."
python3 bench_memory_probe_hipporag.py > logs_memory_probe_hipporag.txt 2>&1
echo "[$(date)] memory-probe 完成，开始 StructMemEval..."
python3 bench_structmemeval_hipporag.py > logs_structmemeval_hipporag.txt 2>&1
echo "[$(date)] 全部完成！"
