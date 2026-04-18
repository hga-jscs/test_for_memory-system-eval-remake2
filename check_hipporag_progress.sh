#!/bin/bash
echo "=== HippoRAG benchmark 进度 $(date) ==="
echo ""
echo "--- AMemGym ---"
grep "^\[" logs_amemgym_hipporag.txt 2>/dev/null | tail -5
echo ""
echo "--- memory-probe ---"
grep "^\[" logs_memory_probe_hipporag.txt 2>/dev/null | tail -5
echo ""
echo "--- StructMemEval ---"
grep "^\[" logs_structmemeval_hipporag.txt 2>/dev/null | tail -5
echo ""
echo "--- 运行中进程 ---"
ps aux | grep "bench_.*hipporag\|run_hipporag" | grep -v grep | awk '{print $11, $12}'
