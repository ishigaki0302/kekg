#!/bin/bash
# =============================================================================
# Script   : tests/run_all_tests.sh
# 概要     : 全テストを pytest で実行する
# 使用方法 : ./tests/run_all_tests.sh
# 前提条件 : pip install pytest
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== Running SRO Platform Tests ==="
python -m pytest tests/ -v
