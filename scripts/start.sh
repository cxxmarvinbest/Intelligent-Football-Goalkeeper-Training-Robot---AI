#!/bin/bash
# ============================================================
# 足球AI智能训练系统 — 启动脚本
# ============================================================
# 用途: 手动启动或作为 systemd 服务的启动入口
# ============================================================

set -e

# ── 工作目录 ──
APP_DIR="/home/ztl/code"
cd "$APP_DIR"

# ── Python 环境 (按需启用) ──
# 选项 1: conda 环境
# source /home/ztl/miniconda3/etc/profile.d/conda.sh
# conda activate rknn

# 选项 2: venv 虚拟环境
# source /home/ztl/venv/bin/activate

# ── 设置 MPP 模块路径 ──
export PYTHONPATH="/home/ztl/code/mpp:${PYTHONPATH:-}"

# ── 设置 RKNN 动态库路径 ──
export LD_LIBRARY_PATH="/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

# ── Python 实时输出 (不缓冲) ──
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  足球AI智能训练系统启动中..."
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  目录: $APP_DIR"
echo "============================================"

# ── 启动主程序 ──
exec /usr/bin/python3 "$APP_DIR/main.py"
