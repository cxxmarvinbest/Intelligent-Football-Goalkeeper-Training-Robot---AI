#!/bin/bash
# ============================================================
# 足球AI智能训练系统 — systemd 自启动安装脚本
# ============================================================
# 用法 (在 RK3588 板卡上通过 SSH 执行):
#   chmod +x install_service.sh
#   sudo ./install_service.sh
#
# 或一键安装:
#   curl -sSL http://.../install_service.sh | sudo bash
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVICE_NAME="football-ai"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
APP_DIR="/home/ztl/code"
HTTP_PORT=8098

echo "============================================"
echo "  足球AI智能训练系统 — 自启动安装"
echo "============================================"
echo ""

# ── 1. 检查是否为 root ──
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}[错误] 请使用 sudo 运行此脚本${NC}"
    echo "  sudo ./install_service.sh"
    exit 1
fi

# ── 2. 检查工作目录 ──
if [ ! -d "$APP_DIR" ]; then
    echo -e "${YELLOW}[警告] 工作目录不存在: $APP_DIR${NC}"
    echo "  请确认代码已部署到正确路径，或修改脚本中的 APP_DIR 变量。"
    read -p "  是否继续安装? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "  已取消。"
        exit 0
    fi
fi

# ── 3. 检查 main.py 是否存在 ──
if [ ! -f "$APP_DIR/main.py" ]; then
    echo -e "${RED}[错误] main.py 未找到: $APP_DIR/main.py${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] main.py 已就绪${NC}"

# ── 4. 检查 Python 环境 ──
PYTHON_BIN="/usr/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    echo -e "${YELLOW}[警告] Python3 未找到: $PYTHON_BIN${NC}"
    echo "  请确认 Python 路径并修改 service 文件中的 ExecStart。"
fi
echo -e "${GREEN}[OK] Python: $($PYTHON_BIN --version 2>&1 || echo '未知')${NC}"

# ── 5. 安装 service 文件 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_SERVICE="$SCRIPT_DIR/football-ai.service"

if [ ! -f "$SOURCE_SERVICE" ]; then
    echo -e "${RED}[错误] football-ai.service 未找到，请确认与 install_service.sh 在同一目录${NC}"
    exit 1
fi

cp "$SOURCE_SERVICE" "$SERVICE_FILE"
echo -e "${GREEN}[OK] service 文件已安装: $SERVICE_FILE${NC}"

# ── 6. 重载 systemd ──
systemctl daemon-reload
echo -e "${GREEN}[OK] systemd 已重载${NC}"

# ── 7. 启用开机自启 ──
systemctl enable "$SERVICE_NAME"
echo -e "${GREEN}[OK] 已启用开机自启${NC}"

# ── 8. 询问是否立即启动 ──
read -p "  是否立即启动服务? [Y/n] " start_now
if [ "$start_now" != "n" ] && [ "$start_now" != "N" ]; then
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl start "$SERVICE_NAME"
    sleep 3

    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}[OK] 服务已启动${NC}"
    else
        echo -e "${RED}[失败] 服务启动失败，请检查日志:${NC}"
        echo "  sudo journalctl -u $SERVICE_NAME -n 30 --no-pager"
    fi
fi

echo ""
echo "============================================"
echo -e "  ${GREEN}安装完成！${NC}"
echo ""
echo "  常用命令:"
echo "    sudo systemctl status $SERVICE_NAME     # 查看状态"
echo "    sudo systemctl start  $SERVICE_NAME     # 启动"
echo "    sudo systemctl stop   $SERVICE_NAME     # 停止"
echo "    sudo systemctl restart $SERVICE_NAME    # 重启"
echo "    sudo journalctl -u $SERVICE_NAME -f     # 实时日志"
echo "    sudo journalctl -u $SERVICE_NAME -n 50  # 最近50行日志"
echo ""
echo "  HTTP API 地址:"
echo "    http://192.168.8.75:8098/status"
echo "    http://192.168.8.75:8098/docs    (Swagger UI)"
echo "============================================"
