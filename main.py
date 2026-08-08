# -*- coding: utf-8 -*-
"""
足球AI智能训练系统 — 主线程入口 (11米点位)
==========================================
部署位置: /home/ztl/code/main.py

模块职责:
  - yolov8_detection.py  → 纯检测引擎 (数据/算法/模型)
  - mqtt.py              → MQTT 通信 (MqttClient + MotorDriver + JSON 配置)
  - http_api.py          → HTTP API (FastAPI 路由)
  - main.py              → 主线程入口 (对象创建 + 依赖注入 + 启动编排)

使用方式:
  # 手动启动 (调试用)
  python main.py

  # 开机自启 (生产环境)
  # 1. 将 scripts/football_ai.service 复制到 /etc/systemd/system/
  # 2. sudo systemctl daemon-reload
  # 3. sudo systemctl enable football_ai
  # 4. sudo systemctl start football_ai
  # 或直接运行: sudo scripts/install_service.sh (一键安装)
"""

import os
import sys
import time

# ── 确保当前目录在搜索路径中 ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── 算法模块 (含 MPP 硬件解码) ──
from yolov8_detection import (
    DetectionEngine, cfg_get, MODEL_PATH,
    RKNN_NPU_CORE, RKNN_INPUT_SIZE,
    GOAL_GRID_ROWS, GOAL_GRID_COLS,
    HAS_MPP, get_algo_yaml_path,
)

# ── MQTT 通信模块 ──
from mqtt import MqttClient, MotorDriver, MQTT_AVAILABLE, get_mqtt_config

# ── HTTP API 模块 ──
from http_api import (
    SystemState,
    setup_globals, run_server,
)

# ============================================================
# 主程序
# ============================================================
def main():
    # ── 1. 创建全局状态 ──
    system_state = SystemState()
    print("[main] SystemState 已创建")

    # ── 2. 创建 MQTT 客户端并连接 ──
    mqtt = MqttClient()

    # ── 3. 创建发球机驱动 (引用 MQTT) ──
    motor_driver = MotorDriver(mqtt_client_ref=mqtt)
    print("[main] MotorDriver 已创建")

    # ── 4. 预加载检测引擎 ──
    print(f"[main] 加载 RKNN 模型: {MODEL_PATH}")
    engine = DetectionEngine(MODEL_PATH)
    if not engine.load():
        print("[main] ❌ RKNN 模型加载失败！")
        sys.exit(1)
    print("[main] ✅ DetectionEngine 已就绪")

    # ── 5. 注入全局依赖到 http_api 模块 ──
    setup_globals(system_state, mqtt, motor_driver, engine)

    # ── 6. 连接 MQTT (模型加载完成后再连, 避免竞态) ──
    mqtt.connect()

    # ── 7. 读取 HTTP 配置 ──
    http_host = cfg_get("http", "host", default="192.168.8.75")
    http_port = cfg_get("http", "port", default=8098)

    # ── 8. 打印启动信息 ──
    rtsp_url = cfg_get("camera", "rtsp_url",
                       default="rtsp://admin:siboasi123@192.168.8.142:554/LiveMedia/ch1/Media1")
    mqtt_host = cfg_get("mqtt", "broker_host", default="192.168.8.75")
    mqtt_port = cfg_get("mqtt", "broker_port", default=18883)
    mqtt_topic_tx = cfg_get("mqtt", "topic_tx", default="/SS/FB/DMT/001/AI/TX")
    print("")
    print("=" * 60)
    print("  智能足球守门员 AI智能模式 API v4.0 (11米点位)")
    print(f"  - RTSP:       {rtsp_url}")
    print(f"  - RKNN 模型:  {MODEL_PATH}")
    print(f"  - NPU core:   {RKNN_NPU_CORE}")
    print(f"  - 输入尺寸:   {RKNN_INPUT_SIZE}x{RKNN_INPUT_SIZE}")
    print(f"  - 网格:       {GOAL_GRID_ROWS}x{GOAL_GRID_COLS}={GOAL_GRID_ROWS*GOAL_GRID_COLS}分区")
    print(f"  - MPP 硬解:   {HAS_MPP}")
    print(f"  - MQTT:       {mqtt_host}:{mqtt_port} (TX={mqtt_topic_tx})")
    print(f"  - HTTP:       {http_host}:{http_port}")
    print(f"  - 配置:       {get_algo_yaml_path()}")
    print("=" * 60)
    print("")

    # ── 9. 启动 HTTP 服务器 (阻塞) ──
    run_server(host=http_host, port=http_port)


if __name__ == "__main__":
    main()
