# -*- coding: utf-8 -*-
"""
HTTP API 模块 — 足球AI智能训练 REST 接口
========================================
部署位置: /home/ztl/code/http_api.py

职责:
  1. FastAPI 路由 (open/close/start/pause/resume/stop/params/frames/status)
  2. SystemState — 全局状态管理 (线程安全)
  3. Pydantic 请求模型 + 参数校验
  4. 摄像头采集线程 + AI 算法处理线程 (线程管理, 不含解码/推理实现)

 本模块从 yolov8_detection 导入算法组件 (DetectionEngine + MppVideoSource + HAS_MPP),
 从 mqtt 导入通信组件 (MqttClient + MotorDriver)。
 全局引用通过 setup_globals() 由 main.py 注入。

 使用方式:
   import http_api
   http_api.setup_globals(state, mqtt_client, motor, detection_engine)
   http_api.run_server(host="192.168.8.75", port=8098)
"""

import os
import sys
import json
import time
import random
import threading
import queue
import datetime
import concurrent.futures
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import cv2

# FastAPI / HTTP
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging

# 自定义日志配置 — 用整数级别(如 logging.INFO=20)替代字符串 'INFO'
# 某些包(如 rknn-toolkit2 的依赖)会破坏 logging._nameToLevel 字典,
# 导致 uvicorn 默认 log_config 中的字符串 'INFO' 无法被 _checkLevel 解析
_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": logging.INFO, "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": logging.INFO, "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": logging.INFO, "propagate": False},
    },
    "root": {"handlers": ["default"], "level": logging.INFO},
}

# ── 算法模块导入 (含 MPP 硬件解码 + YOLO 检测引擎) ──
from yolov8_detection import (
    # 配置
    cfg_get, load_config,
    # 常量
    SERVE_MODE_REVERSE, SERVE_MODE_FOLLOW,
    HEIGHT_HIGH, HEIGHT_MID, HEIGHT_LOW,
    PRESET_RANDOM, PRESET_LEFT, PRESET_RIGHT,
    END_FREE, END_COUNT, END_DURATION,
    RUN_STATE_DEFAULT, RUN_STATE_RUNNING, RUN_STATE_PAUSED, RUN_STATE_ENDED,
    SAFE_DURATION_BEFORE_SERVE, SERVE_COOLDOWN,
    AUTO_END_TIMEOUT, TRACK_INTERVAL,
    SERVE_INTERVAL_MIN, SERVE_INTERVAL_MAX,
    SERVE_COUNT_MIN, SERVE_COUNT_MAX,
    DURATION_MIN, DURATION_MAX,
    DEFAULT_WHEEL_SPEED, MODEL_PATH,
    GOAL_GRID_ROWS, GOAL_GRID_COLS,
    COL_REVERSE_LEFT, COL_REVERSE_RIGHT,
    COL_CENTER_LEFT, COL_CENTER_RIGHT,
    # 引擎 + 数据结构
    DetectionEngine, DetectionResult, Detection, GoalBox,
    # 算法函数
    assemble_goal, smooth_goal_box, check_safety,
    find_gk_zone, find_gk_zone_by_overlap, get_gk_position,
    determine_serve_target, resolve_target_row,
    zone_number, zone_to_sdata, get_zone_grid,
    cv2_resize, cv2_resize_letterbox,
    # MPP 硬件解码
    MppVideoSource, HAS_MPP,
    # 配置路径
    get_config_dir, get_algo_yaml_path, get_device_json_path,
)

# ── MQTT 通信模块导入 ──
from mqtt import MqttClient, MotorDriver, MQTT_AVAILABLE, get_mqtt_config


# ============================================================
# API 响应助手
# ============================================================
def api_ok(data: Any = None, msg: str = "") -> Dict:
    return {"code": 200, "msg": msg, "data": data}

def api_err(code: int = 500, msg: str = "", data: Any = None) -> Dict:
    return {"code": code, "msg": msg, "data": data}


# ============================================================
# SystemState — 全局状态（线程安全）
# ============================================================
class SystemState:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.is_paused = False
        self.auto_paused = False
        self.algorithm_thread_running = False
        self.camera_started = False
        self.is_safe = False
        self.safety_reason = "系统未启动"
        self.safety_detail: Dict = {}
        self.serve_count = 0
        self.last_serve_pid = 0
        self.safe_start_time = 0.0
        self.last_serve_time = 0.0
        self.serve_triggered = False
        self.last_serve_data: Optional[List[int]] = None
        self.serve_mode = SERVE_MODE_REVERSE
        self.preset_direction = PRESET_RANDOM
        self.height = HEIGHT_MID
        self.serve_interval = 5
        self.end_condition = END_FREE
        self.serve_count_limit = 10
        self.duration_limit = 180
        self.start_time = 0.0
        self.pending_params: Optional[Dict] = None
        self.target_zone = 0
        self.target_row = -1
        self.target_col = -1
        self.goal_detected = False
        self.gk_detected = False
        self.gk_side = None
        self.frame_shape = (0, 0)

    def get_device_state(self) -> str:
        with self.lock:
            if not self.is_running and not self.is_paused:
                return "stopped" if self.serve_count > 0 else "idle"
            if self.is_paused:
                return "paused"
            return "running"

    def set_safe(self, is_safe: bool, reason: str, detail: Dict = None):
        with self.lock:
            prev_safe = self.is_safe
            self.is_safe = is_safe
            self.safety_reason = reason
            self.safety_detail = detail or {}
            if is_safe and not prev_safe:
                self.safe_start_time = time.time()
                self.serve_triggered = False
            elif not is_safe:
                self.safe_start_time = 0.0
                self.serve_triggered = False

    def get_safe_duration(self) -> float:
        with self.lock:
            if not self.is_safe or self.safe_start_time <= 0:
                return 0.0
            return time.time() - self.safe_start_time

    def should_serve(self) -> bool:
        with self.lock:
            if not self.is_safe:
                return False
            if self.serve_triggered:
                return False
            if time.time() - self.safe_start_time < SAFE_DURATION_BEFORE_SERVE:
                return False
            if time.time() - self.last_serve_time < self.serve_interval:
                return False
            return True

    def record_serve(self, pid: int, w1: int, w2: int, w3: int,
                     h_angle: int, v_angle: int):
        with self.lock:
            self.serve_count += 1
            self.last_serve_pid = pid
            self.last_serve_time = time.time()
            self.serve_triggered = True
            self.last_serve_data = [w1, w2, w3, h_angle, v_angle]

    def check_end_condition(self) -> bool:
        with self.lock:
            if self.end_condition == END_COUNT:
                if self.serve_count >= self.serve_count_limit:
                    return True
            elif self.end_condition == END_DURATION:
                if (time.time() - self.start_time) >= self.duration_limit:
                    return True
            return False

    def apply_pending_params(self):
        with self.lock:
            if self.pending_params is None:
                return False
            p = self.pending_params
            if "serve_mode" in p:
                self.serve_mode = p["serve_mode"]
            if "height" in p:
                self.height = p["height"]
            if "preset_direction" in p:
                self.preset_direction = p["preset_direction"]
            if "serve_interval" in p:
                self.serve_interval = p["serve_interval"]
            if "end_condition" in p:
                self.end_condition = p["end_condition"]
            if "serve_count_limit" in p:
                self.serve_count_limit = p["serve_count_limit"]
            if "duration_limit" in p:
                self.duration_limit = p["duration_limit"]
            self.pending_params = None
            print("[System] 延迟参数已生效")
            return True

    def get_snapshot(self) -> Dict:
        with self.lock:
            device_state = "idle"
            if not self.is_running and not self.is_paused:
                device_state = "idle" if self.serve_count == 0 else "stopped"
            elif self.is_paused:
                device_state = "paused"
            else:
                device_state = "running"
            return {
                "device_state":      device_state,
                "camera_state":      "running" if self.camera_started else "stopped",
                "serve_count":       self.serve_count,
                "is_safe":           self.is_safe,
                "safety_reason":     self.safety_reason,
                "safety_detail":     dict(self.safety_detail),
                "safe_duration":     round(
                    time.time() - self.safe_start_time
                    if self.is_safe and self.safe_start_time > 0 else 0, 1),
                "serve_mode":        self.serve_mode,
                "preset_direction":  self.preset_direction,
                "height":            self.height,
                "serve_interval":    self.serve_interval,
                "end_condition":     self.end_condition,
                "serve_count_limit": self.serve_count_limit,
                "duration_limit":    self.duration_limit,
                "target_zone":       self.target_zone,
                "target_row":        self.target_row,
                "target_col":        self.target_col,
                "goal_detected":     self.goal_detected,
                "gk_detected":       self.gk_detected,
                "gk_side":           self.gk_side,
                "pending_params":    self.pending_params is not None,
            }


# ============================================================
# Pydantic 请求模型
# ============================================================
class StartRequest(BaseModel):
    serve_mode: int
    height: int
    preset_direction: int
    serve_interval: int
    end_condition: int
    serve_count_limit: Optional[int] = None
    duration_limit: Optional[int] = None


class UpdateParamsRequest(BaseModel):
    serve_mode: Optional[int] = None
    height: Optional[int] = None
    preset_direction: Optional[int] = None
    serve_interval: Optional[int] = None
    end_condition: Optional[int] = None
    serve_count_limit: Optional[int] = None
    duration_limit: Optional[int] = None


# ============================================================
# 参数校验
# ============================================================
def validate_start_params(req: StartRequest) -> Tuple[bool, Optional[Dict]]:
    errors = []
    if req.serve_mode not in (SERVE_MODE_REVERSE, SERVE_MODE_FOLLOW):
        errors.append(f"serve_mode 必须为 0(反向)/1(跟随)，当前: {req.serve_mode}")
    if req.height not in (HEIGHT_HIGH, HEIGHT_MID, HEIGHT_LOW):
        errors.append(f"height 必须为 0(高)/1(中)/2(低)，当前: {req.height}")
    if req.preset_direction not in (PRESET_RANDOM, PRESET_LEFT, PRESET_RIGHT):
        errors.append(f"preset_direction 必须为 0(随机)/1(左)/2(右)，当前: {req.preset_direction}")
    if not (SERVE_INTERVAL_MIN <= req.serve_interval <= SERVE_INTERVAL_MAX):
        errors.append(f"serve_interval 必须 {SERVE_INTERVAL_MIN}~{SERVE_INTERVAL_MAX}")
    if req.end_condition not in (END_FREE, END_COUNT, END_DURATION):
        errors.append(f"end_condition 必须 0/1/2")
    if req.end_condition == END_COUNT:
        if req.serve_count_limit is None or not (SERVE_COUNT_MIN <= req.serve_count_limit <= SERVE_COUNT_MAX):
            errors.append(f"serve_count_limit 必须 {SERVE_COUNT_MIN}~{SERVE_COUNT_MAX}")
    if req.end_condition == END_DURATION:
        if req.duration_limit is None or not (DURATION_MIN <= req.duration_limit <= DURATION_MAX):
            errors.append(f"duration_limit 必须 {DURATION_MIN}~{DURATION_MAX}")
    if errors:
        return False, {"errors": errors}
    return True, None


def validate_update_params(params: UpdateParamsRequest) -> Tuple[bool, Optional[Dict]]:
    errors = []
    if params.serve_mode is not None and params.serve_mode not in (SERVE_MODE_REVERSE, SERVE_MODE_FOLLOW):
        errors.append(f"serve_mode 必须为 0/1，当前: {params.serve_mode}")
    if params.height is not None and params.height not in (HEIGHT_HIGH, HEIGHT_MID, HEIGHT_LOW):
        errors.append(f"height 必须 0(高)/1(中)/2(低)")
    if params.preset_direction is not None and params.preset_direction not in (PRESET_RANDOM, PRESET_LEFT, PRESET_RIGHT):
        errors.append(f"preset_direction 必须 0/1/2")
    if params.serve_interval is not None and not (SERVE_INTERVAL_MIN <= params.serve_interval <= SERVE_INTERVAL_MAX):
        errors.append(f"serve_interval 必须 {SERVE_INTERVAL_MIN}~{SERVE_INTERVAL_MAX}")
    if params.end_condition is not None and params.end_condition not in (END_FREE, END_COUNT, END_DURATION):
        errors.append(f"end_condition 必须 0/1/2")
    if params.serve_count_limit is not None and not (SERVE_COUNT_MIN <= params.serve_count_limit <= SERVE_COUNT_MAX):
        errors.append(f"serve_count_limit 必须 {SERVE_COUNT_MIN}~{SERVE_COUNT_MAX}")
    if params.duration_limit is not None and not (DURATION_MIN <= params.duration_limit <= DURATION_MAX):
        errors.append(f"duration_limit 必须 {DURATION_MIN}~{DURATION_MAX}")
    if errors:
        return False, {"errors": errors}
    return True, None


# ============================================================
# 全局引用 — 由 main.py 通过 setup_globals() 注入
# ============================================================
state: Optional[SystemState] = None
mqtt_client: Optional[MqttClient] = None
motor: Optional[MotorDriver] = None
detection_engine: Optional[DetectionEngine] = None
video_source: Optional[MppVideoSource] = None

# 帧队列 & 最新帧 (供摄像头线程和 /frames 路由共享)
frame_queue: queue.Queue = queue.Queue(maxsize=1)
latest_frame: Optional[np.ndarray] = None
latest_frame_lock = threading.Lock()
_frame_read_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# 摄像头参数 (从配置读取)
_RTSP_URL = cfg_get("camera", "rtsp_url",
                    default="rtsp://admin:siboasi123@192.168.8.142:554/LiveMedia/ch1/Media1")
_DISPLAY_W = cfg_get("camera", "frame_scale_width", default=1280)
_DISPLAY_H = cfg_get("camera", "frame_scale_height", default=720)
_IS_RGB = cfg_get("camera", "output_rgb", default=False)

# JPEG 参数
JPEG_QUALITY = cfg_get("camera", "jpeg_quality", default=85)
VIDEO_IMAGE_MAX_SIZE = cfg_get("camera", "video_image_max_size", default=0)
VIDEO_IMAGE_MAX_SIZE_LIMIT = VIDEO_IMAGE_MAX_SIZE > 0


def setup_globals(_state: SystemState, _mqtt: MqttClient,
                  _motor: MotorDriver, _engine: DetectionEngine):
    """由 main.py 调用, 注入所有全局依赖"""
    global state, mqtt_client, motor, detection_engine
    state = _state
    mqtt_client = _mqtt
    motor = _motor
    detection_engine = _engine
    print("[http_api] 全局依赖已注入")


# ============================================================
# 线程1：MPP 摄像头采集
# ============================================================
def camera_capture_thread():
    global latest_frame, video_source

    if not HAS_MPP:
        print("[Camera] MPP 不可用，摄像头线程退出")
        state.camera_started = False
        return

    video_source = MppVideoSource(
        rtsp_url=_RTSP_URL,
        display_width=_DISPLAY_W,
        display_height=_DISPLAY_H,
        is_rgb=_IS_RGB,
    )
    if not video_source.start():
        print("[Camera] MPP 视频源启动失败")
        state.camera_started = False
        return

    print(f"[Camera] MPP 硬件解码已启动 ({_DISPLAY_W}x{_DISPLAY_H})")

    while state.camera_started:
        frame = video_source.get_frame()
        if frame is not None:
            with latest_frame_lock:
                latest_frame = frame.copy()
            # 写入算法队列（仅运行时）
            if state.is_running:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame)
        else:
            time.sleep(0.005)

    if video_source:
        video_source.stop()
        video_source = None
    print("[Camera] 摄像头已释放")


def start_camera():
    if state.camera_started:
        return False
    state.camera_started = True
    threading.Thread(target=camera_capture_thread, daemon=True).start()
    return True


def stop_camera():
    global latest_frame
    if not state.camera_started:
        return False
    state.camera_started = False
    with latest_frame_lock:
        latest_frame = None
    return True


# ============================================================
# 线程2：AI 检测 + 安全判定 + 自动发球
# ============================================================
def algorithm_processor_thread():
    global detection_engine

    frame_idx = 0
    if detection_engine is None:
        print("[AI] 检测引擎未初始化，算法线程退出")
        state.algorithm_thread_running = False
        return
    if not detection_engine.is_loaded():
        if not detection_engine.load():
            print("[AI] 模型加载失败，算法线程退出")
            state.algorithm_thread_running = False
            return

    state.algorithm_thread_running = True
    print("[AI] 算法线程已启动")

    try:
        while state.is_running:
            # 检查自动暂停超时
            if state.is_paused and mqtt_client.check_pause_timeout():
                with state.lock:
                    state.is_running = False
                    state.is_paused = False
                motor.stop_motor()
                mqtt_client.clear_pause_timer()
                mqtt_client.publish_state_change(RUN_STATE_ENDED)
                print(f"[System] 暂停超时（{AUTO_END_TIMEOUT}秒），自动结束")
                break

            # 取帧
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            frame_h, frame_w = frame.shape[:2]

            # 检测
            if frame_idx % TRACK_INTERVAL == 0:
                result = detection_engine.process_frame(frame, force_detect=True)
            else:
                result = detection_engine.process_frame(frame, force_detect=False)

            state.set_safe(result.is_safe, result.safety_reason, result.safety_detail)

            # 更新状态
            with state.lock:
                state.goal_detected = result.smoothed_goal is not None
                state.gk_detected = result.goalkeeper is not None
                state.frame_shape = (frame_h, frame_w)
                state.gk_side = result.gk_side

            # 自动发球
            if not state.is_paused and state.should_serve():
                target_row = resolve_target_row(state.height)

                if state.serve_mode == SERVE_MODE_FOLLOW and result.goalkeeper and result.smoothed_goal:
                    gk_row, gk_col, target_zone = find_gk_zone_by_overlap(
                        result.goalkeeper, result.smoothed_goal)
                    if target_zone > 0:
                        target_row = gk_row
                        target_col = gk_col
                    else:
                        target_col = random.randint(COL_CENTER_LEFT, COL_CENTER_RIGHT)
                        target_zone = zone_number(target_row, target_col)
                elif result.goalkeeper and result.smoothed_goal and result.gk_side:
                    _, gk_overlaps = get_gk_position(result.goalkeeper, result.smoothed_goal)
                    target_col, target_zone = determine_serve_target(
                        state.serve_mode, result.gk_side, gk_overlaps,
                        state.preset_direction, target_row)
                else:
                    target_col = COL_REVERSE_RIGHT
                    target_zone = zone_number(target_row, target_col)

                sdata = zone_to_sdata(target_row, target_col)
                h_angle, v_angle = sdata[3], sdata[4]

                with state.lock:
                    state.target_zone = target_zone
                    state.target_row = target_row
                    state.target_col = target_col

                pid = motor.serve(
                    wheel1=DEFAULT_WHEEL_SPEED,
                    wheel2=DEFAULT_WHEEL_SPEED,
                    wheel3=DEFAULT_WHEEL_SPEED,
                    h_angle=h_angle, v_angle=v_angle,
                )
                state.record_serve(pid, DEFAULT_WHEEL_SPEED, DEFAULT_WHEEL_SPEED,
                                   DEFAULT_WHEEL_SPEED, h_angle, v_angle)
                print(f"[Serve] 自动发球 #{state.serve_count} "
                      f"(PID={pid}) Zone={target_zone} SDATA={sdata}")

                mqtt_client.publish_serve_result(
                    success=True,
                    msg_text=f"发球成功 Zone={target_zone}",
                    serve_count=state.serve_count,
                    sdata=[DEFAULT_WHEEL_SPEED, DEFAULT_WHEEL_SPEED,
                           DEFAULT_WHEEL_SPEED, h_angle, v_angle],
                )

                state.apply_pending_params()

                if state.check_end_condition():
                    with state.lock:
                        state.is_paused = True
                        state.auto_paused = True
                    motor.pause_motor()
                    mqtt_client.start_pause_timer()
                    mqtt_client.publish_state_change(RUN_STATE_PAUSED)
                    print(f"[System] 达到结束条件，自动暂停")

            frame_idx += 1
    finally:
        state.algorithm_thread_running = False
        print("[AI] 算法线程已停止")


# ============================================================
# FastAPI 应用 & 路由
# ============================================================
app = FastAPI(
    title="足球AI智能训练 — 11米点位 API",
    description="RK3588 NPU 推理 + MPP 硬件解码 + HTTP/MQTT 双通道控制",
    version="4.1",
)

# ── CORS 中间件 — 允许跨域请求 (APP/Web前端) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 1. 打开摄像头 ──
@app.post("/open", summary="打开摄像头", tags=["摄像头"])
def api_open():
    if state.camera_started:
        return api_ok({"camera_state": "running"}, "摄像头已打开")
    start_camera()
    print("[System] 摄像头已打开")
    return api_ok({"camera_state": "running"}, "摄像头已打开")


# ── 1b. 关闭摄像头 ──
@app.post("/close", summary="关闭摄像头", tags=["摄像头"])
def api_close():
    if not state.camera_started:
        return api_ok({"camera_state": "stopped"}, "摄像头未打开")
    stop_camera()
    print("[System] 摄像头已关闭")
    return api_ok({"camera_state": "stopped"}, "摄像头已关闭")


# ── 2. 开始 ──
@app.post("/start", summary="开始", tags=["核心控制"])
def api_start(req: StartRequest):
    valid, detail = validate_start_params(req)
    if not valid:
        return api_err(500, "参数错误，请重新上传", detail)

    if not state.camera_started:
        start_camera()
        time.sleep(0.5)

    if not motor.connected:
        motor.connect()
    motor.start_motor()
    time.sleep(0.3)

    with state.lock:
        state.serve_mode = req.serve_mode
        state.height = req.height
        state.preset_direction = req.preset_direction
        state.serve_interval = req.serve_interval
        state.end_condition = req.end_condition
        state.serve_count_limit = req.serve_count_limit
        state.duration_limit = req.duration_limit
        state.is_running = True
        state.is_paused = False
        state.start_time = time.time()
        state.serve_count = 0
        state.serve_triggered = False
        state.safe_start_time = 0.0
        state.pending_params = None

    if not state.algorithm_thread_running:
        threading.Thread(target=algorithm_processor_thread, daemon=True).start()

    mqtt_client.clear_pause_timer()
    mqtt_client.publish_state_change(RUN_STATE_RUNNING)

    print("[System] 系统已启动")
    return api_ok({
        "device_state": state.get_device_state(),
        "serve_mode": state.serve_mode,
        "height": state.height,
        "preset_direction": state.preset_direction,
        "serve_interval": state.serve_interval,
        "end_condition": state.end_condition,
    }, "系统已启动")


# ── 3. 暂停 ──
@app.post("/pause", summary="暂停", tags=["核心控制"])
def api_pause():
    if not state.is_running:
        return api_err(500, "系统未启动，无法暂停")
    state.is_paused = True
    motor.pause_motor()
    mqtt_client.start_pause_timer()
    mqtt_client.publish_state_change(RUN_STATE_PAUSED)
    print("[System] 系统已暂停")
    return api_ok({"device_state": state.get_device_state()}, "系统已暂停")


# ── 4. 继续 ──
@app.post("/resume", summary="继续", tags=["核心控制"])
def api_resume(req: StartRequest):
    valid, detail = validate_start_params(req)
    if not valid:
        return api_err(500, "参数错误，请重新上传", detail)

    if not state.camera_started:
        start_camera()
        time.sleep(0.5)

    if not motor.connected:
        motor.connect()
    motor.start_motor()
    time.sleep(0.3)

    with state.lock:
        state.serve_mode = req.serve_mode
        state.height = req.height
        state.preset_direction = req.preset_direction
        state.serve_interval = req.serve_interval
        state.end_condition = req.end_condition
        state.serve_count_limit = req.serve_count_limit
        state.duration_limit = req.duration_limit
        state.is_running = True
        state.is_paused = False
        state.auto_paused = False
        state.serve_triggered = False
        state.safe_start_time = 0.0
        state.pending_params = None

    if not state.algorithm_thread_running:
        threading.Thread(target=algorithm_processor_thread, daemon=True).start()

    mqtt_client.clear_pause_timer()
    mqtt_client.publish_state_change(RUN_STATE_RUNNING)

    print("[System] 系统已从暂停恢复")
    return api_ok({
        "device_state": state.get_device_state(),
        "serve_mode": state.serve_mode,
        "height": state.height,
        "preset_direction": state.preset_direction,
        "serve_interval": state.serve_interval,
        "end_condition": state.end_condition,
    }, "系统已恢复运行")


# ── 5. 结束 ──
@app.post("/stop", summary="结束", tags=["核心控制"])
def api_stop():
    state.is_running = False
    state.is_paused = False
    state.camera_started = False

    motor.stop_motor()
    time.sleep(0.3)
    motor.disconnect()

    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

    mqtt_client.clear_pause_timer()
    mqtt_client.publish_state_change(RUN_STATE_ENDED)

    print("[System] 系统已停止")
    return api_ok({"device_state": "stopped"}, "系统已停止")


# ── 6. 变更参数 ──
@app.post("/params", summary="变更参数", tags=["参数配置"])
def api_update_params(params: UpdateParamsRequest):
    valid, detail = validate_update_params(params)
    if not valid:
        return api_err(500, "参数错误，请重新上传", detail)

    pending = {}
    if params.serve_mode is not None:
        pending["serve_mode"] = params.serve_mode
    if params.height is not None:
        pending["height"] = params.height
    if params.preset_direction is not None:
        pending["preset_direction"] = params.preset_direction
    if params.serve_interval is not None:
        pending["serve_interval"] = params.serve_interval
    if params.end_condition is not None:
        pending["end_condition"] = params.end_condition
    if params.serve_count_limit is not None:
        pending["serve_count_limit"] = params.serve_count_limit
    if params.duration_limit is not None:
        pending["duration_limit"] = params.duration_limit

    if not pending:
        return api_err(500, "参数错误", {"errors": ["未传入任何参数"]})

    with state.lock:
        if state.pending_params is None:
            state.pending_params = {}
        state.pending_params.update(pending)

    return api_ok({"pending_params": pending}, "参数变更已接收")


# ── 7. 查询最新图片 ──
@app.get("/frames", summary="查询摄像机最新图片", tags=["数据查询"])
async def api_get_frames():
    with latest_frame_lock:
        if latest_frame is None:
            return JSONResponse(
                content=api_err(500, "暂无画面数据（摄像头未启动）"),
                status_code=200, media_type="application/json")
        frame = latest_frame.copy()

    if VIDEO_IMAGE_MAX_SIZE_LIMIT:
        h, w = frame.shape[:2]
        max_side = max(w, h)
        if max_side > VIDEO_IMAGE_MAX_SIZE:
            ratio = max_side / VIDEO_IMAGE_MAX_SIZE
            new_w, new_h = int(w / ratio), int(h / ratio)
            frame = cv2_resize(frame, (new_w, new_h))

    ret, buffer = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return JSONResponse(
            content=api_err(500, "图片编码失败"),
            status_code=200, media_type="application/json")

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


# ── 8. 查询当前状态 ──
@app.get("/status", summary="查询当前状态", tags=["数据查询"])
def api_get_status():
    return api_ok(state.get_snapshot())


# ============================================================
# 启动入口
# ============================================================
def run_server(host: str = "192.168.8.75", port: int = 8098):
    """启动 HTTP API 服务器 (阻塞)"""
    print(f"[http_api] HTTP 服务启动: {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_config=_LOG_CONFIG)
