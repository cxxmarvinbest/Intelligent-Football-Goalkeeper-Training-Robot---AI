# -*- coding: utf-8 -*-
"""
智能足球守门员 AI智能模式（11米点位）v1.0
功能：
  1. RTSP 摄像头实时检测守门员和球门
  2. 三重安全机制：
     a. 守门员检测框必须完全位于球门框内
     b. 守门员不能距离发球机过近（检测框占画面比例判定）
     c. 守门员未回中时暂停发球（识别区相交比例判定）
  4. AI 智能模式：反向/跟随发球、预设方向（守门员在中间时）、高度选择（高/中/低）
  5. JSON 协议 通过 RS485 串口控制发球机

通信协议：HTTP (FastAPI + Uvicorn)，端口 8098
硬件通信：RS485 串口, 115200 baud, JSON 格式                      # xxx
"""

import os
import json
import time
import random
import shutil
import subprocess
import threading
import queue
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ============================================================
# 串口通信
# ============================================================
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial 未安装，将使用模拟串口模式")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="智能足球守门员 AI智能模式 API (11米点位)",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 配置常量
# ============================================================
# ── 模型与摄像头 ──
MODEL_PATH = "runs/detect/train-5/weights/best.pt"
RTSP_URL = "rtsp://admin:siboasi123@192.168.8.108:554/LiveMedia/ch1/Media1"

# ── 球门 3x6=18 分区 ──
GOAL_GRID_ROWS = 3
GOAL_GRID_COLS = 6

# ── 球门横向列定义 ──
COL_REVERSE_LEFT  = 0   # 最左区（反向发球区）
COL_CENTER_LEFT   = 1   # 中间判定左列
COL_CENTER_RIGHT  = 4   # 中间判定右列
COL_REVERSE_RIGHT = 5   # 最右区（反向发球区）

# ── YOLO 类别 ID ──
CLS_GOALKEEPER = 0
CLS_CROSSBAR   = 1
CLS_POST_LEFT  = 2
CLS_POST_RIGHT = 3
CLS_BALL       = 4

# ── 检测参数 ──
TRACK_INTERVAL = 2      # 每 N 帧运行一次 YOLO                    # xxx
EMA_ALPHA      = 0.25   # 球门框 EMA 平滑系数                     # xxx
GOAL_CONF      = 0.25   # 球门检测置信度阈值                       # xxx
GK_CONF        = 0.2    # 守门员检测置信度阈值                     # xxx

# ── 安全机制阈值 ──
GK_FRAME_HEIGHT_RATIO_THRESHOLD = 0.45                          # xxx
GK_MIN_CENTER_RECOG_RATIO = 0.55

# ── 自动发球 ──
SAFE_DURATION_BEFORE_SERVE = 5.0   # 安全持续 5 秒后触发发球        # xxx

# ── 发球默认参数 ──
DEFAULT_WHEEL_SPEED = 50    # 默认发球轮速度 (20~100)              # xxx

# ── 串口 ──
SERIAL_PORT = "COM3"                                            # 串口名称 xxx
SERIAL_BAUD = 115200                                            # 串口波特率 xxx

# ── 摄像头缩放 ──
FRAME_SCALE_WIDTH  = 640                                        # xxx
FRAME_SCALE_HEIGHT = 480                                        # xxx
FRAME_CAPTURE_INTERVAL = 0.1   # 100ms 捕捉一张                   # xxx
FRAME_READ_TIMEOUT_MS   = 3000   # 单帧读取超时 3000ms             # xxx

# ── 视频解码 ──
HWACCEL             = "none"
FRAME_DISCARD_COUNT = 25        # 启动时丢弃前25帧

# ── 图片输出质量 ──
JPEG_QUALITY               = 75   # JPEG 质量
VIDEO_IMAGE_MAX_SIZE_LIMIT = 1    # 图片尺寸限制开关 (1=启用, 0=关闭)
VIDEO_IMAGE_MAX_SIZE       = 960  # 图片最大边尺寸 (px)，等比缩放

# ── 发球参数范围 ──
SERVE_INTERVAL_MIN = 3    # 发球间隔最小值（秒）                     # xxx
SERVE_INTERVAL_MAX = 10   # 发球间隔最大值（秒）                     # xxx
SERVE_COUNT_MIN    = 1    # 发球数量最小值                          # xxx
SERVE_COUNT_MAX    = 999  # 发球数量最大值                          # xxx
DURATION_MIN       = 1    # 运动时长最小值（秒）
DURATION_MAX       = 9999 # 运动时长最大值（秒）

# ── 发球模式代码 ──
SERVE_MODE_REVERSE = 0   # 反向发球
SERVE_MODE_FOLLOW  = 1   # 跟随发球（重叠面积最大分区）

# ── 高度代码（= 行号：0=高顶行, 1=中中间行, 2=低底行）──
HEIGHT_HIGH = 0
HEIGHT_MID  = 1
HEIGHT_LOW  = 2

# ── 预设方向代码 ──
PRESET_RANDOM = 0   # 随机
PRESET_LEFT   = 1   # 左
PRESET_RIGHT  = 2   # 右

# ── 结束条件代码 ──
END_FREE      = 0   # 自由（不自动停止）
END_COUNT     = 1   # 发球数量
END_DURATION  = 2   # 运动时长

# ── 分区 → 角度映射（需实物标定，此处为预估值）──
ZONE_H_ANGLE_MAP = [10, 18, 26, 34, 42, 50]    # 6列 → 左右角度   # xxx
ZONE_V_ANGLE_MAP = [45, 35, 25]                # 3行 → 上下角度   # xxx


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Detection:
    """YOLO 单个检测结果"""
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class GoalBox:
    """球门框"""
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


# ============================================================
# 核心算法函数（球门检测与平滑）
# ============================================================
def smooth_goal_box(prev: Optional[GoalBox], curr: GoalBox,
                    alpha: float = EMA_ALPHA) -> GoalBox:
    """EMA 指数移动平均平滑球门检测框"""
    if prev is None:
        return curr
    return GoalBox(
        left   = prev.left   * (1.0 - alpha) + curr.left   * alpha,
        top    = prev.top    * (1.0 - alpha) + curr.top    * alpha,
        right  = prev.right  * (1.0 - alpha) + curr.right  * alpha,
        bottom = prev.bottom * (1.0 - alpha) + curr.bottom * alpha,
    )


def extract_detections(result) -> List[Detection]:
    """从 YOLO 推理结果提取检测列表"""
    dets = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    boxes = result.boxes
    for i in range(len(boxes)):
        cls  = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        dets.append(Detection(cls, conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
    return dets


def assemble_goal(dets: List[Detection],
                  conf_thresh: float = GOAL_CONF) -> Optional[GoalBox]:
    """从检测结果组装球门框（横梁 + 左柱 + 右柱）"""
    crossbar = post_left = post_right = None
    for d in dets:
        if d.conf < conf_thresh:
            continue
        if d.cls == CLS_CROSSBAR:
            if crossbar is None or d.conf > crossbar.conf:
                crossbar = d
        elif d.cls == CLS_POST_LEFT:
            if post_left is None or d.conf > post_left.conf:
                post_left = d
        elif d.cls == CLS_POST_RIGHT:
            if post_right is None or d.conf > post_right.conf:
                post_right = d

    parts = [p for p in (crossbar, post_left, post_right) if p is not None]
    if not parts:
        return None

    if crossbar and post_left and post_right:
        if post_left.cx > post_right.cx:
            post_left, post_right = post_right, post_left

    left   = min(p.x1 for p in parts)
    top    = min(p.y1 for p in parts)
    right  = max(p.x2 for p in parts)
    bottom = max(p.y2 for p in parts)

    if (right - left) < 30 or (bottom - top) < 30:
        return None
    return GoalBox(left, top, right, bottom)


def get_zone_grid(goal: GoalBox) -> Tuple[List[float], List[float]]:
    """生成 3x6 分区网格线坐标"""
    xs = [goal.left + i * goal.width / GOAL_GRID_COLS
          for i in range(GOAL_GRID_COLS + 1)]
    ys = [goal.top + i * goal.height / GOAL_GRID_ROWS
          for i in range(GOAL_GRID_ROWS + 1)]
    return xs, ys


def zone_number(row: int, col: int) -> int:
    """行列号 -> 分区编号 (1~18)"""
    return row * GOAL_GRID_COLS + col + 1


# ============================================================
# 发球目标判定
# ============================================================
def resolve_target_row(height: int) -> int:
    """高度代码 -> 行号（0=高顶行, 1=中中间行, 2=低底行）"""
    return max(0, min(GOAL_GRID_ROWS - 1, height))


def find_gk_zone(gk_det: Optional[Detection],
                 goal: GoalBox) -> Tuple[int, int, int]:
    """计算守门员中心点所在的分区行列号。"""
    if gk_det is None or goal is None:
        return -1, -1, -1
    xs, ys = get_zone_grid(goal)

    col = -1
    for c in range(GOAL_GRID_COLS):
        if xs[c] <= gk_det.cx < xs[c + 1]:
            col = c
            break
    if col < 0:
        col = 0 if gk_det.cx < xs[0] else GOAL_GRID_COLS - 1

    row = -1
    for r in range(GOAL_GRID_ROWS):
        if ys[r] <= gk_det.cy < ys[r + 1]:
            row = r
            break
    if row < 0:
        row = 0 if gk_det.cy < ys[0] else GOAL_GRID_ROWS - 1

    return row, col, zone_number(row, col)


def find_gk_zone_by_overlap(gk_det: Optional[Detection],
                            goal: GoalBox) -> Tuple[int, int, int]:
    """计算守门员检测框与 3x6=18 个分区的重叠面积，返回重叠面积最大的分区行列号。"""
    if gk_det is None or goal is None:
        return -1, -1, -1
    xs, ys = get_zone_grid(goal)

    max_overlap = 0.0
    best_row, best_col = -1, -1

    gk_x1, gk_x2 = gk_det.x1, gk_det.x2
    gk_y1, gk_y2 = gk_det.y1, gk_det.y2

    for r in range(GOAL_GRID_ROWS):
        for c in range(GOAL_GRID_COLS):
            overlap_x1 = max(gk_x1, xs[c])
            overlap_y1 = max(gk_y1, ys[r])
            overlap_x2 = min(gk_x2, xs[c + 1])
            overlap_y2 = min(gk_y2, ys[r + 1])

            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                if area > max_overlap:
                    max_overlap = area
                    best_row, best_col = r, c

    if best_row < 0 or best_col < 0:
        return -1, -1, -1
    return best_row, best_col, zone_number(best_row, best_col)


def get_gk_position(gk_det: Optional[Detection], goal: GoalBox
                    ) -> Tuple[Optional[str], List[float]]:
    """
    判断守门员在球门中的列位置。
    居中判定: ratio = overlap[2] / (overlap[2] + overlap[3])
      0.4 <= ratio <= 0.6 -> "center"
    """
    if gk_det is None:
        return None, [0.0] * GOAL_GRID_COLS

    xs, _ = get_zone_grid(goal)
    gk_x1, gk_x2 = gk_det.x1, gk_det.x2

    overlaps = [max(0.0, min(gk_x2, xs[c + 1]) - max(gk_x1, xs[c]))
                for c in range(GOAL_GRID_COLS)]

    ov_2_3 = overlaps[2] + overlaps[3]
    if ov_2_3 > 0:
        ratio_col2 = overlaps[2] / ov_2_3
        if 0.4 <= ratio_col2 <= 0.6:
            gk_side = "center"
        elif ratio_col2 > 0.6:
            gk_side = "left"
        else:
            gk_side = "right"
    else:
        left_ov  = overlaps[0] + overlaps[1]
        right_ov = overlaps[4] + overlaps[5]
        gk_side = "left" if left_ov > right_ov else "right"

    return gk_side, overlaps


def determine_serve_target(mode: int, gk_side: Optional[str],
                           gk_overlaps: Optional[List[float]],
                           preset_direction: int, target_row: int) -> Tuple[int, int]:
    """
    反向发球目标判定。
    仅处理反向发球(0)；跟随发球(1)在算法线程中单独处理。

    参数:
      mode: 0=反向发球
      gk_side: "left"/"right"/"center"
      gk_overlaps: 守门员与各列相交宽度
      preset_direction: 0=随机, 1=左, 2=右（GK居中时生效）
      target_row: 目标行号
    """
    # 反向发球
    if gk_side == "left":
        target_col = COL_REVERSE_RIGHT
    elif gk_side == "right":
        target_col = COL_REVERSE_LEFT
    else:
        if preset_direction == PRESET_LEFT:
            target_col = COL_REVERSE_LEFT
        elif preset_direction == PRESET_RIGHT:
            target_col = COL_REVERSE_RIGHT
        else:
            target_col = random.choice([COL_REVERSE_LEFT, COL_REVERSE_RIGHT])

    return target_col, zone_number(target_row, target_col)


def zone_to_angles(row: int, col: int) -> Tuple[int, int]:
    h_idx = max(0, min(GOAL_GRID_COLS - 1, col))
    v_idx = max(0, min(GOAL_GRID_ROWS - 1, row))
    return ZONE_H_ANGLE_MAP[h_idx], ZONE_V_ANGLE_MAP[v_idx]


# ============================================================
# 安全检测（三重规则）
# ============================================================
def check_goalkeeper_in_goal(gk_det: Detection,
                             goal: GoalBox) -> Tuple[bool, str, Dict]:
    detail = {}
    gk_fully_in_goal = (
        gk_det.x1 >= goal.left  and
        gk_det.y1 >= goal.top   and
        gk_det.x2 <= goal.right and
        gk_det.y2 <= goal.bottom
    )

    overlap_x1 = max(gk_det.x1, goal.left)
    overlap_y1 = max(gk_det.y1, goal.top)
    overlap_x2 = min(gk_det.x2, goal.right)
    overlap_y2 = min(gk_det.y2, goal.bottom)

    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        gk_area = gk_det.area
        overlap_ratio = overlap_area / gk_area if gk_area > 0 else 0
    else:
        overlap_ratio = 0.0

    detail["gk_fully_in_goal"]   = gk_fully_in_goal
    detail["goal_overlap_ratio"] = round(overlap_ratio, 3)

    if not gk_fully_in_goal:
        return False, "守门员不在球门框内（未完全包含）", detail
    return True, "规则1通过", detail


def check_safe_distance(gk_det: Detection,
                        frame_shape: Tuple[int, int]) -> Tuple[bool, str, Dict]:
    detail = {}
    frame_h = frame_shape[0]
    gk_frame_height_ratio = gk_det.height / frame_h if frame_h > 0 else 0
    detail["gk_frame_height_ratio"] = round(gk_frame_height_ratio, 3)

    if gk_frame_height_ratio > GK_FRAME_HEIGHT_RATIO_THRESHOLD:
        return False, (
            f"守门员距离过近 "
            f"(检测框占比 {gk_frame_height_ratio:.0%} > 阈值 {GK_FRAME_HEIGHT_RATIO_THRESHOLD:.0%})"
        ), detail
    return True, "规则2通过", detail


def check_return_center(gk_det: Detection, goal: GoalBox) -> Tuple[bool, str, Dict]:
    detail = {}
    xs, _ = get_zone_grid(goal)

    recog_left_bound  = xs[COL_CENTER_LEFT]
    recog_right_bound = xs[COL_CENTER_RIGHT + 1]

    overlap_recog = max(0.0, min(gk_det.x2, recog_right_bound) - max(gk_det.x1, recog_left_bound))
    gk_width = gk_det.width
    center_recog_ratio = overlap_recog / gk_width if gk_width > 0 else 0

    detail["center_recog_ratio"] = round(center_recog_ratio, 3)
    detail["overlap_recog"] = round(overlap_recog, 1)

    if center_recog_ratio < GK_MIN_CENTER_RECOG_RATIO:
        return False, (
            f"守门员未回中，暂停发球 "
            f"(识别区相交比{center_recog_ratio:.1%} < 要求{GK_MIN_CENTER_RECOG_RATIO:.0%})"
        ), detail
    return True, "规则3通过", detail


def check_safety(gk_det: Optional[Detection],
                 goal: Optional[GoalBox],
                 frame_shape: Tuple[int, int]) -> Tuple[bool, str, Dict]:
    """三重安全检查"""
    if gk_det is None:
        return False, "未检测到守门员", {}
    if goal is None:
        return False, "未检测到球门", {}

    is_safe, reason, detail = check_goalkeeper_in_goal(gk_det, goal)
    if not is_safe:
        return False, reason, detail

    is_safe, reason, detail2 = check_safe_distance(gk_det, frame_shape)
    detail.update(detail2)
    if not is_safe:
        return False, reason, detail

    is_safe, reason, detail3 = check_return_center(gk_det, goal)
    detail.update(detail3)
    if not is_safe:
        return False, reason, detail

    return True, "安全", detail


# ============================================================
# MotorDriver -- JSON 协议串口驱动
# ============================================================
class MotorDriver:
    """发球机驱动类，使用 JSON 协议通过 RS485 串口通信。"""

    SERVE_PID_BASE = 10226                           # xxx

    def __init__(self, port: str = SERIAL_PORT, baudrate: int = SERIAL_BAUD):
        self.port_name   = port
        self.baudrate    = baudrate
        self.ser: Optional[Any] = None
        self.connected   = False
        self._serve_pid_counter = self.SERVE_PID_BASE
        self._read_lock  = threading.Lock()

    def connect(self) -> bool:
        if not SERIAL_AVAILABLE:
            print("[Motor] pyserial 未安装，进入模拟模式")
            self.connected = True
            return True
        try:
            self.ser = serial.Serial(
                self.port_name,
                self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
            self.connected = True
            print(f"[Motor] 串口已连接: {self.port_name} @ {self.baudrate} (RS485)")
            return True
        except Exception as e:
            print(f"[Motor] 串口连接失败 ({self.port_name}): {e}，进入模拟模式")
            self.connected = True
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.connected = False
        print("[Motor] 串口已断开")

    def _send_json(self, msg: Dict) -> bool:
        raw = json.dumps(msg, ensure_ascii=False) + "\n"
        print(f"[Motor] TX: {raw.strip()}")
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(raw.encode("utf-8"))
                return True
            except Exception as e:
                print(f"[Motor] TX 失败: {e}")
                return False
        return True

    def _recv_json(self, timeout: float = 2.0) -> Optional[Dict]:
        if self.ser and self.ser.is_open:
            with self._read_lock:
                try:
                    self.ser.timeout = timeout
                    line = self.ser.readline()
                    if line:
                        decoded = line.decode("utf-8", errors="ignore").strip()
                        if decoded:
                            print(f"[Motor] RX: {decoded}")
                            return json.loads(decoded)
                except Exception as e:
                    print(f"[Motor] RX 异常: {e}")
        return None

    def _send_and_recv(self, msg: Dict, timeout: float = 2.0) -> Optional[Dict]:
        self._send_json(msg)
        return self._recv_json(timeout)

    def start_motor(self) -> bool:
        print("[Motor] >>> 启动电机 (STATE=work)")
        msg = {"PID": 26, "MDF": {"STATE": "work"}, "CKS": 0}              # xxx
        resp = self._send_and_recv(msg)
        if resp and resp.get("RST") == 0:
            print("[Motor] OK 电机已启动")
            return True
        print("[Motor] OK 电机启动指令已发送（模拟/无应答）")
        return True

    def pause_motor(self) -> bool:
        print("[Motor] >>> 暂停电机 (STATE=paus)")
        msg = {"PID": 26, "MDF": {"STATE": "paus"}, "CKS": 0}               # xxx
        resp = self._send_and_recv(msg)
        if resp and resp.get("RST") == 0:
            print("[Motor] OK 电机已暂停")
            return True
        print("[Motor] OK 电机暂停指令已发送（模拟/无应答）")
        return True

    def stop_motor(self) -> bool:
        print("[Motor] >>> 停止电机 (STATE=stop)")
        msg = {"PID": 26, "MDF": {"STATE": "stop"}, "CKS": 0}                # xxx
        resp = self._send_and_recv(msg)
        if resp and resp.get("RST") == 0:
            print("[Motor] OK 电机已停止")
            return True
        print("[Motor] OK 电机停止指令已发送（模拟/无应答）")
        return True

    def serve(self,
              wheel1: int = DEFAULT_WHEEL_SPEED,
              wheel2: int = DEFAULT_WHEEL_SPEED,
              wheel3: int = DEFAULT_WHEEL_SPEED,
              h_angle: int = 30,
              v_angle: int = 30) -> int:
        """发球指令。SDATA = [轮1, 轮2, 轮3, 左右角度, 上下角度]"""
        pid = self._serve_pid_counter
        self._serve_pid_counter += 1

        w1  = max(20, min(100, wheel1))
        w2  = max(20, min(100, wheel2))
        w3  = max(20, min(100, wheel3))
        ha  = max(0, min(60, h_angle))
        va  = max(0, min(60, v_angle))

        sdata = [w1, w2, w3, ha, va]
        snext = [w1, w2, w3, ha, va]

        print(f"[Motor] >>> 发球 (PID={pid}) SDATA={sdata}")

        msg = {"PID": pid, "MDF": {"SDATA": sdata, "SNEXT": snext}, "CKS": 0}                 # xxx
        resp = self._send_and_recv(msg)
        if resp and resp.get("RST") == 0:
            print(f"[Motor] OK 发球指令已接受 (PID={pid})")
        else:
            print(f"[Motor] OK 发球指令已发送 (PID={pid})（模拟/无应答）")

        return pid

    def query_status(self) -> Dict:
        msg = {"PID": 1425, "REQ": ["STATE", "ALERT", "SBCNT", "BSTAT", "RUNTM"], "CKS": 0}    # xxx
        resp = self._send_and_recv(msg, timeout=1.0)
        if resp and resp.get("RES"):
            return resp["RES"]
        return {}


# ============================================================
# 统一 API 响应
# ============================================================
def api_ok(data: Any = None, msg: str = "") -> Dict:
    return {"code": 200, "msg": msg, "data": data}

def api_err(code: int = 500, msg: str = "", data: Any = None) -> Dict:
    return {"code": code, "msg": msg, "data": data}


# ============================================================
# 全局状态（线程安全）
# ============================================================
class SystemState:
    """系统全局状态，所有字段通过 lock 保护"""

    def __init__(self):
        self.lock = threading.Lock()

        # -- 运行控制 --
        self.is_running = False
        self.is_paused  = False
        self.algorithm_thread_running = False

        # -- 摄像头 --
        self.camera_started = False

        # -- 安全状态 --
        self.is_safe       = False
        self.safety_reason = "系统未启动"
        self.safety_detail: Dict = {}

        # -- 发球相关 --
        self.serve_count      = 0
        self.last_serve_pid   = 0
        self.safe_start_time  = 0.0
        self.last_serve_time  = 0.0
        self.serve_triggered  = False
        self.last_serve_data: Optional[List[int]] = None

        # -- AI 智能模式参数（全部数字代码）--
        self.serve_mode       = SERVE_MODE_REVERSE    # 0=反向, 1=跟随
        self.preset_direction = PRESET_RANDOM         # 0=随机, 1=左, 2=右
        self.height           = HEIGHT_MID            # 0=高, 1=中, 2=低

        # -- 发球控制参数 --
        self.serve_interval    = 5        # 发球间隔（秒），3~10
        self.end_condition     = END_FREE # 0=自由, 1=发球数量, 2=运动时长
        self.serve_count_limit = 10       # 发球数量上限（1~999）
        self.duration_limit    = 180      # 运动时长上限（秒）
        self.start_time        = 0.0      # 系统启动时间戳

        # -- 延迟参数变更 --
        self.pending_params: Optional[Dict] = None

        # -- 当前发球目标 --
        self.target_zone = 0
        self.target_row  = -1
        self.target_col  = -1

        # -- 检测状态 --
        self.goal_detected = False
        self.gk_detected   = False
        self.gk_side       = None
        self.frame_shape   = (0, 0)

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
            self.is_safe       = is_safe
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
            self.serve_count     += 1
            self.last_serve_pid   = pid
            self.last_serve_time  = time.time()
            self.serve_triggered  = True
            self.last_serve_data  = [w1, w2, w3, h_angle, v_angle]

    def check_end_condition(self) -> bool:
        """检查是否满足结束条件。返回 True 表示应停止。"""
        with self.lock:
            if self.end_condition == END_COUNT:
                if self.serve_count >= self.serve_count_limit:
                    return True
            elif self.end_condition == END_DURATION:
                elapsed = time.time() - self.start_time
                if elapsed >= self.duration_limit:
                    return True
            return False

    def apply_pending_params(self):
        """应用待生效的参数（在每次发球后调用）。"""
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
            device_state = "stopped"
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
                    if self.is_safe and self.safe_start_time > 0 else 0, 1
                ),
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
# 全局实例
# ============================================================
state: SystemState  = SystemState()
motor: MotorDriver  = MotorDriver()
model: Optional[YOLO] = None

# 帧队列（摄像头采集 -> 算法处理）
frame_queue: queue.Queue = queue.Queue(maxsize=1)

# 最新帧（供 /frames 查询，写入时缩放）
latest_frame: Optional[np.ndarray] = None
latest_frame_lock = threading.Lock()

# 帧读取线程池（用于超时控制）
_frame_read_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# 算法线程内部状态
smoothed_goal: Optional[GoalBox] = None
latest_gk: Optional[Detection] = None


# ============================================================
# Pydantic 请求模型
# ============================================================
class StartRequest(BaseModel):
    """开始/继续 请求参数（全部数字代码）"""
    serve_mode: int              # 0=反向发球, 1=跟随发球
    height: int                  # 0=高, 1=中, 2=低
    preset_direction: int        # 0=随机, 1=左, 2=右
    serve_interval: int          # 3~10（秒）
    end_condition: int           # 0=自由, 1=发球数量, 2=运动时长
    serve_count_limit: Optional[int] = None   # 1~999（end_condition=1时必填）
    duration_limit: Optional[int] = None      # 1~9999（end_condition=2时必填）


class UpdateParamsRequest(BaseModel):
    """变更参数请求（全部可选，仅更新传入字段）"""
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
    """
    校验 start/resume 请求参数。
    返回 (是否合法, 错误详情 dict)。
    合法时 detail=None；不合法时 detail={"errors": [...]}。
    """
    errors = []

    if req.serve_mode not in (SERVE_MODE_REVERSE, SERVE_MODE_FOLLOW):
        errors.append(f"serve_mode 必须为 0(反向发球)/1(跟随发球)，当前: {req.serve_mode}")
    if req.height not in (HEIGHT_HIGH, HEIGHT_MID, HEIGHT_LOW):
        errors.append(f"height 必须为 0(高)/1(中)/2(低)，当前: {req.height}")
    if req.preset_direction not in (PRESET_RANDOM, PRESET_LEFT, PRESET_RIGHT):
        errors.append(f"preset_direction 必须为 0(随机)/1(左)/2(右)，当前: {req.preset_direction}")
    if not (SERVE_INTERVAL_MIN <= req.serve_interval <= SERVE_INTERVAL_MAX):
        errors.append(f"serve_interval 必须为 {SERVE_INTERVAL_MIN}~{SERVE_INTERVAL_MAX}，当前: {req.serve_interval}")
    if req.end_condition not in (END_FREE, END_COUNT, END_DURATION):
        errors.append(f"end_condition 必须为 0(自由)/1(发球数量)/2(运动时长)，当前: {req.end_condition}")
    if req.end_condition == END_COUNT:
        if req.serve_count_limit is None or not (SERVE_COUNT_MIN <= req.serve_count_limit <= SERVE_COUNT_MAX):
            errors.append(f"serve_count_limit 必须为 {SERVE_COUNT_MIN}~{SERVE_COUNT_MAX}，当前: {req.serve_count_limit}")
    if req.end_condition == END_DURATION:
        if req.duration_limit is None or not (DURATION_MIN <= req.duration_limit <= DURATION_MAX):
            errors.append(f"duration_limit 必须为 {DURATION_MIN}~{DURATION_MAX}，当前: {req.duration_limit}")

    if errors:
        return False, {"errors": errors}
    return True, None


def validate_update_params(params: UpdateParamsRequest) -> Tuple[bool, Optional[Dict]]:
    """
    校验 params 请求中非 None 的字段。
    返回 (是否合法, 错误详情 dict)。
    """
    errors = []

    if params.serve_mode is not None and params.serve_mode not in (SERVE_MODE_REVERSE, SERVE_MODE_FOLLOW):
        errors.append(f"serve_mode 必须为 0(反向发球)/1(跟随发球)，当前: {params.serve_mode}")
    if params.height is not None and params.height not in (HEIGHT_HIGH, HEIGHT_MID, HEIGHT_LOW):
        errors.append(f"height 必须为 0(高)/1(中)/2(低)，当前: {params.height}")
    if params.preset_direction is not None and params.preset_direction not in (PRESET_RANDOM, PRESET_LEFT, PRESET_RIGHT):
        errors.append(f"preset_direction 必须为 0(随机)/1(左)/2(右)，当前: {params.preset_direction}")
    if params.serve_interval is not None and not (SERVE_INTERVAL_MIN <= params.serve_interval <= SERVE_INTERVAL_MAX):
        errors.append(f"serve_interval 必须为 {SERVE_INTERVAL_MIN}~{SERVE_INTERVAL_MAX}，当前: {params.serve_interval}")
    if params.end_condition is not None and params.end_condition not in (END_FREE, END_COUNT, END_DURATION):
        errors.append(f"end_condition 必须为 0(自由)/1(发球数量)/2(运动时长)，当前: {params.end_condition}")
    if params.serve_count_limit is not None and not (SERVE_COUNT_MIN <= params.serve_count_limit <= SERVE_COUNT_MAX):
        errors.append(f"serve_count_limit 必须为 {SERVE_COUNT_MIN}~{SERVE_COUNT_MAX}，当前: {params.serve_count_limit}")
    if params.duration_limit is not None and not (DURATION_MIN <= params.duration_limit <= DURATION_MAX):
        errors.append(f"duration_limit 必须为 {DURATION_MIN}~{DURATION_MAX}，当前: {params.duration_limit}")

    if errors:
        return False, {"errors": errors}
    return True, None


# ============================================================
# 模型加载
# ============================================================
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            print(f"[AI] 模型加载成功: {MODEL_PATH}")
        except Exception as e:
            print(f"[AI] 模型加载失败: {e}")
            model = None
    else:
        print(f"[AI] 模型文件不存在: {MODEL_PATH}，安全检测不可用")


# ============================================================
# FFmpegCameraReader — 使用 ffmpeg 解码 H265 RTSP 视频流
# ============================================================

class FFmpegCameraReader:
    """
    通过 ffmpeg 子进程解码 H265 RTSP 视频流。

    极简参数：仅 -rtsp_transport tcp -i URL，不加任何额外 flag。
    -loglevel error 屏蔽中途接入 GOP 时正常的 "Could not find ref" 警告。

    GPU 硬解参数 (-hwaccel cuda / hevc_cuvid / +genpts+flush_packets) 会触发
    RTSP CSeq 错乱，锁定纯软件解码。摄像头 RTSP 无固件缺陷。

    核心设计：
      1. FFmpeg 管道 + rawvideo BGR24，绕过 OpenCV 直接解码 H265
      2. -vf scale 过滤器做缩放（比 -s 更可靠）
      3. 极简参数，不干扰 RTSP 会话建立
      4. OpenCV 探测源流真实分辨率与帧率
      5. 启动时丢弃前 N 帧避免绿屏/花屏
      6. stderr 实时打印 + 进程退出自动重启
    """

    FRAME_BYTES = FRAME_SCALE_WIDTH * FRAME_SCALE_HEIGHT * 3  # BGR24 每帧字节数

    def __init__(self, rtsp_url: str = RTSP_URL):
        self.rtsp_url = rtsp_url
        self.process: Optional[subprocess.Popen] = None
        self._stderr_drainer: Optional[threading.Thread] = None
        self._read_count = 0          # 成功读取帧数（含丢弃期）
        self._error_count = 0         # 连续错误次数
        self._source_w = 0            # 源流真实宽度
        self._source_h = 0            # 源流真实高度
        self._source_fps = 30         # 源流真实帧率
        self._current_decoder = "hevc (software)"
        self._current_hwaccel = "none"

    @staticmethod
    def _probe_stream(rtsp_url: str) -> Tuple[int, int, int]:
        """用 OpenCV (CAP_FFMPEG) 探测真实流分辨率与帧率"""
        cap = None
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_raw = cap.get(cv2.CAP_PROP_FPS)
                fps = int(max(fps_raw % 100, 0) or 30.0)
                if w > 0 and h > 0:
                    return w, h, fps
        except Exception as e:
            print(f"[Camera] OpenCV 探测流信息失败（将使用默认值）: {e}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        return 0, 0, 30


    def _build_cmd(self) -> List[str]:
        """软件解码 H265 RTSP"""
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        cmd += [
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
        ]

        # 输出：H265 → BGR24
        cmd += [
            "-an",
            "-vf", f"scale={FRAME_SCALE_WIDTH}:{FRAME_SCALE_HEIGHT}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:",
        ]

        self._current_decoder = "hevc (software)"
        self._current_hwaccel = "none"
        return cmd

    def _drain_stderr(self):
        """后台排水线程：逐行读取 stderr 并打印，防止管道堵塞"""
        try:
            while self.process and self.process.stderr:
                line = self.process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    print(f"[ffmpeg] {text}")
        except Exception:
            pass

    def open(self) -> bool:
        """启动 ffmpeg 子进程解码 RTSP 流"""
        # 1) 用 OpenCV 探测源流真实分辨率/帧率
        self._source_w, self._source_h, self._source_fps = self._probe_stream(self.rtsp_url)
        if self._source_w > 0:
            print(f"[Camera] 源流信息: {self._source_w}x{self._source_h} @ {self._source_fps} FPS")
        else:
            print("[Camera] 未能探测源流信息，将使用默认值")

        print("[Camera] 纯软件解码（极简参数，loglevel=error 静默 GOP 警告）")

        try:
            cmd = self._build_cmd()
            print(f"[Camera] ffmpeg cmd: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024 * 1024,  # 1MB 缓冲区
            )
            self._stderr_drainer = threading.Thread(target=self._drain_stderr, daemon=True)
            self._stderr_drainer.start()
            self._read_count = 0
            self._error_count = 0
            print(f"[Camera] ffmpeg 已启动 | 解码: {self._current_decoder} | 输出: {FRAME_SCALE_WIDTH}x{FRAME_SCALE_HEIGHT}")
            return True
        except Exception as e:
            print(f"[Camera] ffmpeg 启动失败: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        从 ffmpeg stdout 读取一帧。
        帧已在 ffmpeg 中通过 scale 过滤器缩放到 FRAME_SCALE_WIDTH x FRAME_SCALE_HEIGHT。
        返回 (True, frame) 或 (False, None)。采用循环读取。
        """
        if self.process is None or self.process.poll() is not None:
            return False, None
        try:
            raw = b""
            needed = self.FRAME_BYTES
            while len(raw) < needed:
                chunk = self.process.stdout.read(needed - len(raw))
                if not chunk:
                    self._error_count += 1
                    return False, None
                raw += chunk
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (FRAME_SCALE_HEIGHT, FRAME_SCALE_WIDTH, 3)
            )
            self._read_count += 1
            self._error_count = 0  # 成功读取，重置错误计数
            return True, frame
        except Exception as e:
            self._error_count += 1
            print(f"[Camera] ffmpeg 读取异常: {e}")
            return False, None

    def is_alive(self) -> bool:
        """ffmpeg 子进程是否仍在运行"""
        return self.process is not None and self.process.poll() is None

    def release(self):
        """关闭 ffmpeg 进程"""
        if self.process:
            try:
                self.process.stdout.close()
            except Exception:
                pass
            try:
                self.process.stderr.close()
            except Exception:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except Exception:
                self.process.kill()
            self.process = None
        print("[Camera] ffmpeg 进程已关闭")


# ============================================================
# 摄像头启动（供 /open 和 /start 共用）
# ============================================================
def start_camera():
    """启动摄像头采集线程"""
    if state.camera_started:
        return False
    state.camera_started = True
    threading.Thread(target=camera_capture_thread, daemon=True).start()
    return True


def stop_camera():
    """关闭摄像头采集线程 + 清除最新帧"""
    global latest_frame
    if not state.camera_started:
        return False
    state.camera_started = False
    with latest_frame_lock:
        latest_frame = None
    return True


# ============================================================
# 线程1：摄像头采集（100ms 间隔 + ffmpeg 缩放 + 1000ms 超时）
# ============================================================
def _read_frame_with_timeout(reader: FFmpegCameraReader,
                             timeout_s: float):
    """异步读取一帧，超时返回 (False, None)"""
    future = _frame_read_executor.submit(reader.read)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        print(f"[Camera] 读取超时({int(timeout_s * 1000)}ms)，跳过本次")
        future.cancel()
        return False, None


def camera_capture_thread():
    """
    持续从 RTSP 摄像头抓取帧（ffmpeg 解码 H265），写入：
      1. latest_frame  -> 供 /frames 查询（ffmpeg 已缩放至 640x480）
      2. frame_queue   -> 供算法线程消费（仅系统运行时）
    采集间隔：100ms；启动期（含丢弃期）放宽到 5 秒，稳定期 3 秒。
    丢弃前 FRAME_DISCARD_COUNT 帧（避免绿屏/花屏）。
    ffmpeg 进程退出时自动重启。
    """
    global latest_frame
    reader = FFmpegCameraReader(RTSP_URL)

    if not reader.open():
        print("[Camera] 错误：ffmpeg 无法连接 RTSP 摄像头！")
        state.camera_started = False
        return

    print(f"[Camera] 摄像头已连接，开始采集 (100ms/帧, 丢弃前{FRAME_DISCARD_COUNT}帧, H265 软解)")

    last_capture_time = 0

    while state.camera_started:
        # -- ffmpeg 进程存活检查，退出则自动重启 --
        if not reader.is_alive():
            print("[Camera] ffmpeg 进程已退出，尝试重启...")
            reader.release()
            time.sleep(1)
            if not reader.open():
                print("[Camera] 重启失败，等待 2 秒后重试...")
                time.sleep(2)
                continue
            print("[Camera] ffmpeg 重启成功")

        # -- 异步读取 --
        if reader._read_count < FRAME_DISCARD_COUNT + 15:
            timeout_s = 5.0  # 启动期：丢弃期+前15帧给 5 秒
        else:
            timeout_s = FRAME_READ_TIMEOUT_MS / 1000.0  # 稳定期：3秒

        ret, frame = _read_frame_with_timeout(reader, timeout_s)
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        # -- 丢弃前 N 帧 --
        if reader._read_count <= FRAME_DISCARD_COUNT:
            continue

        current_time = time.time()
        if current_time - last_capture_time >= FRAME_CAPTURE_INTERVAL:
            last_capture_time = current_time

            with latest_frame_lock:
                latest_frame = frame.copy()

            # 写入 frame_queue（供算法线程，仅运行时）
            if state.is_running:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame)

        time.sleep(FRAME_CAPTURE_INTERVAL)

    reader.release()
    print("[Camera] 摄像头已释放")


# ============================================================
# 线程2：AI 检测 + 安全判定 + 自动发球
# ============================================================
def algorithm_processor_thread():
    """
    核心算法线程：
      1. 每 TRACK_INTERVAL 帧运行 YOLO 检测
      2. 球门 EMA 平滑、守门员检测
      3. 三重安全检查
      4. 安全持续 5 秒 -> AI发球目标判定 -> 触发发球
      5. 发球后检查结束条件、应用延迟参数
    """
    global smoothed_goal, latest_gk

    frame_idx = 0
    state.algorithm_thread_running = True
    print("[AI] 算法线程已启动")

    try:
        while state.is_running:
            # -- 取帧 --
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            frame_h, frame_w = frame.shape[:2]

            # -- YOLO 检测 --
            if model is not None and frame_idx % TRACK_INTERVAL == 0:
                try:
                    results = model(frame, verbose=False)
                    dets = extract_detections(results[0])

                    curr_goal = assemble_goal(dets, conf_thresh=GOAL_CONF)
                    if curr_goal:
                        smoothed_goal = smooth_goal_box(smoothed_goal, curr_goal, EMA_ALPHA)

                    gk = None
                    for d in dets:
                        if d.cls == CLS_GOALKEEPER and d.conf >= GK_CONF:
                            if gk is None or d.conf > gk.conf:
                                gk = d
                    if gk:
                        latest_gk = gk

                except Exception as e:
                    print(f"[AI] 检测异常: {e}")

            # -- 安全检测 --
            is_safe, reason, detail = check_safety(
                latest_gk, smoothed_goal, (frame_h, frame_w)
            )
            state.set_safe(is_safe, reason, detail)

            # -- 更新检测状态 --
            gk_side = None
            gk_overlaps = None
            with state.lock:
                state.goal_detected = (smoothed_goal is not None)
                state.gk_detected   = (latest_gk is not None)
                state.frame_shape   = (frame_h, frame_w)

                if latest_gk and smoothed_goal:
                    gk_side, gk_overlaps = get_gk_position(latest_gk, smoothed_goal)
                state.gk_side = gk_side

            # -- 安全通过后 -> 自动发球 --
            if not state.is_paused and state.should_serve():
                target_row = resolve_target_row(state.height)

                if state.serve_mode == SERVE_MODE_FOLLOW and latest_gk and smoothed_goal:
                    # 跟随发球：重叠面积最大的分区
                    gk_row, gk_col, target_zone = find_gk_zone_by_overlap(latest_gk, smoothed_goal)
                    if target_zone > 0:
                        target_row = gk_row
                        target_col = gk_col
                    else:
                        target_col = random.randint(COL_CENTER_LEFT, COL_CENTER_RIGHT)
                        target_zone = zone_number(target_row, target_col)
                elif latest_gk and smoothed_goal and gk_side:
                    target_col, target_zone = determine_serve_target(
                        state.serve_mode,
                        gk_side,
                        gk_overlaps,
                        state.preset_direction,
                        target_row,
                    )
                else:
                    target_col = COL_REVERSE_RIGHT
                    target_zone = zone_number(target_row, target_col)

                h_angle, v_angle = zone_to_angles(target_row, target_col)

                with state.lock:
                    state.target_zone = target_zone
                    state.target_row  = target_row
                    state.target_col  = target_col

                # -- 发球 --
                pid = motor.serve(
                    wheel1=DEFAULT_WHEEL_SPEED,
                    wheel2=DEFAULT_WHEEL_SPEED,
                    wheel3=DEFAULT_WHEEL_SPEED,
                    h_angle=h_angle,
                    v_angle=v_angle,
                )
                state.record_serve(
                    pid, DEFAULT_WHEEL_SPEED, DEFAULT_WHEEL_SPEED,
                    DEFAULT_WHEEL_SPEED, h_angle, v_angle
                )
                print(f"[Serve] 自动发球 #{state.serve_count} "
                      f"(PID={pid}) Zone={target_zone} "
                      f"角度=[{h_angle},{v_angle}]")

                # -- 发球后：应用延迟参数（下一球使用新参数）--
                state.apply_pending_params()

                # -- 检查结束条件 --
                if state.check_end_condition():
                    with state.lock:
                        state.is_running = False
                    motor.stop_motor()
                    print("[System] 达到结束条件，自动停止")
                    break

            frame_idx += 1
    finally:
        state.algorithm_thread_running = False
        print("[AI] 算法线程已停止")


# ============================================================
# FastAPI 路由
# ============================================================

# -- 1. 打开摄像头 --
@app.post("/open", summary="打开摄像头", tags=["摄像头"])
def api_open():
    """
    启动摄像头采集线程。
    摄像头以 100ms 间隔采集，通过 ffmpeg 解码 H265 RTSP 流，
    缩放至 640x480，持续覆盖最新帧。
    RTSP 地址: rtsp://admin:siboasi123@192.168.8.108:554/LiveMedia/ch1/Media1
    """
    if state.camera_started:
        return api_ok({"camera_state": "running"}, "摄像头已打开")

    start_camera()
    print("[System] 摄像头已打开")
    return api_ok({"camera_state": "running"}, "摄像头已打开")


# -- 1b. 关闭摄像头 --
@app.post("/close", summary="关闭摄像头", tags=["摄像头"])
def api_close():
    """
    关闭摄像头采集线程，释放 ffmpeg 进程和最新帧缓存。
    不影响系统运行状态和发球机。
    """
    if not state.camera_started:
        return api_ok({"camera_state": "stopped"}, "摄像头未打开")

    stop_camera()
    print("[System] 摄像头已关闭")
    return api_ok({"camera_state": "stopped"}, "摄像头已关闭")


# -- 2. 开始 --
@app.post("/start", summary="开始", tags=["核心控制"])
def api_start(req: StartRequest):
    """
    启动系统：
    1. 参数校验（失败返回 code=500）
    2. 自动启动摄像头（如未启动）
    3. 连接发球机串口，启动电机
    4. 设置发球参数，启动 AI 检测与自动发球线程

    参数全部使用数字代码，详见请求模型字段说明。
    """
    valid, detail = validate_start_params(req)
    if not valid:
        return api_err(500, "参数错误，请重新上传", detail)

    # 自动启动摄像头
    if not state.camera_started:
        start_camera()
        time.sleep(0.5)  # 等待摄像头初始化

    # 连接并启动电机
    if not motor.connected:
        motor.connect()
    motor.start_motor()
    time.sleep(0.3)

    # 设置参数
    with state.lock:
        state.serve_mode       = req.serve_mode
        state.height           = req.height
        state.preset_direction = req.preset_direction
        state.serve_interval   = req.serve_interval
        state.end_condition    = req.end_condition
        state.serve_count_limit = req.serve_count_limit
        state.duration_limit   = req.duration_limit
        state.is_running       = True
        state.is_paused        = False
        state.start_time       = time.time()
        state.serve_count      = 0
        state.serve_triggered  = False
        state.safe_start_time  = 0.0
        state.pending_params   = None

    # 启动算法线程
    if not state.algorithm_thread_running:
        threading.Thread(target=algorithm_processor_thread, daemon=True).start()

    print("[System] 系统已启动")
    return api_ok({
        "device_state":     state.get_device_state(),
        "serve_mode":       state.serve_mode,
        "height":           state.height,
        "preset_direction": state.preset_direction,
        "serve_interval":   state.serve_interval,
        "end_condition":    state.end_condition,
    }, "系统已启动")


# -- 3. 暂停 --
@app.post("/pause", summary="暂停", tags=["核心控制"])
def api_pause():
    """暂停系统：暂停电机，暂停自动发球。摄像头和检测继续运行。"""
    if not state.is_running:
        return api_err(500, "系统未启动，无法暂停")

    state.is_paused = True
    motor.pause_motor()
    print("[System] 系统已暂停")
    return api_ok({"device_state": state.get_device_state()}, "系统已暂停")


# -- 4. 继续（参数与 start 一致）--
@app.post("/resume", summary="继续", tags=["核心控制"])
def api_resume(req: StartRequest):
    """
    从暂停状态恢复运行，参数与 /start 完全一致。
    重新校验参数、重启电机、更新参数、恢复发球。
    """
    valid, detail = validate_start_params(req)
    if not valid:
        return api_err(500, "参数错误，请重新上传", detail)

    # 确保摄像头已启动
    if not state.camera_started:
        start_camera()
        time.sleep(0.5)

    # 重启电机
    if not motor.connected:
        motor.connect()
    motor.start_motor()
    time.sleep(0.3)

    # 更新参数（与 start 一致）
    with state.lock:
        state.serve_mode       = req.serve_mode
        state.height           = req.height
        state.preset_direction = req.preset_direction
        state.serve_interval   = req.serve_interval
        state.end_condition    = req.end_condition
        state.serve_count_limit = req.serve_count_limit
        state.duration_limit   = req.duration_limit
        state.is_running       = True
        state.is_paused        = False
        state.serve_triggered  = False
        state.safe_start_time  = 0.0
        state.pending_params   = None

    # 启动算法线程（如果未运行）
    if not state.algorithm_thread_running:
        threading.Thread(target=algorithm_processor_thread, daemon=True).start()

    print("[System] 系统已从暂停恢复")
    return api_ok({
        "device_state":     state.get_device_state(),
        "serve_mode":       state.serve_mode,
        "height":           state.height,
        "preset_direction": state.preset_direction,
        "serve_interval":   state.serve_interval,
        "end_condition":    state.end_condition,
    }, "系统已恢复运行")


# -- 5. 结束 --
@app.post("/stop", summary="结束", tags=["核心控制"])
def api_stop():
    """完全停止系统：停止电机、停止所有线程、断开串口、关闭摄像头。"""
    global smoothed_goal, latest_gk

    state.is_running      = False
    state.is_paused       = False
    state.camera_started  = False
    smoothed_goal = None
    latest_gk = None

    motor.stop_motor()
    time.sleep(0.3)
    motor.disconnect()

    # 清空帧队列
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

    print("[System] 系统已停止")
    return api_ok({"device_state": "stopped"}, "系统已停止")


# -- 6. 变更参数（延迟一球生效）--
@app.post("/params", summary="变更参数", tags=["参数配置"])
def api_update_params(params: UpdateParamsRequest):
    """
    更新发球参数。所有字段可选，仅更新传入字段。

    变更逻辑（延迟一球）：
      发第1球(旧参数) -> 调用变更 -> 发第2球(旧参数) -> 发第3球(新参数)

    即参数在下次发球后生效，不影响当前发球周期。
    """
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
        return api_err(500, "参数错误，请重新上传", {"errors": ["未传入任何参数"]})

    with state.lock:
        if state.pending_params is None:
            state.pending_params = {}
        state.pending_params.update(pending)

    return api_ok({"pending_params": pending}, "参数变更已接收")


# -- 7. 查询摄像机最新图片 --
@app.get("/frames", summary="查询摄像机最新图片", tags=["数据查询"])
async def api_get_frames():
    """
    获取摄像机最新一帧图片（原始 JPEG 二进制字节流）。

    100ms 采集间隔 + ffmpeg H265 解码，ffmpeg 内部缩放到 640x480。
    输出支持可选的等比缩放限幅 + 可调 JPEG 质量。

    成功: 返回 image/jpeg 原始二进制（前端轮询 <img> 标签直接使用）
    失败: 返回 application/json 错误信息
    """
    with latest_frame_lock:
        if latest_frame is None:
            return JSONResponse(
                content=api_err(500, "暂无画面数据（摄像头未启动）"),
                status_code=200,
                media_type="application/json"
            )
        frame = latest_frame.copy()

    # -- 可选等比缩放（保持宽高比，限幅最大边尺寸）--
    if VIDEO_IMAGE_MAX_SIZE_LIMIT:
        h, w = frame.shape[:2]
        max_side = max(w, h)
        if max_side > VIDEO_IMAGE_MAX_SIZE:
            ratio = max_side / VIDEO_IMAGE_MAX_SIZE
            new_w = int(w / ratio)
            new_h = int(h / ratio)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # -- JPEG 编码（质量可配）--
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return JSONResponse(
            content=api_err(500, "图片编码失败"),
            status_code=200,
            media_type="application/json"
        )

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# -- 8. 查询当前状态 --
@app.get("/status", summary="查询当前状态", tags=["数据查询"])
def api_get_status():
    """获取系统完整状态快照。"""
    return api_ok(state.get_snapshot())


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    load_model()

    print("=" * 60)
    print("  智能足球守门员 AI智能模式 API v2.0 (11米点位)")
    print(f"  - RTSP 摄像头: {RTSP_URL}")
    print(f"  - YOLO 模型:   {MODEL_PATH}")
    print(f"  - 串口:        {SERIAL_PORT} @ {SERIAL_BAUD} baud (RS485)")
    print(f"  - 协议:        JSON (V1.2)")
    print(f"  - 图片缩放:    {FRAME_SCALE_WIDTH}x{FRAME_SCALE_HEIGHT} (100ms/帧, 超时{FRAME_READ_TIMEOUT_MS}ms)")
    print(f"  - 视频解码:    CPU 软件解码（极简参数，loglevel=error 屏蔽 GOP 噪音）")
    print(f"  - 网格:        {GOAL_GRID_ROWS}x{GOAL_GRID_COLS}={GOAL_GRID_ROWS*GOAL_GRID_COLS}分区")
    print(f"  - HTTP 端口:   8068")
    print("=" * 60)

    uvicorn.run(app, host="192.168.8.252", port=8068)
