# -*- coding: utf-8 -*-
"""
足球AI智能训练机器 - v5 逐帧跟踪+散点拖尾版
功能：批量上传守门员准备动作视频 → 逐帧球门跟踪 → 球位置发球 → 散点拖尾轨迹 → 输出模拟视频
关键升级：
- ball类检测：从视频中跟踪足球位置作为发球起始点
- 散点拖尾轨迹：粒子散点 + 光晕拖尾效果（替代实线）
- 逐帧球门EMA跟踪：crossbar+post_left+post_right每2帧检测，指数平滑跟踪移动摄像头
- 正常速度播放（无慢放）
"""

import sys
import os
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QFileDialog, QProgressBar, QHBoxLayout, QVBoxLayout, QGroupBox,
    QDialog, QSizePolicy, QScrollArea, QFrame, QGridLayout,
    QMessageBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QSize, QTimer
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QFont, QPen, QIcon, QPalette
)

from ultralytics import YOLO

# ============================================================
# 配置常量
# ============================================================
MODEL_PATH = "runs/detect/train-5/weights/best.pt"
OUTPUT_DIR = "output_videos"

GOAL_GRID_ROWS = 4
GOAL_GRID_COLS = 4
ZONE_COUNT = GOAL_GRID_ROWS * GOAL_GRID_COLS  # 16

COL_PENALTY_LEFT = 0   # zones 1, 5, 9, 13
COL_RECOG_LEFT = 1     # zones 2, 6, 10, 14
COL_RECOG_RIGHT = 2    # zones 3, 7, 11, 15
COL_PENALTY_RIGHT = 3  # zones 4, 8, 12, 16

ALL_ZONES = list(range(1, 17))

# 安全检测：守门员与球门区域重叠阈值
GK_GOAL_OVERLAP_THRESHOLD = 0.15  # GK框至少15%在球门区域内

CLS_GOALKEEPER = 0
CLS_CROSSBAR = 1
CLS_POST_LEFT = 2
CLS_POST_RIGHT = 3
CLS_BALL = 4

HEIGHT_ROW_MAP = {"高": 0, "中": 1, "低": 3, "随机": -1}


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Detection:
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


@dataclass
class GoalBox:
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
# 球门平滑跟踪（EMA指数移动平均）
# ============================================================
def smooth_goal_box(prev: Optional[GoalBox], curr: GoalBox, alpha: float = 0.25) -> GoalBox:
    """
    EMA平滑球门检测框，稳稳跟住移动摄像头的球门：
      alpha=0.25 → 新值权重25%，历史值权重75%，抖动抑制强
      三柱（crossbar + post_left + post_right）共同参与锚定
    """
    if prev is None:
        return curr
    return GoalBox(
        left   = prev.left   * (1.0 - alpha) + curr.left   * alpha,
        top    = prev.top    * (1.0 - alpha) + curr.top    * alpha,
        right  = prev.right  * (1.0 - alpha) + curr.right  * alpha,
        bottom = prev.bottom * (1.0 - alpha) + curr.bottom * alpha,
    )


# ============================================================
# 检测结果提取
# ============================================================
def extract_detections(result) -> List[Detection]:
    dets = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    boxes = result.boxes
    for i in range(len(boxes)):
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        dets.append(Detection(cls, conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
    return dets


def assemble_goal(dets: List[Detection], conf_thresh: float = 0.3
                  ) -> Tuple[Optional[GoalBox], Optional[Detection], Optional[Detection]]:
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
        return None, None, None

    if crossbar and post_left and post_right:
        if post_left.cx > post_right.cx:
            post_left, post_right = post_right, post_left

    left = min(p.x1 for p in parts)
    top = min(p.y1 for p in parts)
    right = max(p.x2 for p in parts)
    bottom = max(p.y2 for p in parts)

    if (right - left) < 30 or (bottom - top) < 30:
        return None, None, None
    return GoalBox(left, top, right, bottom), post_left, post_right


# ============================================================
# 16分区 & 发球目标
# ============================================================
def get_zone_grid(goal: GoalBox) -> Tuple[List[float], List[float]]:
    xs = [goal.left + i * goal.width / GOAL_GRID_COLS for i in range(GOAL_GRID_COLS + 1)]
    ys = [goal.top + i * goal.height / GOAL_GRID_ROWS for i in range(GOAL_GRID_ROWS + 1)]
    return xs, ys


def zone_number(row: int, col: int) -> int:
    return row * GOAL_GRID_COLS + col + 1


def resolve_target_row(height_label: str) -> int:
    row = HEIGHT_ROW_MAP.get(height_label, 1)
    if row == -1:
        row = random.randint(0, GOAL_GRID_ROWS - 1)
    return row


def find_gk_zone(gk_det: Optional[Detection], goal: GoalBox) -> Tuple[int, int, int]:
    """
    计算守门员中心点所在的分区行列号。
    
    Returns
    -------
    (row, col, zone_number) — 如果 GK/Goalkeeper 或 goal 无效则返回 (-1, -1, -1)
    """
    if gk_det is None or goal is None:
        return -1, -1, -1
    xs, ys = get_zone_grid(goal)
    # 找到 GK/Goalkeeper 中心点所在的列
    col = -1
    for c in range(GOAL_GRID_COLS):
        if xs[c] <= gk_det.cx < xs[c + 1]:
            col = c
            break
    if col < 0:
        col = 0 if gk_det.cx < xs[0] else GOAL_GRID_COLS - 1
    # 找到 GK/Goalkeeper 中心点所在的行
    row = -1
    for r in range(GOAL_GRID_ROWS):
        if ys[r] <= gk_det.cy < ys[r + 1]:
            row = r
            break
    if row < 0:
        row = 0 if gk_det.cy < ys[0] else GOAL_GRID_ROWS - 1
    return row, col, zone_number(row, col)


# ============================================================
# 守门员位置识别
# ============================================================
def get_gk_position(gk_det: Optional[Detection], goal: GoalBox
                    ) -> Tuple[bool, bool, Optional[str], float]:
    """判断守门员在球门中的列位置"""
    if gk_det is None:
        return False, False, None, -1.0

    xs, _ = get_zone_grid(goal)
    gk_x1, gk_x2 = gk_det.x1, gk_det.x2

    overlaps = []
    for c in range(GOAL_GRID_COLS):
        col_l, col_r = xs[c], xs[c + 1]
        ov = max(0.0, min(gk_x2, col_r) - max(gk_x1, col_l))
        overlaps.append(ov)

    recog_ov = overlaps[COL_RECOG_LEFT] + overlaps[COL_RECOG_RIGHT]
    penalty_ov = overlaps[COL_PENALTY_LEFT] + overlaps[COL_PENALTY_RIGHT]

    gk_in_recog = recog_ov > 0 and recog_ov >= penalty_ov
    gk_in_penalty = penalty_ov > 0 and penalty_ov > recog_ov

    gk_side = None
    ratio_left = -1.0
    if gk_in_recog and recog_ov > 0:
        ratio_left = overlaps[COL_RECOG_LEFT] / recog_ov
        if ratio_left > 0.55:
            gk_side = "left"
        elif ratio_left < 0.45:
            gk_side = "right"
        else:
            gk_side = "center"
    elif gk_in_penalty and penalty_ov > 0:
        ratio_left = overlaps[COL_PENALTY_LEFT] / penalty_ov
        if ratio_left > 0.5:
            gk_side = "penalty_left"
        else:
            gk_side = "penalty_right"

    return gk_in_recog, gk_in_penalty, gk_side, ratio_left


# ============================================================
# 安全保护检测
# ============================================================
def check_safety(gk_det: Optional[Detection], goal: Optional[GoalBox]) -> Tuple[bool, str]:
    """
    安全检查：
    1. 守门员必须位于球门区域内
    2. 守门员不能距离发球点过近（不在点球区）
    
    Returns:
        (is_safe, reason)
    """
    if gk_det is None:
        return False, "未检测到守门员"

    if goal is None:
        return False, "未检测到球门"

    # 检查1：守门员是否在球门区域内
    gk_box = (gk_det.x1, gk_det.y1, gk_det.x2, gk_det.y2)
    goal_box = (goal.left, goal.top, goal.right, goal.bottom)

    # 计算两个框的重叠面积
    overlap_x1 = max(gk_box[0], goal_box[0])
    overlap_y1 = max(gk_box[1], goal_box[1])
    overlap_x2 = min(gk_box[2], goal_box[2])
    overlap_y2 = min(gk_box[3], goal_box[3])

    if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
        return False, "守门员不在球门区域内"

    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    gk_area = (gk_box[2] - gk_box[0]) * (gk_box[3] - gk_box[1])

    if gk_area <= 0 or (overlap_area / gk_area) < GK_GOAL_OVERLAP_THRESHOLD:
        return False, "守门员不在球门区域内"

    # 检查2：守门员是否距离发球点过近（在点球区内）
    gk_in_recog, gk_in_penalty, gk_side, _ = get_gk_position(gk_det, goal)

    if gk_in_penalty:
        side_name = "左侧点球区" if gk_side == "penalty_left" else "右侧点球区"
        return False, f"守门员位于{side_name}，距离发球点过近"

    if not gk_in_recog and not gk_in_penalty:
        return False, "守门员不在识别区域内"

    return True, "安全"


# ============================================================
# 发球目标判定
# ============================================================
def determine_serve_target(mode: str, gk_side: Optional[str],
                           preset_direction: str, target_row: int
                           ) -> Tuple[int, int]:
    """根据模式和守门员位置确定发球目标"""
    if mode == "随机发球":
        z = random.choice(ALL_ZONES)
        row = (z - 1) // GOAL_GRID_COLS
        col = (z - 1) % GOAL_GRID_COLS
        return col, z

    # 处理 penalty 侧的情况
    if gk_side == "penalty_left":
        near_col, far_col = COL_PENALTY_LEFT, COL_PENALTY_RIGHT
    elif gk_side == "penalty_right":
        near_col, far_col = COL_PENALTY_RIGHT, COL_PENALTY_LEFT
    elif gk_side == "left":
        near_col, far_col = COL_PENALTY_LEFT, COL_PENALTY_RIGHT
    elif gk_side == "right":
        near_col, far_col = COL_PENALTY_RIGHT, COL_PENALTY_LEFT
    else:
        # 居中，根据预设方向
        if preset_direction == "左":
            near_col, far_col = COL_PENALTY_LEFT, COL_PENALTY_RIGHT
        elif preset_direction == "右":
            near_col, far_col = COL_PENALTY_RIGHT, COL_PENALTY_LEFT
        else:
            if random.random() < 0.5:
                near_col, far_col = COL_PENALTY_LEFT, COL_PENALTY_RIGHT
            else:
                near_col, far_col = COL_PENALTY_RIGHT, COL_PENALTY_LEFT

    if mode == "跟随发球":
        target_col = near_col
    elif mode == "反向发球":
        target_col = far_col
    else:
        target_col = near_col

    return target_col, zone_number(target_row, target_col)


# ============================================================
# 球门可视化绘制
# ============================================================
def draw_goal_zones(frame: np.ndarray, goal: GoalBox,
                    target_col: int = -1, target_row: int = -1,
                    target_zone: int = -1,
                    show_zone_numbers: bool = True) -> np.ndarray:
    """
    绘制球门16分区叠加层
    - 默认所有区域显示为黄色网格线
    - 被选中的发球区域以红色高亮显示
    """
    h, w = frame.shape[:2]
    xs, ys = get_zone_grid(goal)

    # 半透明黄色底色覆盖球门区域
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (int(goal.left), int(goal.top)),
                  (int(goal.right), int(goal.bottom)),
                  (0, 220, 220), -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # 目标落点格子红色半透明高亮（仅高亮选中的那一个分区）
    if target_col >= 0 and target_row >= 0:
        tgt_overlay = frame.copy()
        tx1, tx2 = int(xs[target_col]), int(xs[target_col + 1])
        ty1, ty2 = int(ys[target_row]), int(ys[target_row + 1])
        cv2.rectangle(tgt_overlay, (tx1, ty1), (tx2, ty2), (0, 0, 255), -1)
        cv2.addWeighted(tgt_overlay, 0.35, frame, 0.65, 0, frame)
        # 加粗边框
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 4)
        if target_zone > 0:
            cv2.putText(frame, f"TARGET #{target_zone}",
                        (tx1 + 4, ty1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # 网格线（黄色为主）
    for i in range(1, GOAL_GRID_COLS):
        x = int(xs[i])
        is_boundary = (i == 1 or i == 3)
        thick = 2 if is_boundary else 1
        color = (0, 220, 220) if is_boundary else (0, 180, 180)
        cv2.line(frame, (x, int(goal.top)), (x, int(goal.bottom)), color, thick)
    for i in range(1, GOAL_GRID_ROWS):
        y = int(ys[i])
        cv2.line(frame, (int(goal.left), y), (int(goal.right), y), (0, 180, 180), 1)

    # 球门外框（绿色）
    cv2.rectangle(frame, (int(goal.left), int(goal.top)),
                  (int(goal.right), int(goal.bottom)), (0, 255, 0), 3)

    # 分区编号
    if show_zone_numbers:
        for r in range(GOAL_GRID_ROWS):
            for c in range(GOAL_GRID_COLS):
                cx = int((xs[c] + xs[c + 1]) / 2)
                cy = int((ys[r] + ys[r + 1]) / 2)
                cv2.putText(frame, str(zone_number(r, c)), (cx - 8, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_gk_box(frame: np.ndarray, gk_det: Detection, is_safe: bool = True):
    """绘制守门员检测框"""
    color = (0, 255, 100) if is_safe else (0, 140, 255)
    cv2.rectangle(frame, (int(gk_det.x1), int(gk_det.y1)),
                  (int(gk_det.x2), int(gk_det.y2)), color, 2)
    label = "GK-SAFE" if is_safe else "GK-UNSAFE"
    cv2.putText(frame, label, (int(gk_det.x1), int(gk_det.y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_serve_point(frame: np.ndarray, point: Tuple[int, int]):
    """绘制发球起点标记"""
    cv2.circle(frame, point, 7, (0, 200, 200), -1, cv2.LINE_AA)
    cv2.circle(frame, point, 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "SERVE", (point[0] - 24, point[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1, cv2.LINE_AA)


def draw_info_bar(frame: np.ndarray, text: str):
    """底部信息栏"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 255, 200), 1, cv2.LINE_AA)


def draw_mini_zone_display(frame: np.ndarray, target_zone: int,
                           box_size: int = 120):
    """
    在输出视频右下角绘制 16 分区迷你显示框。
    当 target_zone > 0 时点亮对应分区。
    """
    h, w = frame.shape[:2]
    margin = 12
    box_x = w - box_size - margin
    box_y = h - box_size - margin - 30  # 留出底部信息栏空间
    cell_w = box_size // GOAL_GRID_COLS
    cell_h = box_size // GOAL_GRID_ROWS

    # 半透明黑色背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x - 4, box_y - 4),
                  (box_x + box_size + 4, box_y + box_size + 4),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 标题
    cv2.putText(frame, "ZONES", (box_x, box_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # 绘制 4x4 网格
    for r in range(GOAL_GRID_ROWS):
        for c in range(GOAL_GRID_COLS):
            z = zone_number(r, c)
            x1 = box_x + c * cell_w
            y1 = box_y + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            if z == target_zone and target_zone > 0:
                # 点亮目标分区
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(frame, str(z), (x1 + cell_w // 2 - 5, y1 + cell_h // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                cv2.putText(frame, str(z), (x1 + cell_w // 2 - 5, y1 + cell_h // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1, cv2.LINE_AA)


# ============================================================
# 轨迹管理器
# ============================================================
class StraightTrajectoryManager:
    # 轨迹：从发球起点到目标区域的直线运动
    def __init__(self, anim_frames: int = 6, ball_radius: int = 12):
        self.anim_frames = max(3, anim_frames)
        self.ball_radius = max(6, ball_radius)
        self.start: Optional[Tuple[int, int]] = None
        self.end: Optional[Tuple[int, int]] = None
        self.progress: float = 0.0
        self.completed: bool = False
        self.active: bool = False
        self.zone: int = 0
        self.color: Tuple[int, int, int] = (0, 255, 255)  # BGR: 青色
        self._trail: List[Tuple[int, int]] = []  # 拖尾点

    def setup(self, start: Tuple[int, int], end: Tuple[int, int], zone: int):
        self.start = start
        self.end = end
        self.zone = zone
        self.progress = 0.0
        self.completed = False
        self.active = True
        self._trail.clear()

    def step(self):
        if not self.active or self.completed:
            return
        self.progress += 1.0 / self.anim_frames
        if self.progress >= 1.0:
            self.progress = 1.0
            self.completed = True

    def _current_pos(self) -> Tuple[int, int]:
        if not self.start or not self.end:
            return (0, 0)
        x = int(self.start[0] + (self.end[0] - self.start[0]) * self.progress)
        y = int(self.start[1] + (self.end[1] - self.start[1]) * self.progress)
        return (x, y)

    def draw(self, frame: np.ndarray):
        if not self.active:
            return

        pos = self._current_pos()
        self._trail.append(pos)
        # 只保留最近5个拖尾点
        if len(self._trail) > 5:
            self._trail.pop(0)

        # ─── 直线轨迹线 ──────────────────────────
        if self.start and self.end:
            cv2.line(frame, self.start, self.end, (100, 180, 180), 1, cv2.LINE_AA)
        for i, p in enumerate(self._trail[:-1]):
            alpha = (i + 1) / len(self._trail)
            r = max(2, int(self.ball_radius * 0.4 * alpha))
            c = tuple(int(ch * alpha * 0.5) for ch in self.color)
            cv2.circle(frame, p, r, c, -1, cv2.LINE_AA)
        if not self.completed:
            cv2.circle(frame, pos, self.ball_radius + 4,
                       tuple(int(c * 0.2) for c in self.color), -1, cv2.LINE_AA)
            # 球体
            cv2.circle(frame, pos, self.ball_radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pos, self.ball_radius, self.color, 2, cv2.LINE_AA)
            hl = (pos[0] - self.ball_radius // 3, pos[1] - self.ball_radius // 3)
            cv2.circle(frame, hl, max(2, self.ball_radius // 4),
                       (255, 255, 255), -1, cv2.LINE_AA)
        else:
            # ─── 到达目标 ──────────────────────
            cv2.drawMarker(frame, self.end, (0, 220, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, f"#{self.zone}",
                        (self.end[0] + 12, self.end[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        # ─── 起点标记 ────────────────────────────────────
        if self.start:
            cv2.circle(frame, self.start, 3, (140, 220, 140), -1, cv2.LINE_AA)


# ============================================================
# 单个视频处理器（线程）
# ============================================================
class VideoProcessor(QThread):
    """处理单个视频，生成带球门可视化和轨迹模拟的输出视频"""
    progress_updated = pyqtSignal(int, int, int)  # video_index, current, total
    finished = pyqtSignal(int, str, dict)  # video_index, output_path, result_info
    error_occurred = pyqtSignal(int, str)  # video_index, error_msg

    def __init__(self, video_path: str, video_index: int,
                 mode: str, height_label: str, preset_direction: str,
                 model: YOLO, serve_interval: float = 1.0):
        super().__init__()
        self.video_path = video_path
        self.video_index = video_index
        self.mode = mode
        self.height_label = height_label
        self.preset_direction = preset_direction
        self.serve_interval = serve_interval
        self.model = model
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_result.mp4")

        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(self.video_index, "无法打开视频")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

            # ──────────────────────────────────────────────────────
            # 阶段1：帧10-15 检测球门+守门员+足球位置
            # ──────────────────────────────────────────────────────
            detect_start = max(0, min(10, total_frames - 1))
            detect_end = max(0, min(15, total_frames))

            init_goal: Optional[GoalBox] = None
            init_gk: Optional[Detection] = None
            ball_positions: List[Tuple[float, float]] = []  # 球中心点坐标
            ball_sizes: List[float] = []  # 球检测框尺寸（宽+高/2）

            for fi in range(detect_start, detect_end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue

                results = self.model(frame, verbose=False)
                dets = extract_detections(results[0])

                goal, _, _ = assemble_goal(dets)
                if goal:
                    init_goal = goal

                gk = None
                for d in dets:
                    if d.cls == CLS_GOALKEEPER and d.conf >= 0.2:
                        if gk is None or d.conf > gk.conf:
                            gk = d
                if gk:
                    init_gk = gk

                # 🔴 检测足球位置和尺寸（ball类）
                ball = None
                for d in dets:
                    if d.cls == CLS_BALL and d.conf >= 0.15:
                        if ball is None or d.conf > ball.conf:
                            ball = d
                if ball:
                    ball_positions.append((ball.cx, ball.cy))
                    ball_sizes.append((ball.width + ball.height) / 2)

            # ──────────────────────────────────────────────────────
            # 安全检查 + 发球目标判定
            # ──────────────────────────────────────────────────────
            is_safe, safety_reason = check_safety(init_gk, init_goal)

            target_col, target_zone = -1, -1
            target_row = resolve_target_row(self.height_label)
            gk_side: Optional[str] = None
            ratio_left: float = -1.0

            if is_safe and init_goal:
                gk_in_recog, gk_in_penalty, gk_side, ratio_left = \
                    get_gk_position(init_gk, init_goal)

                if self.mode == "跟随发球":
                    # 跟随发球：球直接发往守门员当前位置
                    gk_row, gk_col, gk_zone = find_gk_zone(init_gk, init_goal)
                    if gk_zone > 0:
                        target_row = gk_row
                        target_col = gk_col
                        target_zone = gk_zone
                    else:
                        target_col, target_zone = determine_serve_target(
                            self.mode, gk_side, self.preset_direction, target_row)
                else:
                    target_col, target_zone = determine_serve_target(
                        self.mode, gk_side, self.preset_direction, target_row)

                if target_col < 0:
                    target_col = COL_PENALTY_RIGHT
                    target_zone = zone_number(target_row, target_col)

            # 🔴 发球起始点 = 球位置（中位数），兜底：画面底部中央
            # 球大小 = 检测到的足球尺寸
            ball_radius = 12  # 默认半径
            if ball_positions:
                xs = [p[0] for p in ball_positions]
                ys = [p[1] for p in ball_positions]
                serve_origin = (int(np.median(xs)), int(np.median(ys)))
                if ball_sizes:
                    ball_radius = max(8, int(np.median(ball_sizes)))
            else:
                serve_origin = (frame_w // 2, frame_h - 12)

            # ──────────────────────────────────────────────────────
            # 阶段2：生成输出视频（逐帧球门EMA跟踪 + 直线轨迹）
            # ──────────────────────────────────────────────────────
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            smoothed_goal: Optional[GoalBox] = init_goal
            latest_gk: Optional[Detection] = init_gk

            # 轨迹：直线快速版，球大小=检测到的足球尺寸
            traj_mgr = StraightTrajectoryManager(anim_frames=6, ball_radius=ball_radius)
            trajectory_started = False
            # 发球间隔：serve_interval 秒后开始轨迹
            trajectory_start_frame = min(int(self.serve_interval * fps), total_frames // 2)
            trajectory_completed = False
            truncate_delay_frames = 5  # 轨迹到达后再保留5帧然后截断

            # 逐帧跟踪参数
            TRACK_INTERVAL = 2       # 每2帧跑一次YOLO检测球门
            EMA_ALPHA = 0.25          # EMA平滑系数（越低越稳）

            frame_idx = 0
            frames_after_complete = 0
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── 逐帧球门跟踪 ─────────────────────────────────
                if frame_idx % TRACK_INTERVAL == 0:
                    try:
                        results_f = self.model(frame, verbose=False)
                        dets_f = extract_detections(results_f[0])
                        curr_goal, _, _ = assemble_goal(dets_f, conf_thresh=0.25)
                        if curr_goal:
                            smoothed_goal = smooth_goal_box(smoothed_goal, curr_goal, EMA_ALPHA)
                        # 更新守门员位置
                        gk_f = None
                        for d in dets_f:
                            if d.cls == CLS_GOALKEEPER and d.conf >= 0.2:
                                if gk_f is None or d.conf > gk_f.conf:
                                    gk_f = d
                        if gk_f:
                            latest_gk = gk_f
                    except Exception:
                        pass  # 单帧检测失败不影响整体

                # ── 绘制球门16区网格 ─────────────────────────────
                if smoothed_goal:
                    frame = draw_goal_zones(
                        frame, smoothed_goal,
                        target_col=target_col,
                        target_row=target_row,
                        target_zone=target_zone,
                        show_zone_numbers=True
                    )

                # ── 守门员检测框 ─────────────────────────────────
                if latest_gk:
                    draw_gk_box(frame, latest_gk, is_safe)

                # ── 发球点标记 ───────────────────────────────────
                draw_serve_point(frame, serve_origin)

                # ── 信息栏 ───────────────────────────────────────
                if is_safe:
                    mode_cn = {"跟随发球": "跟随", "反向发球": "反向",
                               "随机发球": "随机"}.get(self.mode, self.mode)
                    gk_label = ({'left': '识别左', 'right': '识别右', 'center': '居中'}
                                .get(gk_side, '未知') if gk_side else '未知')
                    info = (f"视频: {video_name}  |  模式: {mode_cn}  |  "
                            f"高度: {self.height_label}  |  目标: #{target_zone}  |  "
                            f"GK: {gk_label}  |  球起点: ({serve_origin[0]},{serve_origin[1]})")
                else:
                    info = f"视频: {video_name}  |  ⚠ 安全: {safety_reason}"
                draw_info_bar(frame, info)

                # ── 右下角迷你分区显示框 ────────────────────────
                if is_safe and target_zone > 0:
                    draw_mini_zone_display(frame, target_zone)

                # ── 直线轨迹 ────────────────────────────────────
                if is_safe and smoothed_goal and target_zone > 0:
                    if not trajectory_started and frame_idx >= trajectory_start_frame:
                        xs_t, ys_t = get_zone_grid(smoothed_goal)
                        tgt_cx = int((xs_t[target_col] + xs_t[target_col + 1]) / 2)
                        tgt_cy = int((ys_t[target_row] + ys_t[target_row + 1]) / 2)
                        traj_mgr.setup(serve_origin, (tgt_cx, tgt_cy), target_zone)
                        trajectory_started = True

                    if trajectory_started:
                        traj_mgr.step()
                        traj_mgr.draw(frame)
                        if traj_mgr.completed:
                            trajectory_completed = True

                writer.write(frame)
                self.progress_updated.emit(self.video_index, frame_idx + 1, total_frames)
                frame_idx += 1

                # ── 视频截断：轨迹到达后保留几帧再停止 ──────────
                if trajectory_completed:
                    frames_after_complete += 1
                    if frames_after_complete >= truncate_delay_frames:
                        break

            cap.release()
            writer.release()

            # ── 结果信息 ─────────────────────────────────────────
            result_info = {
                "video_name": video_name,
                "is_safe": is_safe,
                "safety_reason": safety_reason,
                "target_zone": target_zone,
                "target_col": target_col,
                "target_row": target_row,
                "gk_side": str(gk_side) if gk_side else "未知",
                "mode": self.mode,
                "height": self.height_label,
                "total_frames": frame_idx,
                "serve_origin": serve_origin,
                "ball_detected": len(ball_positions) > 0,
            }
            self.finished.emit(self.video_index, output_path, result_info)

        except Exception as e:
            self.error_occurred.emit(self.video_index, f"处理异常: {e}")


# ============================================================
# 可点击的视频预览组件
# ============================================================
class VideoResultWidget(QFrame):
    """单个处理结果：缩略图 + 信息 + 点击查看"""
    clicked = pyqtSignal(str, dict)  # video_path, result_info

    def __init__(self, video_path: str, result_info: dict, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.result_info = result_info
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(220, 180)
        self.setMaximumSize(260, 210)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # 缩略图
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setMinimumHeight(120)
        self.thumb_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px solid #3a3a3a;
                border-radius: 6px;
                color: #888;
                font-size: 11px;
            }
        """)
        self.thumb_label.setText("加载中...")
        layout.addWidget(self.thumb_label)

        # 从视频中提取缩略图
        self._load_thumbnail()

        # 信息
        info = self.result_info
        name = info.get("video_name", os.path.basename(self.video_path))
        is_safe = info.get("is_safe", False)
        target = info.get("target_zone", "-")
        mode = info.get("mode", "")

        safe_text = "✅ 安全" if is_safe else "⚠ 警告"
        safe_color = "#4ade80" if is_safe else "#fbbf24"

        info_text = (f"<b>{name[:18]}</b><br>"
                     f"<span style='color:{safe_color}'>{safe_text}</span> | "
                     f"目标 #{target}<br>"
                     f"模式: {mode}")
        self.info_label = QLabel(info_text)
        self.info_label.setStyleSheet("color: #ccc; font-size: 11px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # 点击提示
        hint = QLabel("🖱 点击查看完整视频")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #2d9e5f; font-size: 10px;")
        layout.addWidget(hint)

    def _load_thumbnail(self):
        """从输出视频中间帧提取缩略图"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mid_frame = total // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                    pix = QPixmap.fromImage(qimg)
                    scaled = pix.scaled(220, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
                    self.thumb_label.setPixmap(scaled)
                    return
        except Exception:
            pass
        self.thumb_label.setText("无预览")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.video_path, self.result_info)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet("""
            VideoResultWidget {
                border: 2px solid #2d9e5f;
                border-radius: 8px;
                background-color: #252830;
            }
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("""
            VideoResultWidget {
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                background-color: #1e2128;
            }
        """)
        super().leaveEvent(event)


class VideoPreviewLabel(QLabel):
    """可点击的视频预览标签"""
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class EnlargedVideoDialog(QDialog):
    """放大播放弹窗"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("视频播放 - 点击任意位置关闭")
        self.resize(960, 540)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

    def update_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        pix = self.label.pixmap()
        if pix and not pix.isNull():
            scaled = pix.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            self.label.setPixmap(scaled)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        self.close()


class VideoPlayerDialog(QDialog):
    """播放完整结果视频的弹窗"""

    def __init__(self, video_path: str, result_info: dict, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.result_info = result_info
        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self._playing = False

        info = result_info
        title = f"结果视频 - {info.get('video_name', '')} | 目标 #{info.get('target_zone', '-')}"
        self.setWindowTitle(title)
        self.resize(800, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 视频显示
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 360)
        layout.addWidget(self.video_label)

        # 控制栏
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(8, 4, 8, 4)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #2d9e5f; color: white;
                border: none; border-radius: 4px;
                padding: 6px 14px; font-size: 12px;
            }
            QPushButton:hover { background-color: #35b870; }
        """)
        ctrl_layout.addWidget(self.btn_play)

        self.btn_restart = QPushButton("↺ 重播")
        self.btn_restart.clicked.connect(self._restart)
        self.btn_restart.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a; color: white;
                border: none; border-radius: 4px;
                padding: 6px 14px; font-size: 12px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        ctrl_layout.addWidget(self.btn_restart)

        # 播放速度选择
        ctrl_layout.addWidget(QLabel("速度:"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["1x", "2x", "4x"])
        self.combo_speed.setCurrentText("1x")
        self.combo_speed.currentTextChanged.connect(self._on_speed_changed)
        self.combo_speed.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a; color: white;
                border: 1px solid #444; border-radius: 4px;
                padding: 4px 8px; font-size: 12px;
                min-width: 50px;
            }
        """)
        ctrl_layout.addWidget(self.combo_speed)

        self._playback_speed = 1.0

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444; border-radius: 4px;
                background: #2a2a2a;
            }
            QProgressBar::chunk {
                background: #2d9e5f; border-radius: 3px;
            }
        """)
        ctrl_layout.addWidget(self.progress_bar, stretch=1)

        layout.addLayout(ctrl_layout)

        # 加载第一帧
        self._open_video()

    def _open_video(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.frame_idx = 0
                self._show_frame(0)
        except Exception:
            pass

    def _show_frame(self, frame_idx: int):
        if not self.cap or not self.cap.isOpened():
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame_idx = frame_idx
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled)
        if self.total_frames > 0:
            self.progress_bar.setValue(int(frame_idx / self.total_frames * 100))

    def _next_frame(self):
        if not self._playing:
            return
        next_idx = self.frame_idx + 1
        if next_idx >= self.total_frames:
            self._playing = False
            self.timer.stop()
            self.btn_play.setText("▶ 播放")
            return
        self._show_frame(next_idx)

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self.timer.stop()
            self.btn_play.setText("▶ 播放")
        else:
            if self.frame_idx >= self.total_frames - 1:
                self._restart()
            self._playing = True
            interval = max(1, int(1000 / self.fps / self._playback_speed)) if self.fps > 0 else 33
            self.timer.start(interval)
            self.btn_play.setText("⏸ 暂停")

    def _on_speed_changed(self, text: str):
        """播放速度切换"""
        speed_map = {"1x": 1.0, "2x": 2.0, "4x": 4.0}
        self._playback_speed = speed_map.get(text, 1.0)
        if self._playing and self.fps > 0:
            interval = max(1, int(1000 / self.fps / self._playback_speed))
            self.timer.setInterval(interval)

    def _restart(self):
        self._playing = False
        self.timer.stop()
        self.btn_play.setText("▶ 播放")
        self._show_frame(0)

    def closeEvent(self, event):
        self._playing = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        event.accept()

    def resizeEvent(self, event):
        pix = self.video_label.pixmap()
        if pix and not pix.isNull():
            scaled = pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled)
        super().resizeEvent(event)


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("足球AI智能训练机器 - 批量视频处理")
        self.resize(1250, 850)

        self.video_paths: List[str] = []
        self.processors: Dict[int, VideoProcessor] = {}
        self.model: Optional[YOLO] = None
        self.result_widgets: List[VideoResultWidget] = []
        self.player_dialog: Optional[VideoPlayerDialog] = None
        self.is_processing: bool = False

        self._build_ui()
        self._load_model()

    def _load_model(self):
        """后台加载YOLO模型"""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = YOLO(MODEL_PATH)
                self.lbl_status.setText("模型加载完成，请选择视频文件")
                self.lbl_status.setStyleSheet("color: #4ade80; font-size: 12px;")
            except Exception as e:
                self.lbl_status.setText(f"模型加载失败: {e}")
                self.lbl_status.setStyleSheet("color: #f88; font-size: 12px;")
        else:
            self.lbl_status.setText(f"模型文件不存在: {MODEL_PATH}")
            self.lbl_status.setStyleSheet("color: #fbbf24; font-size: 12px;")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(14, 10, 14, 10)

        # 标题
        title = QLabel("足球AI智能训练机器")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px; font-weight: bold; color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a6b3c, stop:0.5 #2d9e5f, stop:1 #1a6b3c);
                padding: 10px; border-radius: 8px;
            }
        """)
        root.addWidget(title)

        # ---------- 控制面板 ----------
        ctrl_group = QGroupBox("视频处理控制")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(6)

        # 第一行：视频选择
        row1 = QHBoxLayout()
        self.btn_select_videos = QPushButton("📁 选择多个视频文件")
        self.btn_select_videos.setStyleSheet(self._btn_style(primary=True))
        self.btn_select_videos.clicked.connect(self.on_select_videos)
        row1.addWidget(self.btn_select_videos)

        self.btn_clear_videos = QPushButton("清空列表")
        self.btn_clear_videos.setStyleSheet(self._btn_style())
        self.btn_clear_videos.clicked.connect(self.on_clear_videos)
        row1.addWidget(self.btn_clear_videos)

        self.lbl_video_count = QLabel("已选择: 0 个视频")
        self.lbl_video_count.setStyleSheet("color: #aaa; font-size: 12px;")
        row1.addWidget(self.lbl_video_count)
        row1.addStretch()
        ctrl_layout.addLayout(row1)

        # 视频文件列表
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(80)
        self.video_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a; color: #ccc;
                border: 1px solid #444; border-radius: 4px;
                font-size: 11px;
            }
            QListWidget::item { padding: 2px 4px; }
            QListWidget::item:selected { background-color: #2d9e5f; }
        """)
        ctrl_layout.addWidget(self.video_list)

        # 第二行：参数设置
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("发球模式:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["跟随发球", "反向发球", "随机发球"])
        row2.addWidget(self.combo_mode)

        row2.addWidget(QLabel("球的高度:"))
        self.combo_height = QComboBox()
        self.combo_height.addItems(["低", "中", "高", "随机"])
        self.combo_height.setCurrentText("中")
        row2.addWidget(self.combo_height)

        row2.addWidget(QLabel("预设方向:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["随机", "左", "右"])
        row2.addWidget(self.combo_preset)

        row2.addWidget(QLabel("发球间隔(秒):"))
        self.combo_interval = QComboBox()
        self.combo_interval.addItems(["0.5", "1.0", "1.5", "2.0", "3.0", "5.0"])
        self.combo_interval.setCurrentText("1.0")
        row2.addWidget(self.combo_interval)

        row2.addStretch()
        ctrl_layout.addLayout(row2)

        # 第三行：操作按钮
        row3 = QHBoxLayout()
        self.btn_process = QPushButton("▶ 开始批量处理")
        self.btn_process.setStyleSheet(self._btn_style(primary=True, large=True))
        self.btn_process.clicked.connect(self.on_process)
        row3.addWidget(self.btn_process)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self._btn_style(large=True))
        self.btn_stop.clicked.connect(self.on_stop)
        row3.addWidget(self.btn_stop)

        # 批量进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444; border-radius: 6px;
                text-align: center; height: 22px;
                background: #2a2a2a; color: #fff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2d9e5f, stop:1 #4ade80);
                border-radius: 4px;
            }
        """)
        row3.addWidget(self.progress_bar, stretch=1)
        ctrl_layout.addLayout(row3)

        # 状态行
        self.lbl_status = QLabel("就绪 - 请选择视频文件")
        self.lbl_status.setStyleSheet("color: #8f8; font-size: 12px; padding: 4px;")
        ctrl_layout.addWidget(self.lbl_status)

        root.addWidget(ctrl_group)

        # ---------- 处理结果区域 ----------
        result_group = QGroupBox("处理结果（点击查看完整视频）")
        result_outer = QVBoxLayout(result_group)
        result_outer.setContentsMargins(4, 4, 4, 4)

        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.result_scroll.setStyleSheet("""
            QScrollArea {
                border: none; background-color: #1a1c22;
            }
        """)

        self.result_container = QWidget()
        self.result_layout = QGridLayout(self.result_container)
        self.result_layout.setSpacing(10)
        self.result_layout.setContentsMargins(10, 10, 10, 10)
        self.result_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.result_scroll.setWidget(self.result_container)

        # 占位提示
        self.result_placeholder = QLabel("处理完成后，结果视频将在此显示\n每个视频均可点击查看完整效果")
        self.result_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_placeholder.setStyleSheet("color: #555; font-size: 14px; padding: 40px;")
        self.result_layout.addWidget(self.result_placeholder, 0, 0, 1, 1)

        result_outer.addWidget(self.result_scroll)
        root.addWidget(result_group, stretch=1)

        # 整体样式
        self.setStyleSheet("""
            QMainWindow { background-color: #1e2128; }
            QGroupBox {
                color: #ccc; border: 1px solid #3a3a3a; border-radius: 8px;
                margin-top: 12px; padding-top: 10px; font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px;
            }
            QLabel { color: #ddd; }
            QComboBox {
                background-color: #2a2a2a; color: #fff;
                border: 1px solid #444; border-radius: 4px;
                padding: 4px 8px; font-size: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a; color: #fff;
                selection-background-color: #2d9e5f;
            }
        """)

    def _btn_style(self, primary: bool = False, large: bool = False) -> str:
        bg = "#2d9e5f" if primary else "#3a3a3a"
        hover = "#35b870" if primary else "#4a4a4a"
        size = "padding: 8px 18px; font-size: 13px;" if large else "padding: 6px 14px; font-size: 12px;"
        return f"""
            QPushButton {{
                background-color: {bg}; color: white; border: none;
                border-radius: 6px; {size} font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #555; color: #888; }}
        """

    # ============================================================
    # 事件处理
    # ============================================================
    def on_select_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个视频文件（守门员准备动作，1-3秒）", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)")
        if paths:
            self.video_paths = list(paths)
            self._update_video_list()
            self.lbl_status.setText(f"已选择 {len(self.video_paths)} 个视频，点击「开始批量处理」")
            self.lbl_status.setStyleSheet("color: #8f8; font-size: 12px;")

    def on_clear_videos(self):
        self.video_paths.clear()
        self._update_video_list()
        self.lbl_status.setText("视频列表已清空")
        self.lbl_status.setStyleSheet("color: #aaa; font-size: 12px;")

    def _update_video_list(self):
        self.video_list.clear()
        for p in self.video_paths:
            self.video_list.addItem(os.path.basename(p))
        self.lbl_video_count.setText(f"已选择: {len(self.video_paths)} 个视频")

    def on_process(self):
        if self.is_processing:
            return

        if not self.video_paths:
            QMessageBox.warning(self, "提示", "请先选择视频文件")
            return

        if self.model is None:
            QMessageBox.warning(self, "提示", "模型未加载，请检查模型文件路径")
            return

        # 清理之前的结果
        self._clear_results()

        self.is_processing = True
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)

        mode = self.combo_mode.currentText()
        height = self.combo_height.currentText()
        preset = self.combo_preset.currentText()
        serve_interval = float(self.combo_interval.currentText())

        # 为每个视频创建处理器
        for idx, video_path in enumerate(self.video_paths):
            processor = VideoProcessor(
                video_path=video_path,
                video_index=idx,
                mode=mode,
                height_label=height,
                preset_direction=preset,
                model=self.model,
                serve_interval=serve_interval,
            )
            processor.progress_updated.connect(self.on_video_progress)
            processor.finished.connect(self.on_video_finished)
            processor.error_occurred.connect(self.on_video_error)
            self.processors[idx] = processor

        # 依次启动处理（避免GPU冲突）
        self._process_queue: List[int] = list(range(len(self.video_paths)))
        self._process_next()

    def _process_next(self):
        """处理队列中的下一个视频"""
        if not self._process_queue:
            self._on_all_finished()
            return
        idx = self._process_queue.pop(0)
        if idx in self.processors:
            self.lbl_status.setText(f"正在处理视频 {idx + 1}/{len(self.video_paths)}...")
            self.lbl_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
            self.processors[idx].start()

    def on_stop(self):
        for proc in self.processors.values():
            proc.stop()
        self._process_queue.clear()
        self._on_all_finished()

    def on_video_progress(self, video_index: int, current: int, total: int):
        if total > 0:
            # 计算总体进度
            completed_before = sum(1 for i in range(video_index) if i in self.processors)
            overall_pct = int(((completed_before + current / total) / len(self.video_paths)) * 100)
            self.progress_bar.setValue(overall_pct)

    def on_video_finished(self, video_index: int, output_path: str, result_info: dict):
        self.lbl_status.setText(f"视频 {video_index + 1}/{len(self.video_paths)} 处理完成")
        self.lbl_status.setStyleSheet("color: #4ade80; font-size: 12px;")

        # 添加结果组件
        self._add_result_widget(output_path, result_info)

        # 处理下一个
        self._process_next()

    def on_video_error(self, video_index: int, error_msg: str):
        self.lbl_status.setText(f"视频 {video_index + 1} 处理出错: {error_msg}")
        self.lbl_status.setStyleSheet("color: #f88; font-size: 12px;")

        # 创建错误结果组件
        result_info = {
            "video_name": os.path.splitext(os.path.basename(
                self.video_paths[video_index] if video_index < len(self.video_paths) else ""))[0],
            "is_safe": False,
            "safety_reason": error_msg,
            "target_zone": -1,
            "mode": "",
            "height": "",
            "gk_side": "未知",
        }
        # 不添加output_path，只显示错误信息

        self._process_next()

    def _add_result_widget(self, output_path: str, result_info: dict):
        """添加处理结果到结果区域"""
        # 移除占位符
        if self.result_placeholder:
            self.result_layout.removeWidget(self.result_placeholder)
            self.result_placeholder.deleteLater()
            self.result_placeholder = None

        widget = VideoResultWidget(output_path, result_info)
        widget.clicked.connect(self.on_result_clicked)
        self.result_widgets.append(widget)

        # 网格布局：每行最多4个
        cols = 4
        idx = len(self.result_widgets) - 1
        row = idx // cols
        col = idx % cols
        self.result_layout.addWidget(widget, row, col)

    def _clear_results(self):
        """清理所有结果组件"""
        for w in self.result_widgets:
            self.result_layout.removeWidget(w)
            w.deleteLater()
        self.result_widgets.clear()
        self.processors.clear()
        if self.result_placeholder is None:
            self.result_placeholder = QLabel("处理完成后，结果视频将在此显示\n每个视频均可点击查看完整效果")
            self.result_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_placeholder.setStyleSheet("color: #555; font-size: 14px; padding: 40px;")
            self.result_layout.addWidget(self.result_placeholder, 0, 0, 1, 1)

    def on_result_clicked(self, video_path: str, result_info: dict):
        """点击结果视频缩略图"""
        if self.player_dialog:
            self.player_dialog.close()
            self.player_dialog = None
        self.player_dialog = VideoPlayerDialog(video_path, result_info, self)
        self.player_dialog.show()

    def _on_all_finished(self):
        self.is_processing = False
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.progress_bar.value() > 0:
            self.lbl_status.setText("批量处理完成")
            self.lbl_status.setStyleSheet("color: #4ade80; font-size: 12px;")

    def closeEvent(self, event):
        if self.is_processing:
            self.on_stop()
        if self.player_dialog:
            self.player_dialog.close()
        event.accept()


# ============================================================
# 入口
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
