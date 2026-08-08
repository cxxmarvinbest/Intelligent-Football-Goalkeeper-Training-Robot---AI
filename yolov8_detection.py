# -*- coding: utf-8 -*-
"""
YOLOv8 目标检测算法模块 — RKNN NPU 推理 + MPP 硬件解码
=====================================================
部署位置: /home/ztl/code/yolov8_detection.py
rknn 模型: /home/ztl/code/rknn_model_zoo/examples/yolov8/model/best.rknn

职责:
  1. MPP 硬件解码 RTSP 视频流 (通过 import test_mpp_player, 来自 /home/ztl/code/mpp/)
  2. YAML + JSON 配置加载与合并
  3. RKNN NPU 模型推理 (best.rknn)
  4. 帧检测 → 提取检测结果 (Detection 列表)
  5. 球门框组装 (横梁+左柱+右柱) + EMA 平滑
  6. 3x6=18 分区网格算法
  7. 守门员位置判定 (左/中/右) + 重叠面积计算
  8. 发球目标判定 (反向/跟随/预设方向)
  9. 三重安全检测 (球门内/距离/回中)
 10. 分区→SDATA 发球参数映射


 使用方式:
   # 作为模块导入
   from yolov8_detection import DetectionEngine, MppVideoSource, HAS_MPP

   engine = DetectionEngine()
   engine.load()
   result = engine.process_frame(frame)
   if result.is_safe:
       sdata = engine.get_serve_data(result, serve_mode=0, height=1)

   # 独立运行 (MPP 解码 + NPU 推理)
   python yolov8_detection.py
"""

import os
import sys
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

# ── MPP 硬件解码模块导入 (RK3588 VPU 硬解) ──
# 板卡路径: /home/ztl/code/mpp/
_MPP_PATH = "/home/ztl/code/mpp"
if _MPP_PATH not in sys.path:
    sys.path.insert(0, _MPP_PATH)

HAS_MPP = False

try:
    import test_mpp_player
    HAS_MPP = True
    print(f"[yolov8_detection] MPP 硬件解码模块加载成功 ({_MPP_PATH})")
except ImportError as e:
    HAS_MPP = False
    print(f"[yolov8_detection] 无法导入 test_mpp_player ({_MPP_PATH}): {e}")
    print(f"[yolov8_detection] MPP 硬件解码是必须的，程序退出")
    sys.exit(1)


# ============================================================
# 配置加载 (YAML算法参数 + JSON发球机参数合并, 代码默认值兜底)
# ============================================================
# 运行路径: /home/ztl/code/yolov8_detection.py
_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

_ALGO_YAML_PATH = os.environ.get(
    "ALGO_PARAMS_YAML",
    os.path.join(_CFG_DIR, "algo_params.yaml")
)

_DEVICE_JSON_PATH = os.environ.get(
    "DEVICE_PROTOCOL_JSON",
    os.path.join(_CFG_DIR, "device_protocol.json")
)

_cfg: Dict = {}
_CFG_LOADED = False


def _deep_merge(base: dict, override: dict):
    """递归合并配置, override 覆盖 base"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def load_config() -> Dict:
    """加载 YAML + JSON 配置, 深度合并 (YAML 算法参数 + JSON 发球机参数)

    幂等: 多次调用只加载一次, 返回缓存结果
    """
    global _cfg, _CFG_LOADED
    if _CFG_LOADED:
        return _cfg

    _cfg = {}

    # 1) 加载 JSON (发球机参数: zone_mapping / serve_params 等)
    try:
        with open(_DEVICE_JSON_PATH, "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        print(f"[Config] JSON 已加载: {_DEVICE_JSON_PATH}")
    except FileNotFoundError:
        print(f"[Config] JSON 不存在: {_DEVICE_JSON_PATH}")
    except json.JSONDecodeError as e:
        print(f"[Config] JSON 格式错误: {e}")

    # 2) 加载 YAML (算法参数, 覆盖 JSON 中的同名字段)
    try:
        import yaml
    except ImportError:
        print("[Config] PyYAML 未安装, 跳过 YAML 加载 (pip install pyyaml)")
        _CFG_LOADED = True
        return _cfg

    try:
        with open(_ALGO_YAML_PATH, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
        if yaml_cfg and isinstance(yaml_cfg, dict):
            _deep_merge(_cfg, yaml_cfg)
        print(f"[Config] YAML 已加载: {_ALGO_YAML_PATH}")
    except FileNotFoundError:
        print(f"[Config] YAML 不存在: {_ALGO_YAML_PATH}, 使用代码默认值")
    except Exception as e:
        print(f"[Config] YAML 加载失败: {e}")

    _CFG_LOADED = True
    return _cfg


def cfg_get(*keys, default=None):
    """从合并后的配置字典中逐级取值 (自动触发首次加载)"""
    load_config()
    node = _cfg
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return node


# ============================================================
# 配置常量 (代码默认值, 配置文件可覆盖)
# ============================================================

# -- 检测参数 --
# 默认从本文件所在目录的 rknn_model_zoo/examples/yolov8/model/ 子目录加载 best.rknn
_RKNN_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rknn_model_zoo", "examples", "yolov8", "model")
_default_model_path = os.path.join(_RKNN_MODEL_DIR, "best.rknn")
_cfg_model_path = cfg_get("detection", "model_path", default=_default_model_path)

# 如果配置值为相对路径, 基于脚本目录解析为绝对路径
# (避免 os.path.exists 基于 CWD 解析导致找不到文件)
if _cfg_model_path and not os.path.isabs(_cfg_model_path):
    _cfg_model_path = os.path.join(_CFG_DIR, _cfg_model_path)

MODEL_PATH = _cfg_model_path
RKNN_NPU_CORE = cfg_get("detection", "rknn_npu_core", default=0)
RKNN_INPUT_SIZE = cfg_get("detection", "rknn_input_size", default=640)
TRACK_INTERVAL = cfg_get("detection", "track_interval", default=2)
EMA_ALPHA = cfg_get("detection", "ema_alpha", default=0.25)
GOAL_CONF = cfg_get("detection", "goal_conf", default=0.25)
GK_CONF = cfg_get("detection", "gk_conf", default=0.2)
CLS_GOALKEEPER = cfg_get("detection", "cls_goalkeeper", default=0)
CLS_CROSSBAR = cfg_get("detection", "cls_crossbar", default=1)
CLS_POST_LEFT = cfg_get("detection", "cls_post_left", default=2)
CLS_POST_RIGHT = cfg_get("detection", "cls_post_right", default=3)
CLS_BALL = cfg_get("detection", "cls_ball", default=4)

# -- 网格 --
GOAL_GRID_ROWS = cfg_get("grid", "rows", default=3)
GOAL_GRID_COLS = cfg_get("grid", "cols", default=6)
COL_REVERSE_LEFT = cfg_get("grid", "col_reverse_left", default=0)
COL_CENTER_LEFT = cfg_get("grid", "col_center_left", default=1)
COL_CENTER_RIGHT = cfg_get("grid", "col_center_right", default=4)
COL_REVERSE_RIGHT = cfg_get("grid", "col_reverse_right", default=5)

# -- 安全阈值 --
SAFE_DURATION_BEFORE_SERVE = cfg_get("safety", "safe_duration_before_serve", default=5.0)
SERVE_COOLDOWN = cfg_get("safety", "serve_cooldown", default=2.0)
GK_FRAME_HEIGHT_RATIO_THRESHOLD = cfg_get("safety", "gk_frame_height_ratio_threshold", default=0.45)
GK_MIN_CENTER_RECOG_RATIO = cfg_get("safety", "gk_min_center_recog_ratio", default=0.55)
CENTER_COLUMNS = cfg_get("safety", "center_columns", default=[2, 3])
CENTER_RATIO_MIN = cfg_get("safety", "center_ratio_min", default=0.4)
CENTER_RATIO_MAX = cfg_get("safety", "center_ratio_max", default=0.6)
AUTO_END_TIMEOUT = cfg_get("safety", "auto_end_timeout", default=100)

# -- 发球参数 --
DEFAULT_WHEEL_SPEED = cfg_get("serve_params", "default_wheel_speed", default=50)
SERVE_INTERVAL_MIN = cfg_get("serve_params", "serve_interval_min", default=3)
SERVE_INTERVAL_MAX = cfg_get("serve_params", "serve_interval_max", default=10)
SERVE_COUNT_MIN = cfg_get("serve_params", "serve_count_min", default=1)
SERVE_COUNT_MAX = cfg_get("serve_params", "serve_count_max", default=999)
DURATION_MIN = cfg_get("serve_params", "duration_min", default=1)
DURATION_MAX = cfg_get("serve_params", "duration_max", default=9999)

# -- 发球模式代码 --
SERVE_MODE_REVERSE = 0
SERVE_MODE_FOLLOW = 1

# -- 高度代码 --
HEIGHT_HIGH = 0
HEIGHT_MID = 1
HEIGHT_LOW = 2

# -- 预设方向代码 --
PRESET_RANDOM = 0
PRESET_LEFT = 1
PRESET_RIGHT = 2

# -- 结束条件代码 --
END_FREE = 0
END_COUNT = 1
END_DURATION = 2

# -- 运行状态码 --
RUN_STATE_DEFAULT = 0
RUN_STATE_RUNNING = 1
RUN_STATE_PAUSED = 2
RUN_STATE_ENDED = 3

# -- 分区 SDATA 默认值 --
_DEFAULT_ZONE_SDATA: List[List[List[int]]] = [
    [[50, 50, 50, 10, 45], [50, 50, 50, 18, 45], [50, 50, 50, 26, 45], [50, 50, 50, 34, 45], [50, 50, 50, 42, 45], [50, 50, 50, 50, 45]],
    [[50, 50, 50, 10, 35], [50, 50, 50, 18, 35], [50, 50, 50, 26, 35], [50, 50, 50, 34, 35], [50, 50, 50, 42, 35], [50, 50, 50, 50, 35]],
    [[50, 50, 50, 10, 25], [50, 50, 50, 18, 25], [50, 50, 50, 26, 25], [50, 50, 50, 34, 25], [50, 50, 50, 42, 25], [50, 50, 50, 50, 25]],
]

# 从 JSON 加载分区 SDATA
_json_sdata = cfg_get("zone_mapping", "sdata", default=None)
if _json_sdata and isinstance(_json_sdata, list) and len(_json_sdata) == GOAL_GRID_ROWS:
    valid = True
    for _r in range(GOAL_GRID_ROWS):
        if not isinstance(_json_sdata[_r], list) or len(_json_sdata[_r]) != GOAL_GRID_COLS:
            valid = False
            break
        for _c in range(GOAL_GRID_COLS):
            item = _json_sdata[_r][_c]
            if not isinstance(item, list) or len(item) != 5:
                valid = False
                break
    if valid:
        ZONE_SDATA = _json_sdata
        print(f"[Detection Config] 分区 SDATA 已从 JSON 加载 ({GOAL_GRID_ROWS}x{GOAL_GRID_COLS})")
    else:
        ZONE_SDATA = _DEFAULT_ZONE_SDATA
        print("[Detection Config] JSON 分区 SDATA 格式无效, 使用代码默认值")
else:
    ZONE_SDATA = _DEFAULT_ZONE_SDATA
    print("[Detection Config] JSON 未定义 zone_mapping.sdata, 使用代码默认值")


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


@dataclass
class DetectionResult:
    """单帧检测结果汇总"""
    detections: List[Detection] = field(default_factory=list)
    goal: Optional[GoalBox] = None
    smoothed_goal: Optional[GoalBox] = None
    goalkeeper: Optional[Detection] = None
    is_safe: bool = False
    safety_reason: str = ""
    safety_detail: Dict = field(default_factory=dict)
    gk_side: Optional[str] = None          # "left" / "right" / "center"
    gk_overlaps: List[float] = field(default_factory=list)
    gk_zone: Tuple[int, int, int] = (-1, -1, -1)  # (row, col, zone_number)
    frame_shape: Tuple[int, int] = (0, 0)


# ============================================================
# 核心算法函数
# ============================================================
def smooth_goal_box(prev: Optional[GoalBox], curr: GoalBox,
                    alpha: float = None) -> GoalBox:
    """EMA 指数移动平均平滑球门检测框"""
    if alpha is None:
        alpha = EMA_ALPHA
    if prev is None:
        return curr
    return GoalBox(
        left=prev.left * (1.0 - alpha) + curr.left * alpha,
        top=prev.top * (1.0 - alpha) + curr.top * alpha,
        right=prev.right * (1.0 - alpha) + curr.right * alpha,
        bottom=prev.bottom * (1.0 - alpha) + curr.bottom * alpha,
    )


def extract_detections(result) -> List[Detection]:
    """从推理结果提取检测列表（兼容 Detection 列表和原始元组列表）"""
    dets = []

    if isinstance(result, list):
        for item in result:
            if isinstance(item, Detection):
                dets.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                dets.append(Detection(int(item[0]), float(item[1]),
                                      float(item[2]), float(item[3]),
                                      float(item[4]), float(item[5])))

    return dets


def assemble_goal(dets: List[Detection],
                  conf_thresh: float = None) -> Optional[GoalBox]:
    """从检测结果组装球门框（横梁 + 左柱 + 右柱）"""
    if conf_thresh is None:
        conf_thresh = GOAL_CONF

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

    left = min(p.x1 for p in parts)
    top = min(p.y1 for p in parts)
    right = max(p.x2 for p in parts)
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
    """高度代码 -> 行号（0=高, 1=中, 2=低）"""
    return max(0, min(GOAL_GRID_ROWS - 1, height))


def find_gk_zone(gk_det: Optional[Detection],
                 goal: GoalBox) -> Tuple[int, int, int]:
    """计算守门员中心点所在的分区行列号"""
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
    """计算守门员检测框与各分区的重叠面积，返回最大重叠分区"""
    if gk_det is None or goal is None:
        return -1, -1, -1
    xs, ys = get_zone_grid(goal)

    max_overlap = 0.0
    best_row, best_col = -1, -1

    for r in range(GOAL_GRID_ROWS):
        for c in range(GOAL_GRID_COLS):
            overlap_x1 = max(gk_det.x1, xs[c])
            overlap_y1 = max(gk_det.y1, ys[r])
            overlap_x2 = min(gk_det.x2, xs[c + 1])
            overlap_y2 = min(gk_det.y2, ys[r + 1])

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
    居中判定: ratio = overlap[center_cols[0]] / (overlap[center_cols[0]] + overlap[center_cols[1]])
      CENTER_RATIO_MIN <= ratio <= CENTER_RATIO_MAX -> "center"
    """
    if gk_det is None:
        return None, [0.0] * GOAL_GRID_COLS

    xs, _ = get_zone_grid(goal)
    gk_x1, gk_x2 = gk_det.x1, gk_det.x2

    overlaps = [max(0.0, min(gk_x2, xs[c + 1]) - max(gk_x1, xs[c]))
                for c in range(GOAL_GRID_COLS)]

    # 使用配置的中间列进行居中判定
    col_a = CENTER_COLUMNS[0] if len(CENTER_COLUMNS) >= 1 else 2
    col_b = CENTER_COLUMNS[1] if len(CENTER_COLUMNS) >= 2 else 3
    ov_center = overlaps[col_a] + overlaps[col_b]

    if ov_center > 0:
        ratio_a = overlaps[col_a] / ov_center
        if CENTER_RATIO_MIN <= ratio_a <= CENTER_RATIO_MAX:
            gk_side = "center"
        elif ratio_a > CENTER_RATIO_MAX:
            gk_side = "left"
        else:
            gk_side = "right"
    else:
        left_ov = sum(overlaps[:COL_CENTER_LEFT])
        right_ov = sum(overlaps[COL_CENTER_RIGHT + 1:])
        gk_side = "left" if left_ov > right_ov else "right"

    return gk_side, overlaps


def determine_serve_target(mode: int, gk_side: Optional[str],
                           gk_overlaps: Optional[List[float]],
                           preset_direction: int, target_row: int) -> Tuple[int, int]:
    """
    发球目标判定
    参数:
      mode: 0=反向发球, 1=跟随发球
      gk_side: "left"/"right"/"center"
      preset_direction: 0=随机, 1=左, 2=右
      target_row: 目标行号
    返回: (target_col, target_zone_number)
    """
    import random as _random

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
            target_col = _random.choice([COL_REVERSE_LEFT, COL_REVERSE_RIGHT])

    return target_col, zone_number(target_row, target_col)


def zone_to_sdata(row: int, col: int) -> List[int]:
    """行列号 -> 完整发球参数 [轮1, 轮2, 轮3, 左右角度, 上下角度]"""
    r = max(0, min(GOAL_GRID_ROWS - 1, row))
    c = max(0, min(GOAL_GRID_COLS - 1, col))
    return list(ZONE_SDATA[r][c])


# ============================================================
# 安全检测（三重规则）
# ============================================================
def check_goalkeeper_in_goal(gk_det: Detection,
                             goal: GoalBox) -> Tuple[bool, str, Dict]:
    """规则1: 守门员检测框必须完全位于球门框内"""
    detail = {}
    gk_fully_in_goal = (
        gk_det.x1 >= goal.left and
        gk_det.y1 >= goal.top and
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

    detail["gk_fully_in_goal"] = gk_fully_in_goal
    detail["goal_overlap_ratio"] = round(overlap_ratio, 3)

    if not gk_fully_in_goal:
        return False, "守门员不在球门框内（未完全包含）", detail
    return True, "规则1通过", detail


def check_safe_distance(gk_det: Detection,
                        frame_shape: Tuple[int, int]) -> Tuple[bool, str, Dict]:
    """规则2: 守门员不能距离发球机过近"""
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
    """规则3: 守门员必须回中才能发球"""
    detail = {}
    xs, _ = get_zone_grid(goal)

    recog_left_bound = xs[COL_CENTER_LEFT]
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
# 辅助函数
# ============================================================
def cv2_resize_letterbox(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """等比例缩放 + 填充到正方形 (letterbox)"""
    import cv2
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2_resize(img, (new_w, new_h))

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas


def cv2_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """OpenCV resize 封装"""
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


# ============================================================
# DetectionEngine — 检测引擎（封装完整检测流程）
# ============================================================
class DetectionEngine:
    """
    YOLOv8 检测引擎，封装模型加载、检测、安全判定、发球目标判定全流程。

    用法:
      engine = DetectionEngine()
      engine.load()                          # 加载模型
      result = engine.process_frame(frame)   # 处理一帧
      if result.is_safe:
          sdata = engine.get_serve_data(result, serve_mode=0, preset_direction=1, height=0)
    """

    def __init__(self, model_path: str = None,
                 track_interval: int = None,
                 ema_alpha: float = None):
        raw_path = model_path if model_path is not None else MODEL_PATH
        # 如果传入相对路径, 基于脚本目录解析为绝对路径
        if raw_path and not os.path.isabs(raw_path):
            raw_path = os.path.join(_CFG_DIR, raw_path)
        self.model_path = raw_path
        self.track_interval = track_interval if track_interval is not None else TRACK_INTERVAL
        self.ema_alpha = ema_alpha if ema_alpha is not None else EMA_ALPHA
        self.model = None
        self._letterbox_scale: float = 1.0
        self._letterbox_pad_w: int = 0
        self._letterbox_pad_h: int = 0
        self._smoothed_goal: Optional[GoalBox] = None
        self._latest_gk: Optional[Detection] = None
        self._frame_idx = 0

    # ----------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------
    def load(self) -> bool:
        """加载 RKNN 模型到 NPU"""
        if not os.path.exists(self.model_path):
            print(f"[Detection] 模型文件不存在: {self.model_path}")
            return False
        try:
            from rknnlite.api import RKNNLite
            self.model = RKNNLite()
            ret = self.model.load_rknn(self.model_path)
            if ret != 0:
                print(f"[Detection] load_rknn 失败, ret={ret}")
                return False
            # rknnlite 2.x: init_runtime(core_mask=...) 使用 NPU_CORE_* 常量
            _NPU_CORE_MAP = {
                0: RKNNLite.NPU_CORE_0,
                1: RKNNLite.NPU_CORE_1,
                2: RKNNLite.NPU_CORE_2,
            }
            core_mask = _NPU_CORE_MAP.get(RKNN_NPU_CORE, RKNNLite.NPU_CORE_0)
            ret = self.model.init_runtime(core_mask=core_mask)
            if ret != 0:
                print(f"[Detection] init_runtime 失败, ret={ret}")
                return False
            print(f"[Detection] RKNN 模型加载成功: {self.model_path} (NPU core={RKNN_NPU_CORE})")
            return True
        except ImportError:
            print("[Detection] rknnlite 未安装, 无法加载模型")
            return False
        except Exception as e:
            print(f"[Detection] RKNN 加载失败: {e}")
            return False

    def is_loaded(self) -> bool:
        return self.model is not None

    # ----------------------------------------------------------
    # 推理 (纯 RKNN NPU)
    # ----------------------------------------------------------
    def _run_inference(self, frame: np.ndarray):
        """NPU 推理, 返回 List[Detection] (已做 letterbox 坐标反算)"""
        if self.model is None:
            return None
        try:
            return self._rknn_inference(frame)
        except Exception as e:
            print(f"[Detection] 推理异常: {e}")
            return None

    def _rknn_inference(self, frame: np.ndarray):
        """RKNN NPU 推理: letterbox 预处理 → NPU 推理 → 后处理(含坐标反算)"""
        input_size = RKNN_INPUT_SIZE
        frame_h, frame_w = frame.shape[:2]

        # 记录 letterbox 参数用于后处理坐标反算
        scale = min(input_size / frame_w, input_size / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        self._letterbox_scale = 1.0 / scale
        self._letterbox_pad_w = (input_size - new_w) // 2
        self._letterbox_pad_h = (input_size - new_h) // 2

        # letterbox 缩放 + 填充
        img = cv2_resize_letterbox(frame, input_size)
        # normalize + CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        # NPU 推理
        outputs = self.model.inference(inputs=[img])
        return self._rknn_postprocess(outputs, frame.shape)

    def _rknn_postprocess(self, outputs, frame_shape):
        """RKNN YOLOv8 后处理: (1,84,8400) → Detection 列表 (坐标从 letterbox 反算到原始帧)"""
        try:
            pred = outputs[0]
            if pred.ndim == 3:
                pred = pred[0]  # (84, 8400)
            pred = pred.T       # (8400, 84)

            dets = []
            frame_h, frame_w = frame_shape[:2]
            for row in pred:
                scores = row[4:]
                max_cls = int(np.argmax(scores))
                max_conf = float(scores[max_cls])
                if max_conf < GK_CONF:
                    continue
                # cxcywh (letterbox 坐标系) → xyxy (letterbox)
                cx_lb, cy_lb, w_lb, h_lb = row[0], row[1], row[2], row[3]
                x1_lb = cx_lb - w_lb / 2
                y1_lb = cy_lb - h_lb / 2
                x2_lb = cx_lb + w_lb / 2
                y2_lb = cy_lb + h_lb / 2

                # letterbox → 原始帧坐标反算
                x1 = (x1_lb - self._letterbox_pad_w) * self._letterbox_scale
                y1 = (y1_lb - self._letterbox_pad_h) * self._letterbox_scale
                x2 = (x2_lb - self._letterbox_pad_w) * self._letterbox_scale
                y2 = (y2_lb - self._letterbox_pad_h) * self._letterbox_scale

                # clamp 到帧边界
                x1 = max(0.0, min(float(frame_w), x1))
                y1 = max(0.0, min(float(frame_h), y1))
                x2 = max(0.0, min(float(frame_w), x2))
                y2 = max(0.0, min(float(frame_h), y2))

                dets.append(Detection(int(max_cls), max_conf, x1, y1, x2, y2))
            return dets
        except Exception as e:
            print(f"[Detection] RKNN 后处理异常: {e}")
            return []

    # ----------------------------------------------------------
    # 帧处理
    # ----------------------------------------------------------
    def process_frame(self, frame: np.ndarray,
                      force_detect: bool = False) -> DetectionResult:
        """
        处理一帧视频，返回检测结果汇总。

        参数:
          frame: numpy BGR 帧
          force_detect: 强制运行检测（忽略 track_interval）
        """
        result = DetectionResult()
        result.frame_shape = frame.shape[:2]

        # -- 检测 --
        should_detect = force_detect or (self._frame_idx % self.track_interval == 0)
        if should_detect and self.model is not None:
            detections = self._run_inference(frame)
            if detections is not None:
                result.detections = detections

                # 球门组装 + 平滑
                curr_goal = assemble_goal(result.detections)
                if curr_goal:
                    self._smoothed_goal = smooth_goal_box(
                        self._smoothed_goal, curr_goal, self.ema_alpha
                    )
                result.goal = curr_goal
                result.smoothed_goal = self._smoothed_goal

                # 守门员检测（取置信度最高的）
                for d in result.detections:
                    if d.cls == CLS_GOALKEEPER and d.conf >= GK_CONF:
                        if result.goalkeeper is None or d.conf > result.goalkeeper.conf:
                            result.goalkeeper = d
                if result.goalkeeper:
                    self._latest_gk = result.goalkeeper
            else:
                result.smoothed_goal = self._smoothed_goal
                result.goalkeeper = self._latest_gk
        else:
            result.smoothed_goal = self._smoothed_goal
            result.goalkeeper = self._latest_gk

        # -- 安全检测 --
        result.is_safe, result.safety_reason, result.safety_detail = check_safety(
            result.goalkeeper, result.smoothed_goal, result.frame_shape
        )

        # -- 守门员位置 --
        if result.goalkeeper and result.smoothed_goal:
            result.gk_side, result.gk_overlaps = get_gk_position(
                result.goalkeeper, result.smoothed_goal
            )
            result.gk_zone = find_gk_zone(result.goalkeeper, result.smoothed_goal)

        self._frame_idx += 1
        return result

    # ----------------------------------------------------------
    # 发球目标 + SDATA
    # ----------------------------------------------------------
    def get_serve_data(self, result: DetectionResult,
                       serve_mode: int = SERVE_MODE_REVERSE,
                       preset_direction: int = PRESET_RANDOM,
                       height: int = HEIGHT_MID) -> Dict[str, Any]:
        """
        参数:
          result: process_frame() 的返回值
          serve_mode: 0=反向, 1=跟随
          preset_direction: 0=随机, 1=左, 2=右
          height: 0=高, 1=中, 2=低

        返回:
          {
            "target_row": int,
            "target_col": int,
            "target_zone": int,
            "sdata": [w1, w2, w3, h_angle, v_angle],
          }
        """
        import random as _random

        target_row = resolve_target_row(height)

        if serve_mode == SERVE_MODE_FOLLOW and result.goalkeeper and result.smoothed_goal:
            # 跟随发球: 重叠面积最大的分区
            gk_row, gk_col, target_zone = find_gk_zone_by_overlap(
                result.goalkeeper, result.smoothed_goal
            )
            if target_zone > 0:
                target_row = gk_row
                target_col = gk_col
            else:
                target_col = _random.randint(COL_CENTER_LEFT, COL_CENTER_RIGHT)
                target_zone = zone_number(target_row, target_col)
        elif result.goalkeeper and result.smoothed_goal and result.gk_side:
            target_col, target_zone = determine_serve_target(
                serve_mode,
                result.gk_side,
                result.gk_overlaps,
                preset_direction,
                target_row,
            )
        else:
            target_col = COL_REVERSE_RIGHT
            target_zone = zone_number(target_row, target_col)

        sdata = zone_to_sdata(target_row, target_col)

        return {
            "target_row": target_row,
            "target_col": target_col,
            "target_zone": target_zone,
            "sdata": sdata,
        }

    # ----------------------------------------------------------
    # 状态重置
    # ----------------------------------------------------------
    def reset(self):
        """重置平滑状态"""
        self._smoothed_goal = None
        self._latest_gk = None
        self._frame_idx = 0


# ============================================================
# MPP 视频源 — 基于 test_mpp_player 硬件解码 (RK3588 VPU)
# ============================================================
class MppVideoSource:
    """
    封装 test_mpp_player 硬件解码, 提供线程安全的帧获取接口。

    所有解码逻辑委托给 test_mpp_player 模块 (全局单例播放器),
    本类仅作为接口适配层, 供 http_api 的 camera_capture_thread 调用。

    用法:
      src = MppVideoSource(rtsp_url, display_width=1280, display_height=720)
      src.start()                    # 启动 MPP 解码 (委托 test_mpp_player.start)
      while src.is_running():
          frame = src.get_frame()    # 非阻塞, 返回最新帧或 None
          if frame is not None:
              result = engine.process_frame(frame)
      src.stop()
    """

    def __init__(self, rtsp_url: str,
                 display_width: int = 1280,
                 display_height: int = 720,
                 is_rgb: bool = False):
        self._rtsp = rtsp_url
        self._display_w = display_width
        self._display_h = display_height
        self._is_rgb = is_rgb

    # ----------------------------------------------------------
    # 启动 / 停止 (委托 test_mpp_player)
    # ----------------------------------------------------------
    def start(self) -> bool:
        """启动 MPP 解码 (委托 test_mpp_player.start, 非阻塞返回)"""
        if not HAS_MPP:
            print("[MPP] test_mpp_player 模块不可用")
            return False
        return test_mpp_player.start(
            self._rtsp,
            display_width=self._display_w,
            display_height=self._display_h,
            is_rgb=self._is_rgb,
        )

    def stop(self):
        """停止并释放 MPP 解码器 (委托 test_mpp_player.stop)"""
        test_mpp_player.stop()

    # ----------------------------------------------------------
    # 帧获取 (委托 test_mpp_player)
    # ----------------------------------------------------------
    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新解码帧 (线程安全, 非阻塞, 取出后清空, 适合 YOLO 检测线程)"""
        return test_mpp_player.consume_frame()

    def peek_frame(self) -> Optional[np.ndarray]:
        """查看最新帧但不消费 (适合 HTTP /frames 接口)"""
        return test_mpp_player.get_frame()

    def is_running(self) -> bool:
        """MPP 播放器是否仍在运行"""
        return test_mpp_player.is_running()

    @property
    def video_width(self) -> int:
        w, _ = test_mpp_player.get_video_size()
        return w

    @property
    def video_height(self) -> int:
        _, h = test_mpp_player.get_video_size()
        return h


# ============================================================
# 外部模块配置接口
# ============================================================
def get_config_dir() -> str:
    """返回配置文件目录"""
    return _CFG_DIR


def get_algo_yaml_path() -> str:
    """返回算法 YAML 路径"""
    return _ALGO_YAML_PATH


def get_device_json_path() -> str:
    """返回设备 JSON 路径"""
    return _DEVICE_JSON_PATH


# ============================================================
# 主入口 — MPP 硬件解码 + YOLO 检测实时循环 (独立测试)
# ============================================================
if __name__ == "__main__":
    import cv2

    # ── 从配置加载运行参数 ──
    _rtsp_url = cfg_get("camera", "rtsp_url",
                        default="rtsp://admin:siboasi123@192.168.8.142:554/LiveMedia/ch1/Media1")
    _display_w = cfg_get("camera", "frame_scale_width", default=1280)
    _display_h = cfg_get("camera", "frame_scale_height", default=720)
    _is_rgb = cfg_get("camera", "output_rgb", default=False)

    print("=" * 60)
    print("  YOLOv8 检测算法模块 — RKNN NPU 推理 + MPP 硬件解码")
    print(f"  - RTSP:       {_rtsp_url}")
    print(f"  - 显示尺寸:   {_display_w}x{_display_h}")
    print(f"  - RKNN 模型:  {MODEL_PATH}")
    print(f"  - 输入尺寸:   {RKNN_INPUT_SIZE}x{RKNN_INPUT_SIZE}")
    print(f"  - NPU core:   {RKNN_NPU_CORE}")
    print(f"  - 网格:       {GOAL_GRID_ROWS}x{GOAL_GRID_COLS}={GOAL_GRID_ROWS * GOAL_GRID_COLS}分区")
    print(f"  - 检测间隔:   每{TRACK_INTERVAL}帧")
    print(f"  - 球门置信度: {GOAL_CONF}")
    print(f"  - GK 置信度:  {GK_CONF}")
    print(f"  - 安全持续:   {SAFE_DURATION_BEFORE_SERVE}秒")
    print(f"  - 距离阈值:   {GK_FRAME_HEIGHT_RATIO_THRESHOLD:.0%}")
    print(f"  - 回中阈值:   {GK_MIN_CENTER_RECOG_RATIO:.0%}")
    print(f"  - MPP 可用:   {HAS_MPP}")
    print(f"  - 配置(YAML): {_ALGO_YAML_PATH}")
    print(f"  - 配置(JSON): {_DEVICE_JSON_PATH}")
    print("=" * 60)

    # ── 加载模型 ──
    engine = DetectionEngine(MODEL_PATH)
    if not engine.load():
        print("[ERROR] 模型加载失败, 退出")
        sys.exit(1)

    # ── 启动 MPP 硬件解码 ──
    if not HAS_MPP:
        print("[ERROR] test_mpp_player 模块不可用, 无法启动视频流")
        sys.exit(1)

    video_src = MppVideoSource(
        rtsp_url=_rtsp_url,
        display_width=_display_w,
        display_height=_display_h,
        is_rgb=_is_rgb,
    )
    if not video_src.start():
        print("[ERROR] MPP 视频源启动失败")
        sys.exit(1)

    # ── 检测结果窗口 ──
    cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Detection", _display_w, _display_h)

    print("\n[RUN] 开始实时检测, 按 ESC 退出...")
    fps_counter = 0
    fps_time = time.time()

    try:
        while video_src.is_running():
            frame = video_src.get_frame()
            if frame is None:
                time.sleep(0.005)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # -- YOLO 检测 --
            result = engine.process_frame(frame)

            # -- FPS 统计 --
            fps_counter += 1
            if fps_counter % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - fps_time)
                fps_time = now
                status = f"FPS: {fps:.1f}  "
                if result.gk_side:
                    status += f"GK: {result.gk_side}  "
                if result.is_safe:
                    status += "SAFE"
                else:
                    status += f"UNSAFE({result.safety_reason[:12]}...)"
                print(f"[DETECT] {status}")

            # -- 可视化绘制 --
            display = frame.copy()

            # 画球门框
            if result.smoothed_goal:
                g = result.smoothed_goal
                cv2.rectangle(display, (int(g.left), int(g.top)),
                              (int(g.right), int(g.bottom)), (0, 255, 255), 2)

            # 画网格
            if result.smoothed_goal:
                g = result.smoothed_goal
                row_edges, col_edges = get_zone_grid(g)
                for x in col_edges:
                    cv2.line(display, (int(x), int(g.top)),
                             (int(x), int(g.bottom)), (100, 100, 100), 1)
                for y in row_edges:
                    cv2.line(display, (int(g.left), int(y)),
                             (int(g.right), int(y)), (100, 100, 100), 1)

            # 画守门员框 + 位置标签
            if result.goalkeeper:
                gk = result.goalkeeper
                color = (0, 255, 0) if result.is_safe else (0, 0, 255)
                cv2.rectangle(display, (int(gk.x1), int(gk.y1)),
                              (int(gk.x2), int(gk.y2)), color, 2)
                label = f"GK {result.gk_side or '?'}" + (" SAFE" if result.is_safe else f" {result.safety_reason[:10]}")
                cv2.putText(display, label, (int(gk.x1), int(gk.y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 画所有检测框
            for d in result.detections:
                if d.cls == CLS_GOALKEEPER:
                    continue  # GK 已单独画
                c = (255, 0, 0) if d.cls == CLS_BALL else \
                    (0, 200, 200) if d.cls == CLS_CROSSBAR else (0, 0, 200)
                cv2.rectangle(display, (int(d.x1), int(d.y1)),
                              (int(d.x2), int(d.y2)), c, 1)

            cv2.imshow("YOLOv8 Detection", display)

            if cv2.waitKey(1) == 27:  # ESC
                print("\n[USER] 用户按下 ESC, 退出...")
                break

    except KeyboardInterrupt:
        print("\n[USER] Ctrl+C, 退出...")
    finally:
        print("[CLEANUP] 释放资源...")
        video_src.stop()
        cv2.destroyAllWindows()
        print("[CLEANUP] 程序结束")
