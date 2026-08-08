# -*- coding: utf-8 -*-
"""
MPP 硬件解码模块 — RK3588 VPU 硬解 RTSP 视频流
===================================================
部署位置: /home/ztl/code/test_mpp_player.py

职责:
  1. MPP 硬件解码 RTSP 视频流 (通过 mpp_player SDK)
  2. 提供线程安全的帧获取接口 (get_frame / consume_frame)
  3. 不弹出摄像头窗口，仅提供帧数据给 YOLO 检测和 HTTP /frames 接口

使用方式:
  # 作为模块导入 (由 yolov8_detection.py 的 MppVideoSource 调用)
  import test_mpp_player
  test_mpp_player.start(rtsp_url, display_width=1280, display_height=720)
  frame = test_mpp_player.get_frame()        # 获取当前帧 (不消费, 适合 HTTP 接口)
  frame = test_mpp_player.consume_frame()    # 获取并消费当前帧 (适合 YOLO 检测线程)
  test_mpp_player.stop()

  # 独立运行 (调试)
  python test_mpp_player.py
"""

import sys
import threading
import time

import mpp_player


# ============================================================
# 全局状态
# ============================================================
_current_frame = None
_frame_lock = threading.Lock()
_player = None
_video_w = 0
_video_h = 0
_started = False


# ============================================================
# 回调函数 (由 MPP 内部解码线程调用)
# ============================================================
def _frame_callback(frame_image, scale_image, frame_id, is_rgb):
    """MPP 解码帧回调 — 将最新帧存入全局缓冲"""
    global _current_frame, _video_w, _video_h
    if frame_image is None:
        return
    with _frame_lock:
        _current_frame = frame_image.copy()
    if frame_id <= 1 and _player is not None:
        _video_w = _player.get_width()
        _video_h = _player.get_height()
        print(f"[test_mpp_player] 当前视频分辨率: {_video_w}x{_video_h}")


def _error_callback(error):
    """MPP 错误回调"""
    print(f"[test_mpp_player] 播放错误回调: {error}")


# ============================================================
# 控制 API
# ============================================================
def start(rtsp_url="rtsp://admin:siboasi123@192.168.8.142:554/LiveMedia/ch1/Media1",
          display_width=1280, display_height=720, is_rgb=False):
    """
    启动 MPP 硬件解码 (非阻塞, 内部 spawn daemon 线程)

    参数:
      rtsp_url:      RTSP 流地址
      display_width:  解码输出宽度
      display_height: 解码输出高度
      is_rgb:        输出是否为 RGB 格式

    返回: True=首帧就绪, False=超时/失败
    """
    global _player, _started

    if _started and _player is not None:
        print("[test_mpp_player] 播放器已在运行")
        return True

    _player = mpp_player.MppPlayer()
    _player.set_callback_frame(_frame_callback)
    _player.set_callback_error(_error_callback)
    _player.set_print_fps(True, 100)

    def _play():
        global _started
        state = _player.play(
            rtsp_url,
            display_width=display_width,
            display_height=display_height,
            is_rgb=is_rgb,
        )
        if not state:
            print("[test_mpp_player] 播放结束")
            _started = False

    t = threading.Thread(target=_play, daemon=True, name="test_mpp_player-play")
    t.start()

    # 等待首帧就绪 (最多5秒)
    waited = 0
    while _current_frame is None and waited < 50:
        time.sleep(0.1)
        waited += 1

    _started = _current_frame is not None
    if _started:
        print(f"[test_mpp_player] 首帧就绪 ({_video_w}x{_video_h})")
        return True

    print("[test_mpp_player] 等待首帧超时")
    return False


def get_frame():
    """获取当前帧 (线程安全, 不消费, 适合 HTTP /frames 接口调用)"""
    with _frame_lock:
        return _current_frame.copy() if _current_frame is not None else None


def consume_frame():
    """获取并消费当前帧 (线程安全, 取出后清空, 适合 YOLO 检测线程)"""
    global _current_frame
    with _frame_lock:
        frame = _current_frame
        _current_frame = None
        return frame


def is_running():
    """播放器是否仍在运行"""
    if _player is None:
        return False
    return _player.is_running()


def stop():
    """停止并释放 MPP 解码器"""
    global _player, _started, _current_frame
    if _player is not None:
        try:
            _player.stop()
        except Exception:
            pass
        try:
            _player.close()
        except Exception:
            pass
        _player = None
    _started = False
    with _frame_lock:
        _current_frame = None
    print("[test_mpp_player] 播放器已停止")


def get_video_size():
    """返回视频分辨率 (width, height)"""
    return _video_w, _video_h


# ============================================================
# 独立运行入口 — 不弹出窗口, 仅打印帧信息用于调试
# ============================================================
if __name__ == "__main__":
    print("[test_mpp_player] 独立模式启动...")
    if start():
        print(f"[test_mpp_player] 视频分辨率: {get_video_size()}")
        print("[test_mpp_player] 解码中, 按 Ctrl+C 退出")
        try:
            while is_running():
                frame = get_frame()
                if frame is not None:
                    print(f"[test_mpp_player] 帧: shape={frame.shape}")
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[test_mpp_player] 用户中断, 退出...")
        finally:
            stop()
    else:
        print("[test_mpp_player] 启动失败")
        sys.exit(1)
    print("[test_mpp_player] 程序结束")
