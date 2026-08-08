# -*- coding: utf-8 -*-
"""
MQTT 通信模块 — 发球机 + APP 消息通道
=====================================
部署位置: /home/ztl/code/mqtt.py

职责:
  1. MqttClient — MQTT 通信 (AI ↔ APP + 发球机)
  2. MotorDriver — 发球机 MQTT 驱动 (依赖 MqttClient)
  3. 配置从 cfg/device_protocol.json 读取

 本模块从 yolov8_detection 导入 cfg_get 读取配置。

 使用方式:
   from mqtt import MqttClient, MotorDriver, MQTT_AVAILABLE

   mqtt = MqttClient()
   mqtt.connect()
   motor = MotorDriver(mqtt_client_ref=mqtt)
"""

import json
import time
import threading
import datetime
from typing import Optional, List, Dict, Any

# ── 算法配置 ──
from yolov8_detection import (
    cfg_get,
    DEFAULT_WHEEL_SPEED,
    AUTO_END_TIMEOUT,
    SERVE_INTERVAL_MIN, SERVE_INTERVAL_MAX,
    SERVE_COUNT_MIN, SERVE_COUNT_MAX,
    DURATION_MIN, DURATION_MAX,
)

try:
    import paho.mqtt.client as mqtt_paho
    MQTT_AVAILABLE = True
except ImportError:
    mqtt_paho = None
    MQTT_AVAILABLE = False
    print("[WARN] paho-mqtt 未安装, MQTT 通信不可用")


# ============================================================
# MQTT 配置 — 从 device_protocol.json 读取
# ============================================================
_MQTT_HOST = cfg_get("mqtt", "broker_host", default="192.168.8.75")
_MQTT_PORT = cfg_get("mqtt", "broker_port", default=18883)
_MQTT_USER = cfg_get("mqtt", "username", default="admin")
_MQTT_PASS = cfg_get("mqtt", "password", default="mqtt@123")
_MQTT_CLIENT_ID = cfg_get("mqtt", "client_id", default="mqttx_62f7dbf1_11m")
_MQTT_TOPIC_TX = cfg_get("mqtt", "topic_tx", default="/SS/FB/DMT/001/AI/TX")
_MQTT_TOPIC_RX = cfg_get("mqtt", "topic_rx", default="/SS/FB/DMT/001/AI/RX")
_MACHINE_SUB = cfg_get("device", "subscribe_topic", default="/SS/FB/DMT/001/SubTopic1")
_MACHINE_PUB = cfg_get("device", "publish_topic", default="/SS/FB/DMT/001/PubTopic1")


def get_mqtt_config() -> Dict:
    """返回当前 MQTT 配置摘要"""
    return {
        "host": _MQTT_HOST,
        "port": _MQTT_PORT,
        "user": _MQTT_USER,
        "client_id": _MQTT_CLIENT_ID,
        "topic_tx": _MQTT_TOPIC_TX,
        "topic_rx": _MQTT_TOPIC_RX,
        "machine_sub": _MACHINE_SUB,
        "machine_pub": _MACHINE_PUB,
    }


# ============================================================
# MqttClient — MQTT 消息通信客户端
# ============================================================
class MqttClient:
    """MQTT 客户端，用于 AI 算法 ↔ APP + 发球机 通信"""

    def __init__(self):
        self.client = None
        self.connected = False
        self._paused_start_time: float = 0.0
        self._machine_response: Optional[Dict] = None
        self._machine_response_event = threading.Event()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"[MQTT] 已连接: {_MQTT_HOST}:{_MQTT_PORT}")
            client.subscribe(_MQTT_TOPIC_RX, qos=1)
            print(f"[MQTT] 已订阅 APP 主题: {_MQTT_TOPIC_RX}")
            client.subscribe(_MACHINE_PUB, qos=1)
            print(f"[MQTT] 已订阅发球机主题: {_MACHINE_PUB}")
        else:
            print(f"[MQTT] 连接失败，返回码: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            print(f"[MQTT] 意外断开，返回码: {rc}，将自动重连")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            topic = msg.topic
            if topic == _MACHINE_PUB:
                print(f"[MQTT] 收到发球机消息: {payload}")
                self._handle_machine_message(payload)
            elif topic == _MQTT_TOPIC_RX:
                print(f"[MQTT] 收到 APP 消息: {payload}")
            else:
                print(f"[MQTT] 收到未知主题消息: {topic} -> {payload}")
        except Exception as e:
            print(f"[MQTT] 消息解析异常: {e}")

    def _handle_machine_message(self, payload: Dict):
        pid = payload.get("PID", 0)
        aid = payload.get("AID", 0)
        rst = payload.get("RST", -1)
        if pid == 27:
            rpt = payload.get("RPT", {})
            print(f"[MQTT] 发球完成: 累计{rpt.get('SBCNT', 0)}球")
        elif pid == 26:
            rpt = payload.get("RPT", {})
            print(f"[MQTT] 发球机心跳: 状态={rpt.get('STATE', '?')}")
        elif pid == 58:
            rpt = payload.get("RPT", {})
            print(f"[MQTT] 发球机故障: 告警={rpt.get('ALERT', 0)}")
        elif aid > 0 and rst == 0:
            print(f"[MQTT] 发球机应答: AID={aid}")
            self._machine_response = payload
            self._machine_response_event.set()
        else:
            print(f"[MQTT] 发球机消息: PID={pid}, AID={aid}")

    def publish_to_machine(self, msg: Dict) -> bool:
        if not self.connected or not self.client:
            print("[MQTT] 未连接，无法发送发球机指令")
            return False
        try:
            raw = json.dumps(msg, ensure_ascii=False)
            result = self.client.publish(_MACHINE_SUB, raw, qos=1)
            print(f"[MQTT] TX 发球机: {raw}")
            return result.rc == mqtt_paho.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"[MQTT] 发送异常: {e}")
            return False

    def send_to_machine_and_wait(self, msg: Dict, timeout: float = 2.0) -> Optional[Dict]:
        self._machine_response_event.clear()
        self._machine_response = None
        if not self.publish_to_machine(msg):
            return None
        if self._machine_response_event.wait(timeout=timeout):
            return self._machine_response
        print(f"[MQTT] 发球机应答超时 ({timeout}s)")
        return None

    def connect(self) -> bool:
        if not MQTT_AVAILABLE:
            print("[MQTT] paho-mqtt 未安装，跳过连接")
            return False
        try:
            self.client = mqtt_paho.Client(
                client_id=_MQTT_CLIENT_ID, protocol=mqtt_paho.MQTTv311)
            self.client.username_pw_set(_MQTT_USER, _MQTT_PASS)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
            self.client.connect(_MQTT_HOST, _MQTT_PORT, keepalive=30)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"[MQTT] 连接异常: {e}")
            return False

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.connected = False
        print("[MQTT] 已断开")

    def publish_serve_result(self, success: bool, msg_text: str,
                             serve_count: int, sdata: List[int]):
        payload = {
            "type": "1",
            "data": {
                "serve_sum_count": serve_count,
                "state": success,
                "msg": msg_text,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "SDATA": sdata,
            }
        }
        self._publish(payload)

    def publish_state_change(self, run_state: int):
        payload = {
            "type": "2",
            "data": {
                "run_state": str(run_state),
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
        self._publish(payload)

    def _publish(self, payload: Dict):
        if not self.connected:
            return
        try:
            msg = json.dumps(payload, ensure_ascii=False)
            self.client.publish(_MQTT_TOPIC_TX, msg, qos=1)
            print(f"[MQTT] 已发布: {msg}")
        except Exception as e:
            print(f"[MQTT] 发布异常: {e}")

    def start_pause_timer(self):
        self._paused_start_time = time.time()
        print(f"[MQTT] 暂停计时器已启动（{AUTO_END_TIMEOUT}秒超时）")

    def clear_pause_timer(self):
        self._paused_start_time = 0.0

    def check_pause_timeout(self) -> bool:
        if self._paused_start_time <= 0:
            return False
        return (time.time() - self._paused_start_time) >= AUTO_END_TIMEOUT

    @property
    def host(self) -> str:
        return _MQTT_HOST

    @property
    def port(self) -> int:
        return _MQTT_PORT

    @property
    def topic_tx(self) -> str:
        return _MQTT_TOPIC_TX


# ============================================================
# MotorDriver — MQTT 协议发球机驱动
# ============================================================
class MotorDriver:
    """发球机驱动类，通过 MQTT 协议与发球机通信。"""

    def __init__(self, mqtt_client_ref=None):
        self.mqtt_client = mqtt_client_ref
        self.connected = False
        self._serve_pid_counter = cfg_get("serve", "pid_base", default=10226)
        self._send_lock = threading.Lock()

    def connect(self) -> bool:
        if self.mqtt_client and self.mqtt_client.connected:
            self.connected = True
            print(f"[Motor] 已通过 MQTT 连接发球机")
            return True
        print("[Motor] MQTT 未连接，进入模拟模式")
        self.connected = True
        return False

    def disconnect(self):
        self.connected = False
        print("[Motor] 发球机连接已断开")

    def _build_msg(self, cmd_name: str, extra_mdf: Dict = None,
                   extra_req: List = None, pid_override: int = None) -> Dict:
        cmd_cfg = cfg_get("commands", cmd_name, default={})
        pid = pid_override if pid_override is not None else cmd_cfg.get("PID", 26)
        msg = {"PID": pid, "CKS": 0}
        if "MDF" in cmd_cfg:
            msg["MDF"] = dict(cmd_cfg["MDF"])
        if extra_mdf:
            msg.setdefault("MDF", {}).update(extra_mdf)
        if "REQ" in cmd_cfg:
            msg["REQ"] = list(cmd_cfg["REQ"])
        if extra_req:
            msg["REQ"] = extra_req
        return msg

    def _send_and_recv(self, msg: Dict, timeout: float = 2.0) -> Optional[Dict]:
        if self.mqtt_client and self.mqtt_client.connected:
            with self._send_lock:
                return self.mqtt_client.send_to_machine_and_wait(msg, timeout=timeout)
        print(f"[Motor] 模拟模式，指令未发送: {msg}")
        return None

    def start_motor(self) -> bool:
        print("[Motor] >>> 启动电机 (STATE=work)")
        msg = self._build_msg("start_motor")
        resp = self._send_and_recv(msg)
        if resp and resp.get("RST") == 0:
            print("[Motor] OK 电机已启动")
            return True
        print("[Motor] OK 电机启动指令已发送（模拟/无应答）")
        return True

    def pause_motor(self) -> bool:
        print("[Motor] >>> 暂停电机 (STATE=paus)")
        msg = self._build_msg("pause_motor")
        resp = self._send_and_recv(msg)
        if resp and resp.get("RST") == 0:
            print("[Motor] OK 电机已暂停")
            return True
        print("[Motor] OK 电机暂停指令已发送（模拟/无应答）")
        return True

    def stop_motor(self) -> bool:
        print("[Motor] >>> 停止电机 (STATE=stop)")
        msg = self._build_msg("stop_motor")
        resp = self._send_and_recv(msg)
        if resp and resp.get("RST") == 0:
            print("[Motor] OK 电机已停止")
            return True
        print("[Motor] OK 电机停止指令已发送（模拟/无应答）")
        return True

    def serve(self, wheel1: int = None, wheel2: int = None,
              wheel3: int = None, h_angle: int = 30, v_angle: int = 30) -> int:
        """发球指令。SDATA = [轮1, 轮2, 轮3, 左右角度, 上下角度]"""
        if wheel1 is None:
            wheel1 = DEFAULT_WHEEL_SPEED
        if wheel2 is None:
            wheel2 = DEFAULT_WHEEL_SPEED
        if wheel3 is None:
            wheel3 = DEFAULT_WHEEL_SPEED

        pid = self._serve_pid_counter
        self._serve_pid_counter += 1

        w_min = cfg_get("serve", "wheel_speed", "min", default=20)
        w_max = cfg_get("serve", "wheel_speed", "max", default=100)
        a_min = cfg_get("serve", "h_angle", "min", default=0)
        a_max = cfg_get("serve", "h_angle", "max", default=60)
        v_min = cfg_get("serve", "v_angle", "min", default=0)
        v_max = cfg_get("serve", "v_angle", "max", default=60)

        w1 = max(w_min, min(w_max, wheel1))
        w2 = max(w_min, min(w_max, wheel2))
        w3 = max(w_min, min(w_max, wheel3))
        ha = max(a_min, min(a_max, h_angle))
        va = max(v_min, min(v_max, v_angle))

        sdata = [w1, w2, w3, ha, va]
        print(f"[Motor] >>> 发球 (PID={pid}) SDATA={sdata}")
        msg = {"PID": pid, "MDF": {"SDATA": sdata, "SNEXT": sdata}, "CKS": 0}
        resp = self._send_and_recv(msg)
        if resp and resp.get("RST") == 0:
            print(f"[Motor] OK 发球指令已接受 (PID={pid})")
        else:
            print(f"[Motor] OK 发球指令已发送 (PID={pid})（模拟/无应答）")
        return pid

    def query_status(self) -> Dict:
        msg = self._build_msg("query_status")
        resp = self._send_and_recv(msg, timeout=1.0)
        if resp and resp.get("RES"):
            return resp["RES"]
        return {}
