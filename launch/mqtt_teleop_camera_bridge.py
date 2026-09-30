#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
*****************************************************************************
@file      mqtt_teleop_camera_bridge.py
@brief     MQTT Bridge điều khiển Camera RTSP & Joystick Teleop cho Robot
@details   - Nhận lệnh MQTT từ SaaS/App để điều khiển bật/tắt camera (ros2_to_rtsp).
           - Nhận lệnh Joystick từ SaaS/App để điều khiển di chuyển robot (/cmd_vel).
           - Tự động gửi xung Keepalive gia hạn 5 phút cho camera khi đang gạt joystick.
           - Tích hợp Watchdog an toàn: Tự dừng xe (/cmd_vel = 0) nếu mất tín hiệu joystick > 0.8s.
*****************************************************************************
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Empty
import math
import time
import json
import os
import sys
import paho.mqtt.client as mqtt
from datetime import datetime
import threading

# --- HỖ TRỢ TIN NHẮN VẬT CẢN SWIX M1 ---
try:
    from my_robot_msgs.msg import ObstacleStatus
    HAS_OBSTACLE_STATUS = True
except ImportError:
    HAS_OBSTACLE_STATUS = False

# -------- CẤU HÌNH ĐỌC TỪ robot_config.json --------
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_config.json")

def load_robot_config():
    """Đọc cấu hình định danh xe và thông số MQTT từ robot_config.json. Thoát app ngay nếu lỗi!"""
    if not os.path.exists(CONFIG_FILE):
        print(f"\n{'='*70}", flush=True)
        print(f"❌ [LỖI NGHIÊM TRỌNG] Không tìm thấy file cấu hình: {CONFIG_FILE}", flush=True)
        print("❌ Vui lòng tạo file 'robot_config.json' hợp lệ trước khi khởi chạy app!", flush=True)
        print(f"🛑 ĐANG TẮT ỨNG DỤNG THEO YÊU CẦU AN TOÀN...\n{'='*70}\n", flush=True)
        sys.exit(1)

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if not isinstance(cfg, dict):
            raise ValueError(f"Nội dung file {CONFIG_FILE} không phải là đối tượng JSON hợp lệ (dict).")

        if not cfg.get("robot_id"):
            raise ValueError(f"Thiếu hoặc trống trường 'robot_id' trong file {CONFIG_FILE}.")

        if "mqtt" not in cfg or not isinstance(cfg.get("mqtt"), dict):
            raise ValueError(f"Thiếu hoặc sai định dạng khối cấu hình 'mqtt' trong file {CONFIG_FILE}.")

        return cfg
    except Exception as e:
        print(f"\n{'='*70}", flush=True)
        print(f"❌ [LỖI NGHIÊM TRỌNG] Lỗi phân tích file cấu hình '{CONFIG_FILE}': {e}", flush=True)
        print(f"🛑 ĐANG TẮT ỨNG DỤNG THEO YÊU CẦU AN TOÀN...\n{'='*70}\n", flush=True)
        sys.exit(1)

ROBOT_CONFIG = load_robot_config()
ROBOT_ID = ROBOT_CONFIG.get("robot_id", "")
MQTT_CFG = ROBOT_CONFIG.get("mqtt", {})

BROKER = MQTT_CFG.get("broker", "")
PORT = int(MQTT_CFG.get("port", 0))
BASE_TOPIC = MQTT_CFG.get("base_topic", "")
TOPIC = f"{BASE_TOPIC}/{ROBOT_ID}"
USERNAME = MQTT_CFG.get("username", "")
PASSWORD = MQTT_CFG.get("password", "")
CLIENT_ID = f"python-teleop-{ROBOT_ID}-" + str(int(time.time()))
MQTT_RECONNECT_WATCHDOG_SEC = int(MQTT_CFG.get("reconnect_watchdog_sec", 60))
# ----------------------------------------------------

# -------- THÔNG SỐ HÌNH HỌC ROBOT CYBER SWIX M1 (ĐỒNG BỘ avoidance_config.py & URDF) --------
ROBOT_WIDTH = 0.90                    # (m) Bề ngang tổng (đã tính 2 chổi xòe ra 2 bên)
ROBOT_HALF_WIDTH = ROBOT_WIDTH / 2.0  # (m) Nửa bề ngang xe (0.45m)
ROBOT_FRONT_X = 1.11                  # (m) Khoảng cách từ tâm trục bánh sau tới mũi xe ngoài cùng
ROBOT_REAR_X = 0.35                   # (m) Từ tâm trục bánh sau tới đuôi xe
ROBOT_TOTAL_LENGTH = ROBOT_FRONT_X + ROBOT_REAR_X  # 1.46m
WHEELBASE = 1.00                      # (m) Khoảng cách giữa 2 trục bánh (trước - sau)

MAX_STEER_DEG = 40.0                  # (độ) Góc quay tối đa của cơ cấu lái Festo (~40 độ)
MAX_STEER_RAD = math.radians(MAX_STEER_DEG)  # ~0.6981 rad

SAFETY_OVERRIDE_CRAWL_SPEED = 0.08    # (m/s) Vận tốc bò an toàn khi ép vượt cản đi thẳng
SAFETY_EVASION_SPEED = 0.15           # (m/s) Vận tốc ga tiến tối đa khi bẻ lái né cản
# -------------------------------------------------------------------------------------------

def log_msg(message: str):
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"{timestamp} - {message}", flush=True)

def safe_float(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None

class MQTTTeleopCameraBridge(Node):
    def __init__(self):
        super().__init__("mqtt_teleop_camera_bridge")

        # 1. Publisher điều khiển vận tốc xe (Joystick Teleop)
        # Đối với SwiX sẽ dùng topic /tablet_cmd_vel để gửi lệnh điều khiển 
        self.cmd_vel_pub = self.create_publisher(Twist, "/tablet_cmd_vel", 10)

        # --- BỘ DUY TRÌ TẦN SỐ 50 HZ (LẶP LẠI LỆNH GẦN NHẤT) ---
        self.PUB_FREQ = 50.0  # Tần số phát lệnh cho động cơ: 50 Hz (cứ 20ms phát 1 lần)
        self.LOOP_DT = 1.0 / self.PUB_FREQ

        # Lưu đúng giá trị vận tốc gần nhất từ Web/MQTT gửi đến
        self.cmd_linear_x = 0.0
        self.cmd_angular_z = 0.0
        self.cmd_linear_y = 0.0

        self.last_joy_time = 0.0
        self.joy_active = False
        self.joy_override = False      # Cờ ép vượt cản (Safety Override) từ Web/App
        self.last_override_warn_time = 0.0
        self.zero_cmd_sent_count = 20  # Đếm số gói 0.0 phát ra khi dừng hẳn

        # --- QUẢN LÝ QUYỀN LÁI XE ĐỘC QUYỀN (SINGLE-PILOT EXCLUSIVE LEASE LOCK) ---
        self.active_driver_id = None
        self.driver_lock_expire = 0.0
        self.DRIVER_LEASE_SEC = 2.5    # 2.5s không có lệnh joystick -> tự động nhả quyền lái
        self.last_driver_reject_time = 0.0

        # Vòng lặp duy trì phát đúng lệnh gần nhất liên tục 50 Hz xuống động cơ
        self.smooth_timer = self.create_timer(self.LOOP_DT, self.smooth_control_loop)

        # 2. Publisher điều khiển Camera Stream sang node ros2_to_rtsp
        self.cam_select_pub = self.create_publisher(
            String,
            "/multi_camera_rtsp_streamer/select_cameras",
            10
        )
        self.cam_keepalive_pub = self.create_publisher(
            Empty,
            "/multi_camera_rtsp_streamer/keepalive",
            10
        )
        self.last_cam_keepalive_time = 0.0

        # 3. LỚP BẢO VỆ AN TOÀN TRÁNH VA CHẠM (COLLISION SAFETY GUARD)
        self.node_start_time = time.time()
        self.front_blocked = False
        self.rear_blocked = False
        self.min_front_dist = 999.0
        self.min_rear_dist = 999.0
        self.obstacle_level = 0  # 0=CLEAR, 1=WARN, 2=BLOCK, 3=CRITICAL, 4=DEGRADED
        self.front_obs_width = 0.0
        self.front_obs_offset_y = 0.0
        self.front_obs_edge_left = 0.0
        self.front_obs_edge_right = 0.0
        self.range_front_left_detected = False
        self.range_front_right_detected = False

        self.last_collision_time = 0.0
        self.last_front_warn_time = 0.0
        self.last_rear_warn_time = 0.0
        self.last_steer_warn_time = 0.0
        self.last_degraded_warn_time = 0.0

        # Subscribe trực tiếp trạng thái vật cản chuẩn của SwiX M1 (/perception/obstacle_status)
        if HAS_OBSTACLE_STATUS:
            self.sub_obstacle = self.create_subscription(
                ObstacleStatus,
                "/perception/obstacle_status",
                self.on_obstacle_status,
                10
            )
            self.get_logger().info("🛡️ Đã kết nối hệ thống an toàn vật cản từ '/perception/obstacle_status'")
        else:
            self.get_logger().warn("⚠️ Chưa có package my_robot_msgs, lớp bảo vệ né vật cản tạm thời tắt.")

        self.get_logger().info("=== MQTT Teleop & Camera Bridge Node Sẵn Sàng ===")
        self.get_logger().info("Đã kết nối các cổng ROS 2: /tablet_cmd_vel (Duy trì 50Hz) và /multi_camera_rtsp_streamer/*")

    def set_mqtt_client(self, client_holder=None, topic: str = ""):
        """Không gửi phản hồi an toàn lên Web qua MQTT theo yêu cầu (giữ method để tương thích gọi ngoài)"""
        pass

    def on_obstacle_status(self, msg: ObstacleStatus):
        """Tiếp nhận trực tiếp trạng thái an toàn từ obstacle_detector của SwiX M1"""
        self.last_collision_time = time.time()

        self.obstacle_level = int(msg.level)
        self.front_blocked = bool(msg.front_blocked)
        self.rear_blocked = bool(msg.rear_blocked)
        self.min_front_dist = float(msg.front_min_dist) if math.isfinite(msg.front_min_dist) else 999.0
        self.min_rear_dist = float(msg.rear_min_dist) if math.isfinite(msg.rear_min_dist) else 999.0

        # Kích thước và mép vật cản phía trước từ LiDAR
        self.front_obs_width = float(msg.front_obstacle_width) if math.isfinite(msg.front_obstacle_width) else 0.0
        self.front_obs_offset_y = float(msg.front_obstacle_offset_y) if math.isfinite(msg.front_obstacle_offset_y) else 0.0
        self.front_obs_edge_left = float(msg.front_obstacle_edge_left) if math.isfinite(msg.front_obstacle_edge_left) else 0.0
        self.front_obs_edge_right = float(msg.front_obstacle_edge_right) if math.isfinite(msg.front_obstacle_edge_right) else 0.0

        # Trạng thái siêu âm góc trước trái / phải
        try:
            self.range_front_left_detected = bool(msg.range_front_left.detected)
            self.range_front_right_detected = bool(msg.range_front_right.detected)
        except Exception:
            self.range_front_left_detected = False
            self.range_front_right_detected = False

    def apply_safety_guard(self, linear_x: float, angular_z: float):
        """
        Lớp bảo vệ kiểm tra vật cản cho SwiX M1:
        - Level 4 (DEGRADED / Mất cảm biến / Timeout): KHÓA CỨNG XE, CẤM ĐÈ (Override bị vô hiệu hóa).
        - Khi 1 trong 2 cảm biến siêu âm trước phát hiện cản (<=0.65m) hoặc cản quá sát mũi xe (<0.45m):
            + KHÓA CỨNG cả ga tiến và bẻ lái, bắt buộc phải LÙI xe (đảm bảo an toàn thực tế đầu xe 3 bánh không quét mép).
        - Khi chỉ LiDAR phát hiện cản ở khoảng cách cho phép né (>=0.45m, siêu âm không báo):
            + Cản lệch bên trái: Cấm bẻ trái; Cho phép bẻ lái PHẢI né cản (cấp tốc độ bò 0.15 m/s + góc lái tối đa 40°).
            + Cản lệch bên phải: Cấm bẻ phải; Cho phép bẻ lái TRÁI né cản (cấp tốc độ bò 0.15 m/s + góc lái tối đa 40°).
            + Cản chính diện tim xe: Cho phép bẻ lái trái/phải né cản; Khóa đi thẳng (chỉ cho đi thẳng nếu bật joy_override=True).
        - Lùi xe: Tự do lùi và đánh lái né cản (trừ khi cảm biến sau báo cản).
        """
        if not HAS_OBSTACLE_STATUS:
            return linear_x, angular_z

        # Nếu không có lệnh điều khiển (xe đang dừng/người lái không gạt cần):
        # Trả về ngay (0.0, 0.0), tuyệt đối không chạy kiểm tra và không in bất kỳ log nào!
        if abs(linear_x) <= 0.001 and abs(angular_z) <= 0.001:
            return 0.0, 0.0

        now = time.time()
        is_moving_cmd = True

        # 1. KIỂM TRA MẤT KẾT NỐI / TIMEOUT CẢM BIẾN (Tương đương Level 4 DEGRADED)
        sensor_offline = False
        if self.last_collision_time == 0.0:
            if (now - self.node_start_time) > 3.0:
                sensor_offline = True
        elif (now - self.last_collision_time) > 1.2:
            sensor_offline = True

        # 2. XỬ LÝ LEVEL 4 (LEVEL_DEGRADED) HOẶC SENSOR OFFLINE -> CẤM ĐÈ TUYỆT ĐỐI!
        if self.obstacle_level == 4 or sensor_offline:
            if is_moving_cmd and (now - self.last_degraded_warn_time > 2.0):
                self.last_degraded_warn_time = now
                if self.obstacle_level == 4:
                    log_msg("🛑 [SAFETY LOCKOUT - LV4 DEGRADED] Cảm biến báo lỗi/mất an toàn (Level 4) -> KHÓA CỨNG XE, CẤM VƯỢT CẢN (Override bị vô hiệu hóa)!")
                else:
                    log_msg("🛑 [SAFETY LOCKOUT - SENSOR LOSS] Mất kết nối /perception/obstacle_status (>1.2s) -> KHÓA CỨNG XE, CẤM VƯỢT CẢN!")
            return 0.0, 0.0

        safe_vx = linear_x
        safe_wz = max(-MAX_STEER_RAD, min(float(angular_z), MAX_STEER_RAD))

        # 3. KIỂM TRA VẬT CẢN PHÍA SAU KHI LÙI XE (safe_vx < 0)
        if safe_vx < -0.001:
            if self.rear_blocked:
                if self.joy_override:
                    safe_vx = max(safe_vx, -SAFETY_OVERRIDE_CRAWL_SPEED)
                    if now - self.last_override_warn_time > 1.5:
                        self.last_override_warn_time = now
                        log_msg(f"⚠️ [SAFETY OVERRIDE] Cản phía SAU (~{self.min_rear_dist:.2f}m) -> ÉP LÙI tốc độ rùa ({safe_vx:.2f} m/s)!")
                else:
                    safe_vx = 0.0
                    if now - self.last_rear_warn_time > 1.0:
                        self.last_rear_warn_time = now
                        log_msg(f"🛑 [SAFETY GUARD] Có vật cản phía SAU (~{self.min_rear_dist:.2f}m) -> Khóa ga lùi! Cho phép tiến/xoay.")
            # Khi lùi xe an toàn: Cho phép tự do bẻ lái để de xe né cản
            return safe_vx, safe_wz

        # 4. KIỂM TRA VẬT CẢN PHÍA TRƯỚC (KHI TIẾN HOẶC XOAY/BẺ LÁI TIẾN)
        us_left = self.range_front_left_detected
        us_right = self.range_front_right_detected
        us_triggered = (us_left or us_right)

        has_front_obstacle = self.front_blocked or (0.0 < self.min_front_dist < 1.0) or us_triggered
        if has_front_obstacle:
            y_left = self.front_obs_edge_left
            y_right = self.front_obs_edge_right
            dist = self.min_front_dist

            # Tường chắn ngang trọn vẹn hành lang 0.9m xe (trái >= 0.40m, phải <= -0.40m)
            wall_blocking = (y_left >= 0.40 and y_right <= -0.40)

            # Trường hợp A:
            # - Bất kỳ con siêu âm trước nào phát hiện (us_left hoặc us_right, cự ly <= 0.65m ngay góc mũi xe)
            # - Hoặc khoảng cách LiDAR quá sát mũi xe (< 0.45m)
            # - Hoặc tường chắn ngang toàn bộ bề ngang xe (<= 0.65m)
            # => ĐẶC BIỆT THEO YÊU CẦU: 1 trong 2 siêu âm phát hiện là KHÓA CỨNG CẢ TIẾN VÀ BẺ LÁI!
            #    Xe 3 bánh bẻ lái lúc này góc mũi xe sẽ quét trúng cản. Bắt buộc LÙI xe để đảm bảo an toàn thực tế!
            if us_triggered or dist < 0.45 or (wall_blocking and dist <= 0.65):
                if safe_vx > 0.0 or abs(safe_wz) > 0.001:
                    safe_vx = 0.0
                    safe_wz = 0.0
                    if now - self.last_front_warn_time > 1.5:
                        self.last_front_warn_time = now
                        if us_left and us_right:
                            source = "Cả 2 cảm biến siêu âm trước"
                        elif us_left:
                            source = "Cảm biến siêu âm trước TRÁI"
                        elif us_right:
                            source = "Cảm biến siêu âm trước PHẢI"
                        elif dist < 0.45:
                            source = f"Vật cản sát mũi xe (~{dist:.2f}m)"
                        else:
                            source = f"Tường chắn toàn bộ phía trước (~{dist:.2f}m)"
                        log_msg(f"🛑 [SAFETY GUARD] {source} phát hiện cản -> Khóa cứng ga tiến & bẻ lái! Vui lòng LÙI xe để đảm bảo an toàn thực tế.")
                return safe_vx, safe_wz

            # Trường hợp B: Cản lệch bên TRÁI (từ LiDAR khi cả 2 siêu âm không phát hiện)
            obs_on_left = (y_left > 0.15 and y_right > -0.35)

            # Trường hợp C: Cản lệch bên PHẢI (từ LiDAR khi cả 2 siêu âm không phát hiện)
            obs_on_right = (y_right < -0.15 and y_left < 0.35)

            # Kiểm tra hướng người lái đang muốn bẻ lái:
            is_steering_left = (safe_wz > 0.05)
            is_steering_right = (safe_wz < -0.05)

            if obs_on_left:
                if is_steering_left:
                    # Đang có cản bên TRÁI mà cố bẻ TRÁI -> CẤM!
                    safe_wz = 0.0
                    safe_vx = 0.0
                    if now - self.last_steer_warn_time > 1.5:
                        self.last_steer_warn_time = now
                        log_msg(f"⚠️ [STEERING GUARD] Vật cản phía TRƯỚC-TRÁI (~{dist:.2f}m) -> CẤM bẻ trái! Hãy bẻ lái sang PHẢI để né cản hoặc LÙI xe.")
                elif is_steering_right:
                    # Bẻ PHẢI né cản -> CHO PHÉP! Xuất đồng thời cả góc bẻ lái Festo (tối đa 40°) và ga tiến (tối đa 0.15 m/s)
                    safe_vx = min(safe_vx, SAFETY_EVASION_SPEED) if safe_vx > 0.0 else SAFETY_OVERRIDE_CRAWL_SPEED
                    if now - self.last_steer_warn_time > 1.5:
                        self.last_steer_warn_time = now
                        log_msg(f"↪️ [STEERING EVASION] Bẻ lái né cản sang PHẢI (~{dist:.2f}m) -> Cấp ga tiến ({safe_vx:.2f} m/s) cùng góc lái ({math.degrees(safe_wz):.1f}°)!")
                else:
                    # Đi thẳng vào cản bên trái
                    if self.joy_override:
                        safe_vx = min(safe_vx, SAFETY_OVERRIDE_CRAWL_SPEED)
                        if now - self.last_override_warn_time > 1.5:
                            self.last_override_warn_time = now
                            log_msg(f"⚠️ [SAFETY OVERRIDE] Cản phía TRƯỚC (~{dist:.2f}m) -> ÉP TIẾN tốc độ rùa ({safe_vx:.2f} m/s)!")
                    else:
                        safe_vx = 0.0
                        if now - self.last_front_warn_time > 1.0:
                            self.last_front_warn_time = now
                            log_msg(f"🛑 [SAFETY GUARD] Có vật cản phía TRƯỚC-TRÁI (~{dist:.2f}m) -> Khóa ga tiến thẳng! Hãy bẻ lái sang PHẢI né cản hoặc lùi xe.")

            elif obs_on_right:
                if is_steering_right:
                    # Đang có cản bên PHẢI mà cố bẻ PHẢI -> CẤM!
                    safe_wz = 0.0
                    safe_vx = 0.0
                    if now - self.last_steer_warn_time > 1.5:
                        self.last_steer_warn_time = now
                        log_msg(f"⚠️ [STEERING GUARD] Vật cản phía TRƯỚC-PHẢI (~{dist:.2f}m) -> CẤM bẻ phải! Hãy bẻ lái sang TRÁI để né cản hoặc LÙI xe.")
                elif is_steering_left:
                    # Bẻ TRÁI né cản -> CHO PHÉP! Xuất đồng thời cả góc bẻ lái Festo (tối đa 40°) và ga tiến (tối đa 0.15 m/s)
                    safe_vx = min(safe_vx, SAFETY_EVASION_SPEED) if safe_vx > 0.0 else SAFETY_OVERRIDE_CRAWL_SPEED
                    if now - self.last_steer_warn_time > 1.5:
                        self.last_steer_warn_time = now
                        log_msg(f"↩️ [STEERING EVASION] Bẻ lái né cản sang TRÁI (~{dist:.2f}m) -> Cấp ga tiến ({safe_vx:.2f} m/s) cùng góc lái ({math.degrees(safe_wz):.1f}°)!")
                else:
                    # Đi thẳng vào cản bên phải
                    if self.joy_override:
                        safe_vx = min(safe_vx, SAFETY_OVERRIDE_CRAWL_SPEED)
                        if now - self.last_override_warn_time > 1.5:
                            self.last_override_warn_time = now
                            log_msg(f"⚠️ [SAFETY OVERRIDE] Cản phía TRƯỚC (~{dist:.2f}m) -> ÉP TIẾN tốc độ rùa ({safe_vx:.2f} m/s)!")
                    else:
                        safe_vx = 0.0
                        if now - self.last_front_warn_time > 1.0:
                            self.last_front_warn_time = now
                            log_msg(f"🛑 [SAFETY GUARD] Có vật cản phía TRƯỚC-PHẢI (~{dist:.2f}m) -> Khóa ga tiến thẳng! Hãy bẻ lái sang TRÁI né cản hoặc lùi xe.")

            else:
                # Cản nằm ngay chính diện tim xe (-0.15 <= offset_y <= 0.15, hẹp)
                if abs(safe_wz) > 0.05:
                    # Bẻ lái sang bên nào cũng là né cản chính diện -> Cho phép bò vòng qua
                    safe_vx = min(safe_vx, SAFETY_EVASION_SPEED) if safe_vx > 0.0 else SAFETY_OVERRIDE_CRAWL_SPEED
                    if now - self.last_steer_warn_time > 1.5:
                        self.last_steer_warn_time = now
                        direction_str = "TRÁI" if safe_wz > 0 else "PHẢI"
                        log_msg(f"🔄 [STEERING EVASION] Bẻ lái né cản chính diện sang {direction_str} (~{dist:.2f}m) -> Cấp ga tiến ({safe_vx:.2f} m/s) cùng góc lái ({math.degrees(safe_wz):.1f}°)!")
                else:
                    # Đi thẳng vào cản
                    if self.joy_override:
                        safe_vx = min(safe_vx, SAFETY_OVERRIDE_CRAWL_SPEED)
                        if now - self.last_override_warn_time > 1.5:
                            self.last_override_warn_time = now
                            log_msg(f"⚠️ [SAFETY OVERRIDE] Cản phía TRƯỚC (~{dist:.2f}m) -> ÉP TIẾN tốc độ rùa ({safe_vx:.2f} m/s)!")
                    else:
                        safe_vx = 0.0
                        if now - self.last_front_warn_time > 1.0:
                            self.last_front_warn_time = now
                            log_msg(f"🛑 [SAFETY GUARD] Có vật cản chính diện phía TRƯỚC (~{dist:.2f}m) -> Khóa ga tiến thẳng! Hãy bẻ lái né cản hoặc lùi xe.")

        return safe_vx, safe_wz

    def set_target_velocity(self, linear_x: float, angular_z: float, linear_y: float = 0.0, override: bool = False):
        """Gán ngay lập tức vận tốc gần nhất và kích hoạt nhịp phát 50 Hz"""
        self.last_joy_time = time.time()
        self.cmd_linear_x = float(linear_x)
        self.cmd_angular_z = float(angular_z)
        self.cmd_linear_y = float(linear_y)
        self.joy_override = bool(override)
        self.joy_active = (abs(self.cmd_linear_x) > 0.001 or 
                           abs(self.cmd_angular_z) > 0.001 or 
                           abs(self.cmd_linear_y) > 0.001)

        # Lọc an toàn ngay tại lệnh ban đầu
        safe_vx, safe_wz = self.apply_safety_guard(self.cmd_linear_x, self.cmd_angular_z)

        # Phát ngay lập tức 1 lần để độ trễ bằng 0
        self.publish_cmd_vel(safe_vx, safe_wz, self.cmd_linear_y)

    def smooth_control_loop(self):
        """Vòng lặp chạy chuẩn 50 Hz: Lặp lại đúng giá trị gần nhất để động cơ không bị thiếu xung"""
        now = time.time()

        # 1. Watchdog an toàn: Quá 0.8s không nhận được tín hiệu joystick từ MQTT -> Tự động dừng xe
        if self.joy_active and (now - self.last_joy_time > 0.8):
            self.joy_active = False
            self.joy_override = False
            self.cmd_linear_x = 0.0
            self.cmd_angular_z = 0.0
            self.cmd_linear_y = 0.0
            log_msg("🛑 [Safety Watchdog] Mất tín hiệu Joystick (>0.8s) -> Dừng xe (cmd = 0)")

        # 1.1. Watchdog quyền lái: Quá hạn lease -> Tự động giải phóng quyền lái để người khác lái
        if self.active_driver_id and (now > self.driver_lock_expire):
            log_msg(f"🔓 [Driver Watchdog] Hết hạn phiên lái của '{self.active_driver_id}' (> {self.DRIVER_LEASE_SEC}s) -> Robot [{ROBOT_ID}] trở về trạng thái rảnh.")
            self.active_driver_id = None
            self.driver_lock_expire = 0.0

        # 2. Áp dụng lớp bảo vệ tránh va chạm (CHỈ gọi khi người lái đang gạt cần điều khiển)
        if self.joy_active and (abs(self.cmd_linear_x) > 0.001 or abs(self.cmd_angular_z) > 0.001):
            safe_vx, safe_wz = self.apply_safety_guard(self.cmd_linear_x, self.cmd_angular_z)
        else:
            safe_vx, safe_wz = 0.0, 0.0

        # 3. Phát liên tục giá trị đã qua kiểm tra an toàn xuống topic /cmd_vel_teleop ở tần số 50 Hz
        is_moving = (abs(safe_vx) > 0.001 or 
                     abs(safe_wz) > 0.001 or 
                     abs(self.cmd_linear_y) > 0.001)

        if is_moving or self.joy_active:
            self.publish_cmd_vel(safe_vx, safe_wz, self.cmd_linear_y)
            self.zero_cmd_sent_count = 0
        elif self.zero_cmd_sent_count < 10:
            # Khi xe đã dừng hẳn về 0: gửi thêm 10 lần 0.0 liên tiếp để đảm bảo driver nhận lệnh khóa dừng
            self.publish_cmd_vel(0.0, 0.0, 0.0)
            self.zero_cmd_sent_count += 1

    def publish_cmd_vel(self, linear_x: float = 0.0, angular_z: float = 0.0, linear_y: float = 0.0):
        """Gửi lệnh vận tốc xuống topic /cmd_vel"""
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)

    def publish_camera_control(self, action: str, cameras: str = "all"):
        """Gửi lệnh điều khiển bật/tắt camera sang node ros2_to_rtsp"""
        action = str(action).lower().strip()
        if action in ["on", "enable", "open", "all", "true"]:
            msg = String()
            msg.data = cameras if cameras and str(cameras).lower() not in ["none", "off"] else "all"
            self.cam_select_pub.publish(msg)
            log_msg(f"📷 [CAMERA STREAM] BẬT Camera: '{msg.data}' -> /multi_camera_rtsp_streamer/select_cameras")
        elif action in ["off", "disable", "close", "none", "false"]:
            msg = String()
            msg.data = "none"
            self.cam_select_pub.publish(msg)
            log_msg("📷 [CAMERA STREAM] TẮT tất cả Camera -> /multi_camera_rtsp_streamer/select_cameras")
        elif action in ["keepalive", "reset", "reset_timeout"]:
            msg = Empty()
            self.cam_keepalive_pub.publish(msg)
            log_msg("📷 [CAMERA STREAM] Gửi Keepalive gia hạn 5 phút -> /multi_camera_rtsp_streamer/keepalive")
        else:
            msg = String()
            msg.data = cameras if cameras else action
            self.cam_select_pub.publish(msg)
            log_msg(f"📷 [CAMERA STREAM] Chuyển kênh Camera: '{msg.data}' -> /multi_camera_rtsp_streamer/select_cameras")

    def trigger_joy_keepalive(self, linear_x: float = 0.0, angular_z: float = 0.0):
        """Khi xe đang có vận tốc di chuyển thực tế, tự động gửi keepalive sang ros2_to_rtsp để gia hạn 5 phút cho camera"""
        # CHỈ GIA HẠN KHI XE ĐANG CÓ VẬN TỐC DI CHUYỂN THỰC TẾ, KHÔNG GIA HẠN KHI DỪNG XE (0, 0)
        if abs(linear_x) <= 0.01 and abs(angular_z) <= 0.01:
            return

        now = time.time()
        # Throttling: phát keepalive tối đa mỗi 10 giây 1 lần khi đang lái để tránh spam mạng ROS 2
        if now - self.last_cam_keepalive_time >= 10.0:
            self.last_cam_keepalive_time = now
            msg = Empty()
            self.cam_keepalive_pub.publish(msg)
            log_msg("🔄 [Joystick -> Camera] Tự động gửi Keepalive gia hạn 5 phút sang ros2_to_rtsp")


mqtt_connected = False
last_connect_time = 0

def main(args=None):
    rclpy.init(args=args)
    node = MQTTTeleopCameraBridge()
    log_msg(f"🤖 Đang khởi chạy Teleop Bridge cho Robot ID: [{ROBOT_ID}] (Lắng nghe Topic: {TOPIC})")

    global mqtt_connected, last_connect_time
    mqtt_lock = threading.Lock()
    client_holder = [None]

    def handle_command(data):
        try:
            if isinstance(data, str):
                data = json.loads(data)

            # Hỗ trợ tự động giải nén nếu payload bọc trong trường 'Message' hoặc 'message'
            if isinstance(data, dict):
                if isinstance(data.get("Message"), dict):
                    data = data.get("Message")
                elif isinstance(data.get("message"), dict):
                    data = data.get("message")

            frame = data.get("frame_id") if isinstance(data, dict) else None
            log_msg(f"Received MQTT command: {data}")

            # 1. LỆNH ĐIỀU KHIỂN CAMERA STREAM (Gửi sang node ros2_to_rtsp)
            if frame in ["control_camera", "camera_stream", "stream_camera", "camera_control"]:
                action = data.get("action", "")
                state = data.get("state", None)
                cameras = data.get("cameras", data.get("mode", "all"))

                if state is not None:
                    if isinstance(state, bool):
                        action = "on" if state else "off"
                    elif isinstance(state, str):
                        action = state
                    elif isinstance(state, dict):
                        action = state.get("action", "on")
                        cameras = state.get("cameras", state.get("mode", cameras))

                if not action:
                    action = "on" if str(cameras).lower() not in ["none", "off", ""] else "off"

                log_msg(f"📹 Nhận lệnh Camera: action='{action}', cameras='{cameras}'")
                node.publish_camera_control(action=action, cameras=cameras)
                return

            # 2. LỆNH DI CHUYỂN TỪ JOYSTICK (KÈM TỰ ĐỘNG GỬI KEEPALIVE GIA HẠN 5 PHÚT CAMERA)
            elif frame in ["joystick", "joy", "cmd_vel", "teleop", "manual_control"]:
                linear_x = 0.0
                angular_z = 0.0
                linear_y = 0.0

                # Dạng 1: linear_x, angular_z hoặc speed, steer
                if "linear_x" in data:
                    linear_x = safe_float(data.get("linear_x")) or 0.0
                elif "speed" in data:
                    linear_x = safe_float(data.get("speed")) or 0.0
                elif "x" in data and not isinstance(data.get("x"), dict):
                    linear_x = safe_float(data.get("x")) or 0.0

                if "angular_z" in data:
                    angular_z = safe_float(data.get("angular_z")) or 0.0
                elif "steer" in data:
                    angular_z = safe_float(data.get("steer")) or 0.0
                elif "yaw" in data:
                    angular_z = safe_float(data.get("yaw")) or 0.0
                elif "z" in data and not isinstance(data.get("z"), dict):
                    angular_z = safe_float(data.get("z")) or 0.0

                # Dạng 2: data = {"linear": {"x": 0.5, "y": 0.0}, "angular": {"z": -0.2}}
                if isinstance(data.get("linear"), dict):
                    linear_x = safe_float(data["linear"].get("x")) or linear_x
                    linear_y = safe_float(data["linear"].get("y")) or 0.0
                if isinstance(data.get("angular"), dict):
                    angular_z = safe_float(data["angular"].get("z")) or angular_z

                # Dạng 3: data = {"axes": [linear_x, angular_z]}
                if isinstance(data.get("axes"), list) and len(data["axes"]) >= 2:
                    linear_x = safe_float(data["axes"][0]) or linear_x
                    angular_z = safe_float(data["axes"][1]) or angular_z

                # Chuẩn hóa nếu Web gửi góc theo độ (VD: steer = 30 hay 40 độ -> đổi sang rad)
                if abs(angular_z) > 1.5:
                    angular_z = math.radians(angular_z)

                # Giới hạn góc bẻ lái Festo tối đa 40 độ (MAX_STEER_RAD ~ 0.6981 rad)
                angular_z = max(-MAX_STEER_RAD, min(angular_z, MAX_STEER_RAD))

                # Nhận cờ Override từ Web/App (hỗ trợ override / bypass_safety / force)
                override_val = data.get("override", data.get("bypass_safety", data.get("force", False)))
                override = False
                if isinstance(override_val, bool):
                    override = override_val
                elif isinstance(override_val, (int, float)):
                    override = bool(override_val)
                elif isinstance(override_val, str):
                    override = override_val.lower() in ["true", "1", "yes", "on"]

                # --- BỘ LỌC KHÓA LÁI XE ĐỘC QUYỀN (SINGLE-PILOT FILTER) ---
                is_stop_cmd = (abs(linear_x) <= 0.001 and abs(angular_z) <= 0.001 and abs(linear_y) <= 0.001)
                now = time.time()
                driver_id = str(data.get("driver_id", "")).strip()

                # NGUYÊN TẮC AN TOÀN: Lệnh Dừng Khẩn Cấp (v=0, w=0) LUÔN ĐƯỢC CHẤP NHẬN từ bất kỳ ai
                if not is_stop_cmd:
                    # 1. Trường hợp xe rảnh hoặc phiên cũ đã hết hạn
                    if node.active_driver_id is None or (now > node.driver_lock_expire):
                        if not driver_id or driver_id in ["anonymous", "unknown"]:
                            return
                        if node.active_driver_id != driver_id:
                            log_msg(f"👑 [DRIVER LOCK] Robot [{ROBOT_ID}] cấp quyền lái mới cho: '{driver_id}'")
                        node.active_driver_id = driver_id
                        node.driver_lock_expire = now + node.DRIVER_LEASE_SEC
                    # 2. Trường hợp đúng người đang giữ quyền lái -> Gia hạn lease
                    elif node.active_driver_id == driver_id:
                        node.driver_lock_expire = now + node.DRIVER_LEASE_SEC
                    # 3. Trường hợp người khác cố tình gửi lệnh can thiệp trong khi xe đang có người lái -> Gạt bỏ!
                    else:
                        if now - node.last_driver_reject_time > 1.5:
                            node.last_driver_reject_time = now
                            log_msg(f"⛔ [DRIVER LOCK] Từ chối lệnh từ '{driver_id}'. Robot [{ROBOT_ID}] đang được lái bởi '{node.active_driver_id}'")
                        return

                # Cập nhật vận tốc mục tiêu và cờ override vào bộ làm mượt 50 Hz
                node.set_target_velocity(linear_x=linear_x, angular_z=angular_z, linear_y=linear_y, override=override)

                # TỰ ĐỘNG GỬI KEEPALIVE SANG ROS2_TO_RTSP ĐỂ GIA HẠN 5 PHÚT CAMERA (CHỈ KHI XE ĐANG DI CHUYỂN THỰC TẾ)
                node.trigger_joy_keepalive(linear_x=linear_x, angular_z=angular_z)
                return

            # 3. LỆNH NHƯỜNG QUYỀN LÁI XE CHỦ ĐỘNG TỪ WEB / APP
            elif frame in ["release_driver", "driver_release", "unlock_teleop"]:
                rel_id = str(data.get("driver_id", "")).strip()
                if node.active_driver_id and rel_id and node.active_driver_id == rel_id:
                    log_msg(f"🔓 [DRIVER LOCK] Robot [{ROBOT_ID}] đã được nhường quyền bởi '{node.active_driver_id}'")
                    node.active_driver_id = None
                    node.driver_lock_expire = 0.0
                    node.set_target_velocity(0.0, 0.0, 0.0)
                return

            # 4. LỆNH CẤU HÌNH BẬT/TẮT SAFETY OVERRIDE ĐỘC LẬP
            elif frame in ["override", "safety_override", "bypass_safety"]:
                state = data.get("state", data.get("override", data.get("enabled", True)))
                if isinstance(state, str):
                    state = state.lower() in ["true", "1", "on", "yes"]
                node.joy_override = bool(state)
                log_msg(f"🛡️ [SAFETY] Cập nhật chế độ Safety Override: {node.joy_override}")
                return

            else:
                log_msg(f"Bỏ qua lệnh không liên quan: frame_id='{frame}'")

        except Exception as e:
            log_msg(f"Lỗi xử lý lệnh MQTT: {e}")

    # Callback tin nhắn từ MQTT broker
    def on_message(client, userdata, msg):
        payload = msg.payload.decode("utf-8")
        handle_command(payload)

    def create_mqtt_client():
        global mqtt_connected, last_connect_time
        client_id = f"{CLIENT_ID}-{int(time.time())}"
        c = mqtt.Client(client_id=client_id)
        c.username_pw_set(USERNAME, PASSWORD)
        c.on_message = on_message

        def on_connect(client, userdata, flags, rc):
            global mqtt_connected, last_connect_time
            if rc == 0:
                mqtt_connected = True
                last_connect_time = time.time()
                log_msg("✅ Đã kết nối MQTT Broker thành công!")
                client.subscribe(TOPIC)
                log_msg(f"📡 Đã subscribe topic: {TOPIC}")
            else:
                mqtt_connected = False
                log_msg(f"❌ Kết nối MQTT Broker thất bại, mã lỗi: {rc}")

        def on_disconnect(client, userdata, rc):
            global mqtt_connected
            mqtt_connected = False
            if rclpy.ok() and client_holder[0] is not None:
                log_msg(f"⚠️ Mất kết nối MQTT (rc={rc}) -> đang thử kết nối lại...")

        c.on_connect = on_connect
        c.on_disconnect = on_disconnect
        c.reconnect_delay_set(min_delay=1, max_delay=30)
        return c

    def start_mqtt_client():
        global last_connect_time
        last_connect_time = time.time()
        c = create_mqtt_client()
        try:
            c.connect_async(BROKER, PORT, keepalive=30)
            c.loop_start()
            with mqtt_lock:
                client_holder[0] = c
            log_msg("MQTT client đã khởi chạy.")
        except Exception as e:
            log_msg(f"Lỗi khởi chạy MQTT: {e}")

    def mqtt_watchdog_loop():
        global mqtt_connected, last_connect_time
        while rclpy.ok():
            time.sleep(10)
            if not mqtt_connected:
                elapsed = time.time() - last_connect_time
                if elapsed > MQTT_RECONNECT_WATCHDOG_SEC:
                    log_msg(f"Watchdog: Mất kết nối MQTT {elapsed:.0f}s -> khởi động lại client")
                    with mqtt_lock:
                        old_client = client_holder[0]
                        client_holder[0] = None
                    if old_client is not None:
                        try:
                            old_client.loop_stop()
                            old_client.disconnect()
                        except Exception:
                            pass
                    start_mqtt_client()

    # Khởi chạy MQTT và Watchdog thread
    start_mqtt_client()
    threading.Thread(target=mqtt_watchdog_loop, daemon=True).start()

    try:
        from rclpy.executors import ExternalShutdownException
    except ImportError:
        ExternalShutdownException = ()

    # Chạy ROS 2 Node Spin
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with mqtt_lock:
            if client_holder[0] is not None:
                try:
                    client_holder[0].loop_stop()
                    client_holder[0].disconnect()
                except Exception:
                    pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

if __name__ == "__main__":
    main()
