#!/usr/bin/env python3
"""
HƯỚNG DẪN KIỂM TRA NHANH QUA TERMINAL:
--------------------------------------------------------------------------------
1. Bật/Tắt 1 camera cụ thể (Service):
   ros2 service call /multi_camera_rtsp_streamer/camera/front/enable std_srvs/srv/SetBool "{data: true}"
   ros2 service call /multi_camera_rtsp_streamer/camera/rear/enable std_srvs/srv/SetBool "{data: true}"
   ros2 service call /multi_camera_rtsp_streamer/camera/birdview/enable std_srvs/srv/SetBool "{data: true}"

2. Chọn tổ hợp camera tùy ý (1, 2, 3 camera hoặc tất cả) qua Topic:
   # Chỉ bật 1 camera (các cam khác tự tắt):
   ros2 topic pub --once /multi_camera_rtsp_streamer/select_cameras std_msgs/msg/String "{data: 'front'}"
   
   # Bật camera sau:
   ros2 topic pub --once /multi_camera_rtsp_streamer/select_cameras std_msgs/msg/String "{data: 'rear'}"
   
   # Bật 3 camera (front, left, right):
   ros2 topic pub --once /multi_camera_rtsp_streamer/select_cameras std_msgs/msg/String "{data: 'front,left,right'}"
   
   # Bật tất cả 4 camera:
   ros2 topic pub --once /multi_camera_rtsp_streamer/select_cameras std_msgs/msg/String "{data: 'all'}"
   
   # Tắt tất cả:
   ros2 topic pub --once /multi_camera_rtsp_streamer/select_cameras std_msgs/msg/String "{data: 'none'}"

3. Bật/Tắt TẤT CẢ cùng lúc qua Service:
   ros2 service call /multi_camera_rtsp_streamer/enable_all std_srvs/srv/SetBool "{data: true}"
   ros2 service call /multi_camera_rtsp_streamer/enable_all std_srvs/srv/SetBool "{data: false}"

4. Xe tự động kích hoạt khi có sự cố (E-Stop, lỗi an toàn):
   ros2 topic pub --once /multi_camera_rtsp_streamer/stream_trigger std_msgs/msg/Bool "{data: true}"

5. RESET / GIA HẠN THỜI GIAN 5 PHÚT (Reset countdown về ban đầu):
   # Cách 1: Qua Service (Trả về thông báo xác nhận và danh sách camera được gia hạn)
   ros2 service call /multi_camera_rtsp_streamer/reset_timeout std_srvs/srv/Trigger "{}"

   # Cách 2: Qua Topic Heartbeat / Keepalive (Cho script/UI gửi định kỳ để duy trì stream)
   ros2 topic pub --once /multi_camera_rtsp_streamer/keepalive std_msgs/msg/Empty "{}"
--------------------------------------------------------------------------------
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, String, Empty
from std_srvs.srv import SetBool, Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess
import time
import json
import os
import sys
from urllib.parse import quote
import signal

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_config.json")

# Hàm kiểm tra và ngắt tiến trình trước đó, đảm bảo chỉ 1 tiến trình nhận lệnh đẩy dư liệu luồng camera
def kill_duplicate_instances():
    """Tự động kiểm tra và ngắt các tiến trình ros2_to_rtsp.py cũ và gst-launch tồn đọng"""
    current_pid = os.getpid()
    try:
        # Tìm tất cả PID đang chạy ros2_to_rtsp.py
        output = subprocess.check_output(["pgrep", "-f", "ros2_to_rtsp.py"]).decode().strip()
        pids = [int(p) for p in output.split() if p.isdigit() and int(p) != current_pid]
        
        if pids:
            for old_pid in pids:
                print(f"⚠️ [SINGLETON] Phát hiện tiến trình ros2_to_rtsp cũ (PID: {old_pid}). Đang ngắt tiến trình cũ...", flush=True)
                try:
                    os.kill(old_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            
            # Dọn dẹp sạch các tiến trình GStreamer mồ côi của tiến trình cũ
            subprocess.run(["killall", "-9", "gst-launch-1.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
            print("✅ [SINGLETON] Đã ngắt tiến trình cũ, tiến trình mới bắt đầu hoạt động.", flush=True)
    except subprocess.CalledProcessError:
        pass  # Không có tiến trình nào khác đang chạy
    except Exception as e:
        print(f"⚠️ [SINGLETON] Cảnh báo kiểm tra tiến trình: {e}", flush=True)


def load_robot_config():
    """Đọc cấu hình định danh xe và thông số RTSP từ robot_config.json. Thoát app ngay nếu lỗi!"""
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

        if "rtsp" not in cfg or not isinstance(cfg.get("rtsp"), dict):
            raise ValueError(f"Thiếu hoặc sai định dạng khối cấu hình 'rtsp' trong file {CONFIG_FILE}.")

        return cfg
    except Exception as e:
        print(f"\n{'='*70}", flush=True)
        print(f"❌ [LỖI NGHIÊM TRỌNG] Lỗi phân tích file cấu hình '{CONFIG_FILE}': {e}", flush=True)
        print(f"🛑 ĐANG TẮT ỨNG DỤNG THEO YÊU CẦU AN TOÀN...\n{'='*70}\n", flush=True)
        sys.exit(1)

class CameraStreamChannel:
    """Quản lý pipeline GStreamer và vòng đời stream cho từng Camera riêng biệt (H.265 Pass-through <1% CPU)"""
    def __init__(self, node: Node, cam_id: str, topic_name: str, stream_path: str, server_ip: str,
                 user_pub: str = "", pass_pub: str = "",
                 width=640, height=480, fps=15, bitrate=180000, timeout_sec=300, flip_method=0, 
                 codec="h265", rtsp_port="8554"):
        self.node = node
        self.cam_id = cam_id
        self.topic_name = topic_name
        self.stream_path = stream_path
        self.server_ip = server_ip
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.timeout_sec = timeout_sec
        self.flip_method = flip_method
        self.codec = codec
        
        self.user_pub = user_pub
        self.pass_pub = pass_pub
        self.rtsp_port = rtsp_port or "8554"
        
        self.is_streaming = False
        self.stream_start_time = 0.0
        self.warned_30s = False
        self.process = None
        self.last_retry_time = 0.0
        
        # 1. Subscriber Camera: Chỉ đăng ký khi thực sự bật stream để camera_stream_node đóng van nén khi rảnh (0% CPU)
        self.sub = None
        
        # 2. Service riêng cho từng camera
        self.srv = self.node.create_service(
            SetBool,
            f'~/camera/{self.cam_id}/enable',
            self.handle_service
        )

    def start_streaming(self, trigger_source="Manual"):
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_start_time = time.time()
            self.warned_30s = False
            rot_info = " (Xoay 180 độ HW tại node gốc)" if self.flip_method == 2 else ""
            self.node.get_logger().warn(f"[{self.cam_id}] BẬT stream bởi: {trigger_source}{rot_info} (Timeout: {self.timeout_sec}s)")

            # Khởi chạy pipeline GStreamer pass-through H.265
            self.init_gst_process()

            # Subscribe vào topic compressed (H.265 HW NVENC) để nhận gói tin ~1.5KB
            if self.sub is None:
                self.sub = self.node.create_subscription(
                    CompressedImage,
                    self.topic_name,
                    self.image_callback,
                    rclpy.qos.qos_profile_sensor_data
                )
        else:
            # Nếu đang chạy mà nhận lệnh -> Tự động reset bộ đếm thời gian về lại ban đầu
            self.reset_timeout(reason=f"Nhận lệnh kích hoạt lại ({trigger_source})")

    def reset_timeout(self, reason="Manual"):
        """Reset bộ đếm thời gian về lại ban đầu (5 phút)"""
        if self.is_streaming:
            self.stream_start_time = time.time()
            self.warned_30s = False
            self.node.get_logger().info(f"🔄 [{self.cam_id}] Đã reset bộ đếm về {self.timeout_sec}s! (Lý do: {reason})")

    def stop_streaming(self, reason="Manual"):
        if self.is_streaming:
            self.is_streaming = False
            self.node.get_logger().warn(f"[{self.cam_id}] TẮT stream! Lý do: {reason}")
            self.stop_gst_process()

            # Hủy đăng ký nhận dữ liệu để camera_stream_node nhận biết 0 subscriber và đóng van nén (0% CPU)
            if self.sub is not None:
                self.node.destroy_subscription(self.sub)
                self.sub = None

    def stop_gst_process(self):
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None
                self.node.get_logger().info(f"[{self.cam_id}] Đã ngắt tiến trình RTSP.")

    def handle_service(self, request, response):
        if request.data:
            self.start_streaming(trigger_source=f"Service {self.cam_id}")
            response.success = True
            response.message = f"[{self.cam_id}] Đã BẬT stream: rtsp://{self.server_ip}:{self.rtsp_port}/{self.stream_path} (Timeout: {self.timeout_sec}s)"
        else:
            self.stop_streaming(reason=f"Service {self.cam_id}")
            response.success = True
            response.message = f"[{self.cam_id}] Đã TẮT stream."
        return response

    def check_watchdog(self):
        if self.is_streaming and self.timeout_sec > 0:
            elapsed = time.time() - self.stream_start_time
            remaining = self.timeout_sec - elapsed

            # Cảnh báo khi còn dưới 30 giây
            if remaining <= 30 and not self.warned_30s:
                self.warned_30s = True
                self.node.get_logger().warn(f"[{self.cam_id}] ⚠️ CẢNH BÁO: Sắp hết hạn stream! Còn {int(remaining)}s trước khi tự ngắt.")

            if elapsed > self.timeout_sec:
                self.stop_streaming(reason=f"Hết hạn watchdog ({self.timeout_sec}s)")

    def init_gst_process(self):
        # Thiết lập parser và caps tương ứng với chuẩn nén HW
        if self.codec == "h265":
            parse_plugin = 'h265parse'
            media_caps = 'video/x-h265,stream-format=byte-stream'
        else:
            parse_plugin = 'h264parse'
            media_caps = 'video/x-h264,stream-format=byte-stream'

        # Xử lý xác thực (Authentication) an toàn với URL-Encoding
        if self.user_pub and self.pass_pub:
            enc_user = quote(str(self.user_pub), safe='')
            enc_pass = quote(str(self.pass_pub), safe='')
            auth_str = f"{enc_user}:{enc_pass}@"
        elif self.user_pub:
            enc_user = quote(str(self.user_pub), safe='')
            auth_str = f"{enc_user}@"
        else:
            auth_str = ""

        port_str = f":{self.rtsp_port}" if self.rtsp_port else ""
        RTSP_URL = f"rtsp://{auth_str}{self.server_ip}{port_str}/{self.stream_path}"
        CLEAN_URL = f"rtsp://{self.server_ip}{port_str}/{self.stream_path}"

        # GStreamer Pass-Through: Nhận thẳng NAL units từ fdsrc, parse SPS/PPS và đẩy sang MediaMTX
        # Tiêu thụ < 1% CPU vì KHÔNG giải mã và KHÔNG mã hóa lại!
        gst_cmd = [
            'gst-launch-1.0', '-q',
            'fdsrc', 'do-timestamp=true', '!',
            media_caps, '!',
            parse_plugin, 'config-interval=-1', '!',
            'rtspclientsink', f'location={RTSP_URL}', 'protocols=tcp'
        ]

        try:
            self.process = subprocess.Popen(gst_cmd, stdin=subprocess.PIPE)
            # Mở rộng pipe buffer 2MB
            try:
                import fcntl
                fcntl.fcntl(self.process.stdin.fileno(), 1031, 2 * 1024 * 1024)
            except Exception:
                pass
            rot_msg = " [Xoay 180 độ HW tại node gốc]" if self.flip_method == 2 else ""
            self.node.get_logger().info(f"[{self.cam_id}] ⚡ KẾT NỐI RTSP HW PASS-THROUGH (<1% CPU): {CLEAN_URL}{rot_msg}")
        except Exception as e:
            self.node.get_logger().error(f"[{self.cam_id}] Lỗi khởi chạy GStreamer: {e}")
            self.process = None

    def image_callback(self, msg: CompressedImage):
        # 0% CPU overhead khi kênh này không bật
        if not self.is_streaming:
            return

        now = time.time()
        if self.process is None:
            if now - self.last_retry_time < 2.0:
                return
            self.last_retry_time = now
            self.init_gst_process()
            if self.process is None:
                return

        # ZERO-CPU PASS-THROUGH:
        # Gói tin H.265 chỉ ~1.5 KB. Bơm thẳng memoryview(msg.data) vào GStreamer pipe!
        try:
            self.process.stdin.write(memoryview(msg.data))
            self.process.stdin.flush()
        except (BrokenPipeError, IOError):
            returncode = self.process.poll() if self.process else None
            self.node.get_logger().error(f"[{self.cam_id}] Mất kết nối RTSP (Broken Pipe, exitcode: {returncode}). Sẽ thử lại sau...")
            self.stop_gst_process()


class MultiCameraRTSPStreamerNode(Node):
    def __init__(self):
        super().__init__('multi_camera_rtsp_streamer')
        
        # --- ĐỌC CẤU HÌNH TỪ robot_config.json ---
        cfg = load_robot_config()
        self.robot_id = cfg.get("robot_id", "RBT0003")
        rtsp_cfg = cfg.get("rtsp", {})

        self.SERVER_IP = rtsp_cfg.get("server_ip", "192.168.14.64")
        self.AUTO_TIMEOUT_SEC = rtsp_cfg.get("auto_timeout_sec", 600)
        self.TARGET_WIDTH = rtsp_cfg.get("target_width", 640)
        self.TARGET_HEIGHT = rtsp_cfg.get("target_height", 480)
        self.FPS = rtsp_cfg.get("fps", 15)
        self.BITRATE = rtsp_cfg.get("bitrate", 180000)
        self.CODEC = rtsp_cfg.get("codec", "h265")
        self.USER_PUB = str(rtsp_cfg.get("user_pub", "") or "")
        self.PASS_PUB = str(rtsp_cfg.get("pass_pub", "") or "")
        self.SERVER_PORT = rtsp_cfg.get("rtsp_port", 8554)

        # --- DANH SÁCH 4 CAMERA (Tự động gắn mã robot_id vào đường dẫn stream) ---
        # Đọc từ topic Compressed (đã được nén H.265 phần cứng NVENC 0% CPU bởi camera_stream_node)
        # Giữ camera_stream_node là tiến trình duy nhất mở V4L2 và public đồng thời /camera/<name>/image_raw cho các tác vụ khác!
        self.CAMERAS_DEF = {
            "front": {
                "topic": "/camera/front/image_raw/compressed",
                "path": f"{self.robot_id}/front",
                "flip_method": 0
            },
            "rear": {
                "topic": "/camera/rear/image_raw/compressed",
                "path": f"{self.robot_id}/rear",
                "flip_method": 0
            },
            "left": {
                "topic": "/camera/left/image_raw/compressed",
                "path": f"{self.robot_id}/left",
                "flip_method": 0
            },
            "right": {
                "topic": "/camera/right/image_raw/compressed",
                "path": f"{self.robot_id}/right",
                "flip_method": 0
            },
            "birdview": {
                "topic": "/camera/birdview/raw/compress",
                "path": f"{self.robot_id}/birdview",
                "flip_method": 0
            }
        }
        
        # Khởi tạo các kênh camera độc lập
        self.channels = {}
        for cam_id, cfg in self.CAMERAS_DEF.items():
            self.channels[cam_id] = CameraStreamChannel(
                node=self,
                cam_id=cam_id,
                topic_name=cfg["topic"],
                stream_path=cfg["path"],
                server_ip=self.SERVER_IP,
                width=self.TARGET_WIDTH,
                height=self.TARGET_HEIGHT,
                fps=self.FPS,
                bitrate=self.BITRATE,
                timeout_sec=self.AUTO_TIMEOUT_SEC,
                flip_method=cfg.get("flip_method", 0), 
                codec=self.CODEC,
                user_pub=self.USER_PUB,
                pass_pub=self.PASS_PUB,
                rtsp_port=self.SERVER_PORT
            )
            
        # 1. Master Service: Bật/tắt TẤT CẢ camera cùng lúc
        self.srv_all = self.create_service(
            SetBool,
            '~/enable_all',
            self.handle_enable_all
        )
        
        # 2. Topic chọn danh sách camera linh hoạt (Ví dụ: "front_left,front_right", "all", "none")
        self.sub_select = self.create_subscription(
            String,
            '~/select_cameras',
            self.handle_select_cameras_topic,
            10
        )
        
        # 3. Topic Báo lỗi tổng từ xe (Kích hoạt 2 cam phía trước khi lỗi)
        self.sub_trigger = self.create_subscription(
            Bool,
            '~/stream_trigger',
            self.handle_emergency_trigger,
            10
        )
        
        # 4. Service Reset Timeout: Reset lại 5 phút cho tất cả camera đang chạy
        self.srv_reset = self.create_service(
            Trigger,
            '~/reset_timeout',
            self.handle_reset_timeout_service
        )

        # 5. Topic Keepalive / Heartbeat: Nhận xung để reset lại 5 phút
        self.sub_keepalive = self.create_subscription(
            Empty,
            '~/keepalive',
            self.handle_keepalive_topic,
            10
        )

        # 6. Timer watchdog chung (Kiểm tra timeout mỗi 1 giây)
        self.timer = self.create_timer(1.0, self.watchdog_check)
        
        self.get_logger().info(f"=== MULTI-CAMERA RTSP STREAMER [{self.robot_id}] SẴN SÀNG (4 Cameras) ===")
        self.get_logger().info(f"Server MediaMTX: rtsp://{self.SERVER_IP}:{self.SERVER_PORT}/{self.robot_id}/<camera_id>")
        self.get_logger().info(f"Các kênh camera: {list(self.CAMERAS_DEF.keys())}")
        self.get_logger().info(f"Cơ chế Watchdog: {self.AUTO_TIMEOUT_SEC}s (Có hỗ trợ reset via ~/reset_timeout & ~/keepalive)")

    def reset_all_active_timeouts(self, source="Keepalive"):
        """Gia hạn/reset lại 5 phút cho tất cả các camera đang hoạt động"""
        active_cams = [cam_id for cam_id, ch in self.channels.items() if ch.is_streaming]
        if not active_cams:
            msg = "Hiện không có camera nào đang stream để gia hạn."
            self.get_logger().warn(f"[{source}] {msg}")
            return False, msg

        for cam_id in active_cams:
            self.channels[cam_id].reset_timeout(reason=source)

        msg = f"Đã reset bộ đếm về {self.AUTO_TIMEOUT_SEC}s ({(self.AUTO_TIMEOUT_SEC/60)} phút) cho các camera: {', '.join(active_cams)}"
        self.get_logger().info(f"🔄 [{source}] {msg}")
        return True, msg

    def handle_reset_timeout_service(self, request, response):
        """Service xử lý lệnh reset timeout từ người vận hành"""
        success, msg = self.reset_all_active_timeouts(source="Service ~/reset_timeout")
        response.success = success
        response.message = msg
        return response

    def handle_keepalive_topic(self, msg: Empty):
        """Topic nhận tín hiệu keepalive/heartbeat từ hệ thống điều khiển"""
        self.reset_all_active_timeouts(source="Topic ~/keepalive")

    def handle_enable_all(self, request, response):
        if request.data:
            for ch in self.channels.values():
                ch.start_streaming(trigger_source="Master Service (All)")
            response.success = True
            response.message = f"Đã BẬT tất cả {len(self.channels)} camera (Timeout: {self.AUTO_TIMEOUT_SEC}s)."
        else:
            for ch in self.channels.values():
                ch.stop_streaming(reason="Master Service (All)")
            response.success = True
            response.message = "Đã TẮT tất cả camera."
        return response

    def handle_select_cameras_topic(self, msg: String):
        """
        Cho phép chọn bất kỳ tổ hợp nào qua text:
        - "all": Bật tất cả 4 camera
        - "none" / "off": Tắt tất cả
        - "front": Chỉ bật 1 camera này, tắt các camera khác
        - "rear": Bật camera sau
        - "front,left,right": Bật 3 camera
        """
        raw_cmd = msg.data.strip().lower()
        if raw_cmd in ["all", "*"]:
            for ch in self.channels.values():
                ch.start_streaming(trigger_source="Select Topic (All)")
            return

        if raw_cmd in ["none", "off", ""]:
            for ch in self.channels.values():
                ch.stop_streaming(reason="Select Topic (None)")
            return

        selected_ids = [s.strip() for s in raw_cmd.split(",") if s.strip()]
        for cam_id, ch in self.channels.items():
            if cam_id in selected_ids:
                # Tự động bật hoặc reset lại 5 phút nếu đang chạy
                ch.start_streaming(trigger_source="Select Topic")
            else:
                ch.stop_streaming(reason="Not in selected list")

    def handle_emergency_trigger(self, msg: Bool):
        """Khi xe gặp lỗi, tự động bật camera trước để can thiệp"""
        if msg.data:
            for cam_id in ["front"]:
                if cam_id in self.channels:
                    ch = self.channels[cam_id]
                    ch.start_streaming(trigger_source="Emergency Fault Trigger")
        else:
            for ch in self.channels.values():
                ch.stop_streaming(reason="Emergency Fault Cleared")

    def watchdog_check(self):
        for ch in self.channels.values():
            ch.check_watchdog()

    def destroy_node(self):
        for ch in self.channels.values():
            ch.stop_gst_process()
        super().destroy_node()

def main(args=None):
    
    # Tự động ngắt bản cũ nếu bị chạy trùng
    # kill_duplicate_instances()
    
    rclpy.init(args=args)    
    node = MultiCameraRTSPStreamerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
