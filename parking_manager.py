import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog
from collections import deque
from src.logic import ViolationLogic, MOVING, STOPPED, PARKED
from src.telegram_bot import send_telegram_image, send_telegram_video
from src.utils import ensure_dir, now_ts

class ParkingManager:
    def __init__(self, root, app_instance):
        self.root = root
        self.app = app_instance
        self.no_park_polygon = None
        
        # --- ILLEGAL PARKING CONFIGS ---
        self.stop_seconds = 30
        self.move_thr_px = 10.0
        self.cooldown_seconds = 30.0
        self.telegram_enabled = False
        self.save_violation_frames = True
        self.telegram_bot_token = ""
        self.telegram_chat_id = ""

        self.logic = None
        self.frame_buffer = None
        self.fps = 30.0

    def init_ui(self):
        self.frame_no_park = tk.LabelFrame(self.root, text="3. Quản lý Vùng Cấm Đỗ", font=("Arial", 11, "bold"))
        self.frame_no_park.pack(fill="x", padx=10, pady=5)

        self.btn_load_no_park = tk.Button(self.frame_no_park, text="Load Vùng Cấm", command=self.load_no_park, width=14, state=tk.NORMAL, font=("Arial", 10))
        self.btn_load_no_park.grid(row=0, column=0, padx=5, pady=5)

        self.btn_clear_no_park = tk.Button(self.frame_no_park, text="Hủy Vùng Cấm", command=self.clear_no_park, width=12, state=tk.NORMAL, font=("Arial", 10))
        self.btn_clear_no_park.grid(row=0, column=1, padx=5, pady=5)

        self.btn_draw_no_park = tk.Button(self.frame_no_park, text="Vẽ Vùng Cấm", command=self.open_draw_no_park, width=14, state=tk.DISABLED, font=("Arial", 10))
        self.btn_draw_no_park.grid(row=0, column=2, padx=5, pady=5)

        self.lbl_no_park_status = tk.Label(self.frame_no_park, text="Vùng cấm: Chưa có", fg="red", font=("Arial", 10, "italic"))
        self.lbl_no_park_status.grid(row=1, column=0, columnspan=3, pady=2)

    def load_no_park(self):
        path = filedialog.askopenfilename(title="Chọn File Vùng Cấm", filetypes=[("JSON Files", "*.json")])
        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.no_park_polygon = np.array(json.load(f).get("points"))
                self.lbl_no_park_status.config(text=f"Vùng cấm: Đã load {os.path.basename(path)}", fg="green")

    def clear_no_park(self):
        self.no_park_polygon = None
        self.lbl_no_park_status.config(text="Vùng cấm: Chưa có", fg="red")

    def open_draw_no_park(self):
        if not self.app.video_path: return
        cap = cv2.VideoCapture(self.app.video_path)
        ret, first_frame = cap.read()
        cap.release()
        polygon = self.app.draw_polygon(first_frame, self.no_park_polygon, "Draw No Parking Zone", (0, 0, 255))
        if polygon is not None:
            self.no_park_polygon = polygon
            self.lbl_no_park_status.config(text="Vùng cấm: Đã vẽ tạm", fg="orange")
            
            # Thêm prompt lưu file
            from tkinter import messagebox
            if messagebox.askyesno("Lưu Vùng Cấm", "Bạn có muốn lưu vùng cấm đỗ này không?"):
                video_name = os.path.splitext(os.path.basename(self.app.video_path))[0]
                save_dir = "layouts"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{video_name}_parking_layout.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"points": self.no_park_polygon.tolist()}, f)
                self.lbl_no_park_status.config(text=f"Vùng cấm: Đã lưu {os.path.basename(save_path)}", fg="green")

    def enable_draw_btn(self):
        self.btn_draw_no_park.config(state=tk.NORMAL)

    def setup_detection(self, fps):
        self.fps = fps
        ensure_dir("outputs/violations")
        self.logic = ViolationLogic(self.stop_seconds, self.move_thr_px, self.cooldown_seconds, fps=fps)
        self.frame_buffer = deque(maxlen=int(5 * fps))

    def update_buffer(self, frame_copy):
        if self.frame_buffer is not None:
            self.frame_buffer.append(frame_copy)

    def save_violation_video(self, ts, track_id):
        if not self.frame_buffer: return ""
        video_dir = os.path.join("outputs", "violations")
        ensure_dir(video_dir)
        video_path = os.path.join(video_dir, f"violation_{track_id}_{ts}.mp4")
        first_frame = self.frame_buffer[0]
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
        for f in self.frame_buffer:
            out.write(f)
        out.release()
        return video_path

    def process_vehicle(self, frame, track_id, label, cx, cy, frame_count):
        """Kiểm tra và cập nhật trạng thái đỗ xe, trả về display_label và box_color mới (nếu có)"""
        if self.logic is None:
            return None, None

        in_no_park = False
        if self.no_park_polygon is not None:
            in_no_park = cv2.pointPolygonTest(self.no_park_polygon, (cx, cy), False) >= 0

        if in_no_park:
            still_time = self.logic.update(track_id, (cx, cy), frame_count)
            veh_state = self.logic.get_vehicle_state(track_id)
            
            if veh_state == PARKED:
                box_color = (0, 0, 255) # Đỏ
            elif veh_state == STOPPED:
                box_color = (0, 165, 255) # Cam
            else:
                box_color = None
                
            state_str = "DO XE" if veh_state == PARKED else ("DUNG" if veh_state == STOPPED else "CHAY")
            display_label = f"ID:{track_id} {label} {state_str} {still_time:.1f}s"
            
            if self.logic.should_flag_violation(track_id, frame_count, in_no_park=in_no_park):
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
                cv2.putText(frame, "VI PHAM: DO XE SAI QUY DINH!", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                
                ts = now_ts()
                if self.save_violation_frames:
                    img_path = os.path.join("outputs", "violations", f"violation_{track_id}_{ts}.jpg")
                    cv2.imwrite(img_path, frame)
                    if self.telegram_enabled:
                        caption = f"XE ĐỖ SAI QUY ĐỊNH\nID xe: {track_id}\nLoại xe: {label}\nThời gian đứng: {still_time:.1f}s"
                        send_telegram_image(img_path, caption, self.telegram_bot_token, self.telegram_chat_id)
                
                if self.frame_buffer:
                    video_path = self.save_violation_video(ts, track_id)
                    if self.telegram_enabled and video_path:
                        video_caption = f"VIDEO VI PHẠM ĐỖ XE\nID xe: {track_id}\nLoại xe: {label}\nThời gian đứng: {still_time:.1f}s"
                        send_telegram_video(video_path, video_caption, self.telegram_bot_token, self.telegram_chat_id)
            return display_label, box_color
        return None, None

    def draw_polygon_overlay(self, frame):
        """Vẽ vùng cấm đỗ màu đỏ lên frame"""
        if self.no_park_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.no_park_polygon], (0, 0, 180))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [self.no_park_polygon], True, (0, 0, 255), 2)
