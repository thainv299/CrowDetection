import cv2
import numpy as np
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import json
import time
import math
from collections import Counter
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import re          
from paddleocr import PaddleOCR
import queue
import ocr_processor

class_names = {
    0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 
    4: "License Plate", 5: "Bus", 6: "Truck"    
}

colors = {
    0: (0, 255, 0), 1: (255, 0, 0), 2: (255, 255, 0), 3: (0, 255, 255), 
    4: (0, 0, 255), 5: (0, 165, 255), 6: (255, 0, 255)
}

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection CV")
        self.root.geometry("500x500")
        
        self.video_path = None
        self.model_path = None  
        self.model = None
        self.ocr_reader = None
        self.roi_polygon = None

        self.OCR_INTERVAL    = 4       # Chạy OCR mỗi N frame
        self.VOTE_THRESHOLD  = 3       # Đọc giống nhau N lần thì chốt (Confirm)
        self.CONF_THRESHOLD  = 0.32    # Ngưỡng tin cậy của YOLO
        self.MAX_LOST_FRAMES = 5       # Số frame giữ hộp viền nếu YOLO không detect được

        # --- OCR ASYNC QUEUE ---
        self._ocr_queue = queue.Queue(maxsize=3)
        self._ocr_pending_results = {}
        self._ocr_worker_running = False

        # --- KHỞI TẠO GIAO DIỆN ---
        self.lbl_title = tk.Label(root, text="Phát hiện đông đúc / tắc nghẽn", font=("Arial", 14, "bold"))
        self.lbl_title.pack(pady=5)

        self.frame_top = tk.LabelFrame(root, text="1. Nguồn dữ liệu", font=("Arial", 11, "bold"))
        self.frame_top.pack(fill="x", padx=10, pady=5)

        self.btn_select_model = tk.Button(self.frame_top, text="Chọn Model YOLO", command=self.select_model, width=18, font=("Arial", 10))
        self.btn_select_model.grid(row=0, column=0, padx=5, pady=5)
        self.lbl_model_path = tk.Label(self.frame_top, text="Chưa chọn model", wraplength=250, fg="gray", font=("Arial", 10))
        self.lbl_model_path.grid(row=0, column=1, sticky="w", padx=5)

        self.btn_select = tk.Button(self.frame_top, text="Chọn Video", command=self.select_video, width=18, font=("Arial", 10))
        self.btn_select.grid(row=1, column=0, padx=5, pady=5)
        self.lbl_path = tk.Label(self.frame_top, text="Chưa chọn video", wraplength=250, fg="gray", font=("Arial", 10))
        self.lbl_path.grid(row=1, column=1, sticky="w", padx=5)

        self.frame_layout = tk.LabelFrame(root, text="2. Quản lý Vùng Giám Sát (ROI)", font=("Arial", 11, "bold"))
        self.frame_layout.pack(fill="x", padx=10, pady=5)

        self.btn_load_layout = tk.Button(self.frame_layout, text="Load Layout", command=self.load_layout, width=12, state=tk.NORMAL, font=("Arial", 10))
        self.btn_load_layout.grid(row=0, column=0, padx=5, pady=5)

        self.btn_clear_layout = tk.Button(self.frame_layout, text="Hủy Layout", command=self.clear_layout, width=12, state=tk.NORMAL, font=("Arial", 10))
        self.btn_clear_layout.grid(row=0, column=1, padx=5, pady=5)

        self.btn_draw_roi = tk.Button(self.frame_layout, text="Vẽ Vùng Quan Sát", command=self.open_draw_roi, width=16, state=tk.DISABLED, font=("Arial", 10))
        self.btn_draw_roi.grid(row=0, column=2, padx=5, pady=5)

        self.lbl_layout_status = tk.Label(self.frame_layout, text="Layout: Chưa có", fg="red", font=("Arial", 10, "italic"))
        self.lbl_layout_status.grid(row=1, column=0, columnspan=3, pady=2)

        self.frame_action = tk.Frame(root)
        self.frame_action.pack(fill="x", padx=10, pady=10)

        self.btn_start = tk.Button(self.frame_action, text="Bắt đầu Detect", command=self.start_detection, width=25, height=2, state=tk.DISABLED, font=("Arial", 12, "bold"), bg="#4CAF50", fg="black")
        self.btn_start.pack(pady=5)

        self.lbl_status = tk.Label(root, text="Sẵn sàng", fg="black", font=("Arial", 10))
        self.lbl_status.pack(side="bottom", pady=10)


    def _ocr_worker(self):
        """Thread nền: Lấy ảnh từ queue -> chạy PaddleOCR (có format) -> Trả về kết quả"""
        while self._ocr_worker_running:
            try:
                track_id, img_crop = self._ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # Gọi module ocr_processor [cite: 3, 4]
                clean_text, final_text, img_processed, img_plate_color, status_text, dst_w, dst_h = ocr_processor.run_ocr(self.ocr_reader, img_crop)
                
                # Trả kết quả dưới dạng dict để luồng chính nhận
                self._ocr_pending_results[track_id] = {
                    'clean_text': clean_text,
                    'final_text': final_text,
                    'img_processed': img_processed,
                    'img_before': img_crop,
                    'status_text': status_text,
                    'dst_w': dst_w,
                    'dst_h': dst_h
                }
            except Exception as e:
                print(f"[OCR Worker] Lỗi: {e}")
            finally:
                self._ocr_queue.task_done()

    def _start_ocr_worker(self):
        if not self._ocr_worker_running:
            self._ocr_worker_running = True
            t = threading.Thread(target=self._ocr_worker, daemon=True)
            t.start()

    def _stop_ocr_worker(self):
        self._ocr_worker_running = False

    def select_model(self):
        path = filedialog.askopenfilename(title="Chọn Model YOLO", filetypes=[("YOLO Model", "*.pt *.engine"), ("All Files", "*.*")])
        if path:
            self.model_path = path
            self.lbl_model_path.config(text=f"{os.path.basename(self.model_path)}", fg="blue")
            self.model = None

    def select_video(self):
        path = filedialog.askopenfilename(title="Chọn Video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")])
        if path:
            self.video_path = path
            self.lbl_path.config(text=f"{os.path.basename(self.video_path)}", fg="blue")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_draw_roi.config(state=tk.NORMAL)
            
            if self.roi_polygon is None:
                video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                layout_path = os.path.join("layouts", f"{video_name}.json")
                if os.path.exists(layout_path):
                    try:
                        with open(layout_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self.roi_polygon = np.array(data["points"])
                        self.lbl_layout_status.config(text=f"Layout: Tự động tải {os.path.basename(layout_path)}", fg="green")
                        self.update_status(f"Đã tải: {os.path.basename(layout_path)}", "green")
                    except Exception as e:
                        pass

    def load_layout(self):
        path = filedialog.askopenfilename(title="Chọn File Layout", filetypes=[("JSON Files", "*.json")])
        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.roi_polygon = np.array(json.load(f).get("points"))
                self.lbl_layout_status.config(text=f"Layout: Đã load {os.path.basename(path)}", fg="green")

    def clear_layout(self):
        self.roi_polygon = None
        self.lbl_layout_status.config(text="Layout: Chưa có", fg="red")

    def open_draw_roi(self):
        if not self.video_path: return
        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()
        cap.release()
        polygon = self.draw_roi(first_frame)
        if polygon is not None:
            self.roi_polygon = polygon
            self.lbl_layout_status.config(text="Layout: Đã vẽ tạm", fg="orange")

    def draw_roi(self, frame):
        points = []
        if self.roi_polygon is not None: points = [tuple(p) for p in self.roi_polygon.tolist()]
        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN: points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN: points.clear()
        clone = frame.copy()
        cv2.namedWindow("Draw ROI")
        cv2.setMouseCallback("Draw ROI", mouse)
        while True:
            temp = clone.copy()
            for p in points: cv2.circle(temp, p, 5, (0, 0, 255), -1)
            if len(points) > 1: cv2.polylines(temp, [np.array(points)], False, (255, 0, 0), 2)
            cv2.imshow("Draw ROI", temp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == 13: break
        cv2.destroyWindow("Draw ROI")
        return np.array(points) if len(points) >= 3 else None

    def start_detection(self):
        if not self.video_path or self.roi_polygon is None: return
        self.btn_start.config(state=tk.DISABLED)
        threading.Thread(target=self.detect_video, daemon=True).start()

    def update_status(self, text, color="black"):
        self.root.after(0, lambda: self.lbl_status.config(text=text, fg=color))

    def reset_ui(self):
        self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))

    def load_model(self):
        model = YOLO(self.model_path).to("cuda") if not self.model_path.endswith(".engine") else YOLO(self.model_path, task="detect")
        if self.ocr_reader is None:
            self.ocr_reader = PaddleOCR(use_angle_cls=False, det=True, lang='en', show_log=False)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(5): model.predict(dummy, verbose=False)
        return model

    def detect_video(self):
        try:
            if self.model is None: self.model = self.load_model()
            self._start_ocr_worker()
            self.update_status("Đang nhận diện...", "green")

            cap = cv2.VideoCapture(self.video_path)
            target_classes = ["person", "car", "motorcycle", "license_plate", "bus", "truck"]
            
            # Cấu hình Vận tốc và Giao thông
            congestion_threshold = 10
            crowd_threshold = 20
            speed_threshold = 10
            track_history = {}

            # ================= CÁC BIẾN THEO DÕI OCR VÀ BÙ FRAME =================
            plate_history   = {}   # Ghi nhận các text đọc được
            plate_confirmed = {}   # Lưu kết quả chốt cuối cùng
            plate_raw_cache = {}   # Text thô
            active_tracks   = {}   # Bù frame rớt {track_id: {'bbox': (x,y,x,y), 'missed_frames': 0}}
            spatial_memory  = {}   # Kế thừa ID {track_id: (cx, cy, final_text, frame_count)}
            last_seen_plate = {}   # Hỗ trợ dọn rác bộ nhớ
            last_comparison_window = None
            # =====================================================================

            prev_time = time.time()
            fps_frame_count, current_fps, frame_count = 0, 0.0, 0
            last_results = None

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 
            ideal_frame_time = 1.0 / video_fps

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                current_time = time.time()
                frame_count += 1

                # Frame Skipping (Tăng tốc xử lý)
                if frame_count % 2 == 0 and last_results is not None:
                    results = last_results
                else:
                    results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    last_results = results
                
                vehicle_count, people_count = 0, 0
                current_ids_in_roi = []
                current_plate_ids = set()
                
                for r in results:
                    valid_vehicles = []
                    for box in r.boxes:
                        tmp_label = self.model.names[int(box.cls[0])]
                        if tmp_label in ["car", "bus", "truck"]:
                            valid_vehicles.append(tuple(map(int, box.xyxy[0])))

                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        conf = float(box.conf[0])

                        if label not in target_classes or conf <= self.CONF_THRESHOLD:
                            continue

                        track_id = int(box.id[0]) if box.id is not None else -1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Kiểm tra xem vật thể có nằm trong Vùng ROI hay không
                        if cv2.pointPolygonTest(self.roi_polygon, (cx, cy), False) >= 0:
                            box_color = colors.get(cls_id, (255, 255, 255))

                            if label == "person":
                                people_count += 1
                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                cv2.putText(frame, f"ID:{track_id} person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                                
                            elif label in ["car", "motorcycle", "bus", "truck"]:
                                vehicle_count += 1
                                if track_id != -1:
                                    current_ids_in_roi.append(track_id)
                                    if track_id not in track_history: track_history[track_id] = []
                                    track_history[track_id].append((cx, cy, current_time))
                                    track_history[track_id] = [p for p in track_history[track_id] if current_time - p[2] <= 2.0]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                                cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                            elif label == "license_plate" and track_id != -1:
                                # Lọc biển số: Chỉ xử lý OCR nếu tâm biển số nằm trong ô tô/bus/truck
                                is_valid_plate = False
                                for vx1, vy1, vx2, vy2 in valid_vehicles:
                                    if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                                        is_valid_plate = True
                                        break
                                
                                if not is_valid_plate:
                                    continue

                                current_plate_ids.add(track_id)
                                last_seen_plate[track_id] = current_time

                                w, h = x2 - x1, y2 - y1
                                if w <= 20 or h <= 10: continue

                                # 1. Kế thừa Spatial Memory[cite: 3, 4]
                                if track_id not in plate_confirmed:
                                    for old_id, (old_cx, old_cy, old_text, old_frame) in list(spatial_memory.items()):
                                        if frame_count - old_frame < 150 and math.hypot(cx - old_cx, cy - old_cy) < 50:
                                            plate_confirmed[track_id] = old_text
                                            plate_raw_cache[track_id] = "INHERITED"
                                            break

                                # Cập nhật Grace Period
                                active_tracks[track_id] = {'bbox': (x1, y1, x2, y2), 'missed_frames': 0}

                                # 2. Lấy Text và Hiển thị
                                if track_id in plate_confirmed:
                                    final_text = plate_confirmed[track_id]
                                    display_text = f"[OK] {final_text}"
                                    spatial_memory[track_id] = (cx, cy, final_text, frame_count)
                                    
                                elif track_id in self._ocr_pending_results:
                                    # Nhận kết quả từ Queue
                                    res = self._ocr_pending_results.pop(track_id)
                                    final_text = res['final_text']
                                    plate_raw_cache[track_id] = res['clean_text']

                                    # 3. Lọc Regex[cite: 3, 4]
                                    if ocr_processor.is_valid_vn_plate(final_text):
                                        if track_id not in plate_history: plate_history[track_id] = []
                                        plate_history[track_id].append(final_text)

                                        counter = Counter(plate_history[track_id])
                                        best, count = counter.most_common(1)[0]

                                        if count >= self.VOTE_THRESHOLD:
                                            plate_confirmed[track_id] = best
                                            display_text = f"[OK] {best}"
                                            spatial_memory[track_id] = (cx, cy, best, frame_count)
                                        else:
                                            display_text = f"[?] {best} ({count}/{self.VOTE_THRESHOLD})"
                                    else:
                                        display_text = f"[SKIP] {final_text}" if final_text else "..."

                                    # Cập nhật cửa sổ debug (Tùy chọn hiển thị)
                                    img_before_r = cv2.resize(res['img_before'], (res['dst_w'], res['dst_h']))
                                    img_after_r = cv2.resize(res['img_processed'], (res['dst_w'], res['dst_h']))
                                    cv2.putText(img_after_r, f"RAW: {res['clean_text']}", (5, res['dst_h'] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                                    cv2.putText(img_after_r, f"FIX: {display_text}", (5, res['dst_h'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                                    last_comparison_window = np.vstack((img_before_r, img_after_r))

                                else:
                                    display_text = (plate_history.get(track_id) or [plate_raw_cache.get(track_id, "...")])[-1]

                                # Gửi ảnh mới vào Queue để đọc (Chỉ gửi nếu chưa confirm)
                                if track_id not in plate_confirmed and frame_count % self.OCR_INTERVAL == 0:
                                    pad = 2
                                    x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
                                    x2_p, y2_p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
                                    img_crop = frame[y1_p:y2_p, x1_p:x2_p].copy()
                                    try:
                                        self._ocr_queue.put_nowait((track_id, img_crop))
                                    except queue.Full:
                                        pass

                                # Vẽ Box Biển Số
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                color = (0, 255, 0) if "[OK]" in display_text else ((0, 0, 255) if "[SKIP]" in display_text else (0, 255, 255))
                                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # ================= LOGIC BÙ FRAME (GRACE PERIOD) =================
                for tid in list(active_tracks.keys()):
                    if tid not in current_plate_ids:
                        active_tracks[tid]['missed_frames'] += 1
                        if active_tracks[tid]['missed_frames'] > self.MAX_LOST_FRAMES:
                            del active_tracks[tid]
                        else:
                            old_x1, old_y1, old_x2, old_y2 = active_tracks[tid]['bbox']
                            if tid in plate_confirmed:
                                display_text, color = f"[OK] {plate_confirmed[tid]}", (0, 255, 0) 
                            else:
                                display_text = (plate_history.get(tid) or ["..."])[-1]
                                color = (0, 165, 255) 
                            cv2.rectangle(frame, (old_x1, old_y1), (old_x2, old_y2), color, 2)
                            cv2.putText(frame, display_text, (old_x1, old_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                # =================================================================

                # Dọn dẹp rác bộ nhớ (Memory Leak Prevention)
                for tid in list(last_seen_plate.keys()):
                    if current_time - last_seen_plate[tid] > 5.0: # Không thấy trong 5 giây thì xóa
                        plate_history.pop(tid, None)
                        plate_confirmed.pop(tid, None)
                        plate_raw_cache.pop(tid, None)
                        del last_seen_plate[tid]
                
                for sid in list(spatial_memory.keys()):
                    if frame_count - spatial_memory[sid][3] > 300: # Xóa vết cũ sau khoảng 10s
                        del spatial_memory[sid]

                total_speed, valid_speed_count = 0.0, 0
                for tid in list(track_history.keys()):
                    if tid not in current_ids_in_roi:
                        if len(track_history[tid]) > 0 and (current_time - track_history[tid][-1][2]) > 1.0:
                            del track_history[tid]
                        continue
                    points = track_history[tid]
                    if len(points) >= 2:
                        dt = points[-1][2] - points[0][2]
                        if dt > 0.2: 
                            speed = np.sqrt((points[-1][0]-points[0][0])**2 + (points[-1][1]-points[0][1])**2) / dt 
                            total_speed += speed
                            valid_speed_count += 1

                avg_speed = total_speed / valid_speed_count if valid_speed_count > 0 else 0.0

                if vehicle_count > congestion_threshold or people_count > crowd_threshold:
                    if valid_speed_count > 0 and avg_speed < speed_threshold:
                        status_text, status_color = "Trang thai: TAC NGHEN!", (0, 0, 255) 
                    else:
                        status_text, status_color = "Trang thai: Dong duc", (0, 165, 255) 
                else:
                    status_text, status_color = "Trang thai: Thong thoang", (0, 255, 0) 

                cv2.polylines(frame, [self.roi_polygon], True, (255, 0, 0), 2)
                cv2.putText(frame, "Bam ESC de thoat", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Vehicles: {vehicle_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Avg Speed: {int(avg_speed)} px/s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, status_text, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                curr_time = time.time()
                fps_frame_count += 1
                if curr_time - prev_time >= 1.0:
                    current_fps, prev_time, fps_frame_count = fps_frame_count / (curr_time - prev_time), curr_time, 0

                cv2.putText(frame, f"FPS: {int(current_fps)}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if last_comparison_window is not None:
                    cv2.imshow("Before vs After Processing", last_comparison_window)
                cv2.imshow("Vehicle Detection", frame)

                processing_time = time.time() - current_time 
                wait_time_ms = max(1, int((ideal_frame_time - processing_time) * 1000)) 

                if cv2.waitKey(wait_time_ms) == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
            self._stop_ocr_worker()
            self.update_status("Đã hoàn thành!", "black")
            self.reset_ui()

        except Exception as e:
            self._stop_ocr_worker()
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            self.reset_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()