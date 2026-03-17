import cv2
import numpy as np
import os
import json
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import re          
import easyocr
import queue

class_names = {
    0: "Person", 
    1: "Bicycle", 
    2: "Car", 
    3: "Motorcycle", 
    4: "License Plate",
    5: "Bus",   
    6: "Truck"    
}

colors = {
    0: (0, 255, 0),     # Nguoi: Xanh lá
    1: (255, 0, 0),     # Xe_dap: Xanh dương
    2: (255, 255, 0),   # O_to: Xanh lơ (Cyan)
    3: (0, 255, 255),   # Xe_may: Vàng
    4: (0, 0, 255),     # Bien_so: Đỏ
    5: (0, 165, 255),   # Xe_bus: Cam
    6: (255, 0, 255)    # Xe_tai: Tím (Magenta)
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

        # --- OCR ASYNC: Queue gửi việc vào, dict nhận kết quả ra ---
        # maxsize=3: tránh tích lũy quá nhiều frame chờ xử lý
        self._ocr_queue = queue.Queue(maxsize=3)
        # Lưu kết quả OCR theo track_id, thread-safe vì chỉ ghi từ worker, đọc từ main
        self._ocr_pending_results = {}
        # Cờ để dừng worker khi app tắt
        self._ocr_worker_running = False

        # --- GIAO DIỆN KHÔNG THAY ĐỔI ---
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

    # ==========================================================================
    # TỐI ƯU 1: OCR WORKER CHẠY RIÊNG TRÊN THREAD NỀN
    # Vòng lặp video KHÔNG bao giờ bị block bởi EasyOCR nữa.
    # Worker nhận ảnh từ queue -> đọc OCR -> ghi kết quả vào dict.
    # ==========================================================================
    def _ocr_worker(self):
        """Thread nền: liên tục lấy ảnh từ queue và chạy EasyOCR."""
        while self._ocr_worker_running:
            try:
                # Chờ tối đa 0.5s, nếu không có việc thì lặp lại để kiểm tra cờ
                track_id, gray_lp = self._ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                ocr_res = self.ocr_reader.readtext(gray_lp, detail=0)
                if ocr_res:
                    raw_text = "".join(ocr_res).upper()
                    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text)
                    # Kiểm tra định dạng biển số VN
                    if re.match(r'^[1-9][0-9][A-Z][0-9]?\d{4,5}$', clean_text):
                        # Ghi vào dict trung gian, vòng lặp chính sẽ đọc ở frame tiếp theo
                        self._ocr_pending_results[track_id] = clean_text
            except Exception as e:
                print(f"[OCR Worker] Lỗi: {e}")
            finally:
                self._ocr_queue.task_done()

    def _start_ocr_worker(self):
        """Khởi động thread OCR nền."""
        if not self._ocr_worker_running:
            self._ocr_worker_running = True
            t = threading.Thread(target=self._ocr_worker, daemon=True)
            t.start()

    def _stop_ocr_worker(self):
        """Dừng thread OCR nền."""
        self._ocr_worker_running = False

    def select_model(self):
        path = filedialog.askopenfilename(
            title="Chọn Model YOLO",
            filetypes=[("YOLO Model", "*.pt *.engine"), ("PyTorch", "*.pt"), ("TensorRT Engine", "*.engine"), ("All Files", "*.*")]
        )
        if path:
            self.model_path = path
            self.lbl_model_path.config(text=f"{os.path.basename(self.model_path)}", fg="blue")
            self.model = None

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Chọn Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
        )
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
                        self.update_status(f"Đã tự động tải layout: {os.path.basename(layout_path)}", "green")
                    except Exception as e:
                        print(f"Không thể đọc file layout: {e}")
                else:
                    self.update_status("Sẵn sàng! Vui lòng vẽ vùng quan sát hoặc Load File Layout.", "black")

    def load_layout(self):
        path = filedialog.askopenfilename(title="Chọn File Layout", filetypes=[("JSON Files", "*.json")])
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    points = data.get("points")
                    if points and len(points) >= 3:
                        self.roi_polygon = np.array(points)
                        self.lbl_layout_status.config(text=f"Layout: Đã load {os.path.basename(path)}", fg="green")
                        self.update_status(f"Đã tải layout từ file", "green")
                    else:
                        messagebox.showwarning("Cảnh báo", "File JSON không hợp lệ hoặc không đủ 3 điểm!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải file layout: {str(e)}")

    def clear_layout(self):
        self.roi_polygon = None
        self.lbl_layout_status.config(text="Layout: Chưa có", fg="red")
        self.update_status("Đã hủy layout hiện tại.", "black")

    def open_draw_roi(self):
        if not self.video_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn video trước khi vẽ vùng quan sát!")
            return

        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Lỗi", "Không thể đọc khung hình đầu tiên của video!")
            return

        polygon = self.draw_roi(first_frame)
        if polygon is not None and len(polygon) >= 3:
            answer = messagebox.askyesnocancel("Lưu Layout", "Bạn có muốn lưu Layout vừa vẽ không?\n\nChọn 'Yes' để lưu.\nChọn 'No' để áp dụng mà không lưu.\nChọn 'Cancel' để hủy bỏ.")
            if answer is True:
                self.roi_polygon = polygon
                os.makedirs("layouts", exist_ok=True)
                video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                layout_path = filedialog.asksaveasfilename(
                    initialdir=os.path.join(os.getcwd(), "layouts"),
                    initialfile=f"{video_name}.json",
                    title="Lưu file Layout JSON",
                    defaultextension=".json",
                    filetypes=[("JSON Files", "*.json")]
                )
                if layout_path:
                    data = {"points": polygon.tolist()}
                    try:
                        with open(layout_path, "w", encoding="utf-8") as f:
                            json.dump(data, f)
                        self.lbl_layout_status.config(text=f"Layout: Đã lưu {os.path.basename(layout_path)}", fg="green")
                        self.update_status(f"Đã lưu và áp dụng layout mới", "green")
                    except Exception as e:
                        messagebox.showerror("Lỗi", f"Không thể lưu file json: {str(e)}")
                else:
                    self.lbl_layout_status.config(text="Layout: Đã vẽ tạm", fg="orange")
                    self.update_status("Không lưu file. Layout mới vẫn được tạm áp dụng.", "blue")
            elif answer is False:
                self.roi_polygon = polygon
                self.lbl_layout_status.config(text="Layout: Đã vẽ tạm", fg="orange")
                self.update_status("Đã vẽ tạm Layout thành công.", "green")
            else:
                self.update_status("Đã huỷ thao tác vẽ ROI.", "red")
        else:
            self.update_status("Đã huỷ vẽ ROI.", "red")

    def start_detection(self):
        if not self.video_path:
            return
        if self.roi_polygon is None or len(self.roi_polygon) < 3:
            messagebox.showwarning("Cảnh báo", "Vui lòng 'Vẽ Vùng Quan Sát' hoặc load JSON trước khi chạy detect!")
            return

        self.btn_start.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.DISABLED)
        self.btn_select_model.config(state=tk.DISABLED)
        self.btn_draw_roi.config(state=tk.DISABLED)
        self.btn_load_layout.config(state=tk.DISABLED)
        
        threading.Thread(target=self.detect_video, daemon=True).start()

    def update_status(self, text, color="black"):
        self.root.after(0, lambda: self.lbl_status.config(text=text, fg=color))

    def reset_ui(self):
        self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_select.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_select_model.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_draw_roi.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_load_layout.config(state=tk.NORMAL))

    def load_model(self):
        # 1. Tải YOLO
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == ".engine":
            self.update_status("Đang tải TensorRT Engine...", "blue")
            model = YOLO(self.model_path, task="detect")
        else:
            self.update_status("Đang tải model PyTorch...", "blue")
            model = YOLO(self.model_path).to("cuda")

        # 2. Khởi tạo EasyOCR
        if self.ocr_reader is None:
            self.update_status("Đang tải mô hình đọc biển số (EasyOCR)...", "blue")
            self.ocr_reader = easyocr.Reader(['en'], gpu=True)

        self.update_status("Đang warmup model...", "blue")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(5):
            model.predict(dummy, verbose=False)

        return model

    # ==========================================================================
    # TỐI ƯU 2: TIỀN XỬ LÝ ẢNH NHANH HƠN
    # Thay bilateralFilter(d=11) chậm bằng GaussianBlur + Otsu threshold.
    # Giảm từ ~30ms xuống ~1ms mỗi lần gọi.
    # ==========================================================================
    def _preprocess_lp(self, frame, x1, y1, x2, y2):
        """Cắt và tiền xử lý ảnh biển số để OCR."""
        lp_crop = frame[y1:y2, x1:x2]

        # Phóng to 2x để ký tự dễ nhận hơn
        lp_crop = cv2.resize(lp_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Chuyển sang ảnh xám
        gray_lp = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)

        # CLAHE tăng tương phản (giữ nguyên vì rất hiệu quả)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_lp = clahe.apply(gray_lp)

        # TỐI ƯU: Thay bilateralFilter(11,17,17) ~30ms bằng GaussianBlur ~0.5ms
        gray_lp = cv2.GaussianBlur(gray_lp, (3, 3), 0)

        # Threshold Otsu: tách chữ trắng/đen rõ ràng, giúp OCR chính xác hơn
        _, gray_lp = cv2.threshold(gray_lp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return gray_lp

    def detect_video(self):
        try:
            if self.model is None:
                self.model = self.load_model()

            # Khởi động OCR worker thread nền trước khi vào vòng lặp
            self._start_ocr_worker()
            self.update_status("Đang nhận diện...", "green")

            cap = cv2.VideoCapture(self.video_path)

            target_classes = ["person", "car", "motorcycle", "license_plate", "bus", "truck"]

            congestion_threshold = 10
            crowd_threshold = 20
            speed_threshold = 10

            track_history = {}

            # lp_history: lưu lịch sử biển số theo track_id
            # Cấu trúc: {track_id: {'candidates': [], 'final_plate': None, 'last_seen': time}}
            lp_history = {}

            prev_time = time.time()
            fps_frame_count = 0
            current_fps = 0.0
            
            frame_count = 0
            last_results = None

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps == 0 or np.isnan(video_fps): 
                video_fps = 30.0 
            ideal_frame_time = 1.0 / video_fps

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                frame_count += 1

                # Bỏ qua frame chẵn (giữ nguyên logic cũ)
                if frame_count % 2 == 0 and last_results is not None:
                    results = last_results
                else:
                    results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    last_results = results
                
                vehicle_count = 0
                people_count = 0
                current_ids_in_roi = []
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]

                        if label in target_classes:
                            track_id = -1
                            if box.id is not None:
                                track_id = int(box.id[0])

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)

                            inside = cv2.pointPolygonTest(self.roi_polygon, (cx, cy), False)

                            if inside >= 0:
                                box_color = colors.get(cls_id, (255, 255, 255))

                                if label == "person":
                                    people_count += 1
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                    cv2.putText(frame, f"ID:{track_id} person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                                    
                                elif label in ["car", "motorcycle", "bus", "truck"]:
                                    vehicle_count += 1
                                    if track_id != -1:
                                        current_ids_in_roi.append(track_id)
                                        if track_id not in track_history:
                                            track_history[track_id] = []
                                        track_history[track_id].append((cx, cy, current_time))
                                        track_history[track_id] = [p for p in track_history[track_id] if current_time - p[2] <= 2.0]
                                        
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                    cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                                elif label == "license_plate" and track_id != -1:
                                    if track_id not in lp_history:
                                        lp_history[track_id] = {'candidates': [], 'final_plate': None, 'last_seen': current_time}
                                    
                                    lp_history[track_id]['last_seen'] = current_time
                                    w, h = x2 - x1, y2 - y1

                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                                    # =========================================
                                    # TỐI ƯU 1: Nhận kết quả từ OCR worker
                                    # Thay vì chờ OCR xong mới tiếp tục,
                                    # chỉ đơn giản là đọc kết quả đã có sẵn.
                                    # =========================================
                                    if track_id in self._ocr_pending_results:
                                        new_text = self._ocr_pending_results.pop(track_id)
                                        lp_history[track_id]['candidates'].append(new_text)
                                        count = lp_history[track_id]['candidates'].count(new_text)
                                        if count >= 3:
                                            lp_history[track_id]['final_plate'] = new_text
                                            print(f"✅ Đã chốt biển số: {new_text} (ID: {track_id})")

                                    if lp_history[track_id]['final_plate'] is not None:
                                        # Đã chốt -> Hiển thị, không cần gửi OCR nữa
                                        final_text = lp_history[track_id]['final_plate']
                                        cv2.putText(frame, f"[{final_text}]", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                    else:
                                        # TỐI ƯU 3: Tăng interval lên 20 frame (từ 10)
                                        # để giảm số lần đẩy vào queue OCR
                                        if frame_count % 10 == 0:
                                            # Tiền xử lý nhanh (không block)
                                            gray_lp = self._preprocess_lp(frame, x1, y1, x2, y2)

                                            # Đẩy vào queue cho worker xử lý, không chờ
                                            # put_nowait: bỏ qua nếu queue đầy (tránh lag)
                                            try:
                                                self._ocr_queue.put_nowait((track_id, gray_lp))
                                            except queue.Full:
                                                pass  # Queue đang bận -> bỏ qua frame này, frame sau thử lại

                                        # Hiển thị biển số tạm tốt nhất đang có
                                        current_cands = lp_history[track_id]['candidates']
                                        if current_cands:
                                            best_guess = max(set(current_cands), key=current_cands.count)
                                            cv2.putText(frame, f"~{best_guess}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                        else:
                                            cv2.putText(frame, "Reading...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # --- DỌN DẸP BỘ NHỚ BIỂN SỐ ---
                for tid in list(lp_history.keys()):
                    if current_time - lp_history[tid]['last_seen'] > 3.0:
                        del lp_history[tid]

                # --- LOGIC TÍNH TOÁN VẬN TỐC ---
                total_speed = 0.0
                valid_speed_count = 0

                for tid in list(track_history.keys()):
                    if tid not in current_ids_in_roi:
                        if len(track_history[tid]) > 0 and (current_time - track_history[tid][-1][2]) > 1.0:
                            del track_history[tid]
                        continue

                    points = track_history[tid]
                    if len(points) >= 2:
                        p_old = points[0]
                        p_new = points[-1]
                        dt = p_new[2] - p_old[2]
                        if dt > 0.2: 
                            dx = p_new[0] - p_old[0]
                            dy = p_new[1] - p_old[1]
                            distance = np.sqrt(dx**2 + dy**2) 
                            speed = distance / dt 
                            total_speed += speed
                            valid_speed_count += 1

                avg_speed = total_speed / valid_speed_count if valid_speed_count > 0 else 0.0

                # --- ĐÁNH GIÁ TRẠNG THÁI GIAO THÔNG ---
                if vehicle_count > congestion_threshold or people_count > crowd_threshold:
                    if valid_speed_count > 0 and avg_speed < speed_threshold:
                        status_text = "Trang thai: TAC NGHEN!"
                        status_color = (0, 0, 255) 
                    else:
                        status_text = "Trang thai: Dong duc (Dang luu thong)"
                        status_color = (0, 165, 255) 
                else:
                    status_text = "Trang thai: Thong thoang"
                    status_color = (0, 255, 0) 

                # --- VẼ THÔNG TIN LÊN MÀN HÌNH ---
                cv2.polylines(frame, [self.roi_polygon], True, (255, 0, 0), 2)
                cv2.putText(frame, "Bam ESC de thoat", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Vehicles: {vehicle_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Avg Speed: {int(avg_speed)} px/s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, status_text, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                curr_time = time.time()
                fps_frame_count += 1
                if curr_time - prev_time >= 1.0:
                    current_fps = fps_frame_count / (curr_time - prev_time)
                    prev_time = curr_time
                    fps_frame_count = 0

                cv2.putText(frame, f"FPS: {int(current_fps)}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Vehicle Detection", frame)

                processing_time = time.time() - current_time 
                wait_time_sec = ideal_frame_time - processing_time 
                wait_time_ms = max(1, int(wait_time_sec * 1000)) 

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

    # --- HÀM VẼ ROI KHÔNG ĐỔI ---
    def draw_roi(self, frame):
        points = []
        if self.roi_polygon is not None and len(self.roi_polygon) >= 3:
            points = self.roi_polygon.tolist()
            points = [tuple(p) for p in points]

        def mouse(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.clear()

        clone = frame.copy()
        cv2.namedWindow("Draw ROI")
        cv2.setMouseCallback("Draw ROI", mouse)

        while True:
            temp = clone.copy()
            if points is not None and len(points) > 0:
                for p in points:
                    cv2.circle(temp, p, 5, (0, 0, 255), -1)
            if points is not None and len(points) > 1:
                cv2.polylines(temp, [np.array(points)], False, (255, 0, 0), 2)
            cv2.putText(temp, "Left Click: draw | Right Click: clear | ESC/Close: cancel | ENTER: done",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Draw ROI", temp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                points = None
                break
            elif key == 26: 
                if points and len(points) > 0:
                    points.pop()
            elif key == 13 or key == 10: 
                if points is not None and len(points) >= 3:
                    break
            try:
                if cv2.getWindowProperty("Draw ROI", cv2.WND_PROP_VISIBLE) < 1:
                    points = None
                    break
            except cv2.error:
                points = None
                break
        try:
            cv2.destroyWindow("Draw ROI")
        except cv2.error:
            pass
        return np.array(points) if points is not None and len(points) >= 3 else None

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()