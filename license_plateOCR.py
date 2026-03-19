import cv2
import numpy as np
import re
from collections import Counter
from ultralytics import YOLO
from paddleocr import PaddleOCR

VIDEO_PATH = r"E:\DATN_code\videos\Parking\parking_video3.mp4"
MODEL_PATH = r"E:\DATN_code\models\best.pt"

OCR_INTERVAL   = 4       # Chỉ chạy OCR mỗi N frame (tăng tốc)
VOTE_THRESHOLD = 3       # Cần đọc đúng bao nhiêu lần để confirm
MIN_PLATE_LEN  = 7       # Độ dài tối thiểu biển số hợp lệ
CONF_THRESHOLD = 0.32    # Ngưỡng confidence của YOLO


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def preprocess_plate(img_bgr):
    """Tiền xử lý nhẹ nhàng: Scale x2 và CLAHE, giữ lại cấu trúc nét chữ."""
    # Scale x2 giúp OCR nhận diện chi tiết tốt hơn
    h, w = img_bgr.shape[:2]
    img_scaled = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    
    # Cân bằng sáng nhẹ (clipLimit nhỏ) tránh đẩy nhiễu lên thành vệt đen
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    img_enhanced = clahe.apply(img_gray)
    
    # PaddleOCR hoạt động ổn định nhất khi nhận đầu vào là ảnh 3 channel (BGR)
    img_enhanced_bgr = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)
    return img_enhanced_bgr


def correct_plate_format(text):
    """Sửa lỗi OCR dựa trên cấu trúc biển số Việt Nam."""
    if len(text) < MIN_PLATE_LEN:
        return text

    dict_char_to_num = {'O': '0', 'Q': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
    dict_num_to_char = {'0': 'D', '8': 'B', '4' : 'A'}

    text_list = list(text)

    # LUẬT 1: 2 ký tự đầu (mã tỉnh) phải là SỐ
    for i in range(0, 2):
        if text_list[i].isalpha() and text_list[i] in dict_char_to_num:
            text_list[i] = dict_char_to_num[text_list[i]]

    # LUẬT 2: Ký tự thứ 3 (sê-ri) phải là CHỮ
    if text_list[2].isdigit() and text_list[2] in dict_num_to_char:
        text_list[2] = dict_num_to_char[text_list[2]]

    # LUẬT 3: 4 ký tự cuối phải là SỐ
    for i in range(max(3, len(text_list) - 4), len(text_list)):
        if text_list[i].isalpha() and text_list[i] in dict_char_to_num:
            text_list[i] = dict_char_to_num[text_list[i]]

    return "".join(text_list)


def run_ocr(ocr_reader, img_bgr):
    """Chạy OCR toàn bộ ảnh, dùng PaddleOCR detection để tự động gom dòng."""
    img_processed = preprocess_plate(img_bgr)
    read_text = ""
    
    # Chạy OCR (yêu cầu det=True khi khởi tạo PaddleOCR)
    res = ocr_reader.ocr(img_processed, cls=False)
    
    if res and res[0]:
        # Sắp xếp các bounding box text tìm được theo tọa độ Y (từ trên xuống dưới)
        lines = sorted(res[0], key=lambda x: x[0][0][1])
        for line in lines:
            read_text += line[1][0].upper()

    clean_text = re.sub(r'[^A-Z0-9]', '', read_text)
    final_text = correct_plate_format(clean_text)
    
    return clean_text, final_text, img_processed


def main():
    model = YOLO(MODEL_PATH).to("cuda")

    # QUAN TRỌNG: Đổi det=True để Paddle tự tìm vùng chữ trên biển số
    ocr_reader = PaddleOCR(
        use_angle_cls=False,
        det=True, 
        lang='en',
        use_gpu=True,
        show_log=False
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0

    plate_history   = {}   
    plate_confirmed = {}   
    plate_raw_cache = {}   

    last_comparison_window = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(frame, persist=True, verbose=False, imgsz=640)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                conf   = float(box.conf[0])

                if label != "license_plate" or conf <= CONF_THRESHOLD:
                    continue

                track_id = int(box.id[0]) if box.id is not None else -1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w <= 20 or h <= 10:
                    continue

                # Giảm padding để không bị dính nẹp xe hay ngoại cảnh gây nhiễu
                pad = 2
                x1_p = max(0, x1 - pad)
                y1_p = max(0, y1 - pad)
                x2_p = min(frame.shape[1], x2 + pad)
                y2_p = min(frame.shape[0], y2 + pad)

                img_before = frame[y1_p:y2_p, x1_p:x2_p].copy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                ratio = w / h
                if ratio < 1.8:
                    dst_w, dst_h = 240, 180   
                else:
                    dst_w, dst_h = 480, 120   

                # --- Perspective correction ---
                gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(blur, 50, 200)
                dilated = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

                rect_pts = None
                for c in contours:
                    perimeter = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
                    if len(approx) == 4:
                        rect_pts = approx.reshape(4, 2)
                        break

                if rect_pts is not None:
                    ordered_pts = order_points(rect_pts)
                    dst_pts = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                    img_plate_color = cv2.warpPerspective(img_before, M, (dst_w, dst_h))
                    status_text = f"Nan goc ({dst_w}x{dst_h})"
                else:
                    img_plate_color = cv2.resize(img_before, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
                    status_text = f"Phong to ({dst_w}x{dst_h})"

                # --- OCR với voting ---
                if track_id in plate_confirmed:
                    clean_text = plate_raw_cache.get(track_id, "")
                    final_text = plate_confirmed[track_id]
                    display_text = f"[OK] {final_text}"
                    img_processed = preprocess_plate(img_plate_color)

                elif frame_count % OCR_INTERVAL == 0:
                    # Chạy OCR truyền thẳng ảnh BGR
                    clean_text, final_text, img_processed = run_ocr(ocr_reader, img_plate_color)
                    plate_raw_cache[track_id] = clean_text

                    if len(final_text) >= MIN_PLATE_LEN and track_id != -1:
                        if track_id not in plate_history:
                            plate_history[track_id] = []
                        plate_history[track_id].append(final_text)

                        counter = Counter(plate_history[track_id])
                        best, count = counter.most_common(1)[0]

                        if count >= VOTE_THRESHOLD:
                            plate_confirmed[track_id] = best
                            display_text = f"[OK] {best}"
                        else:
                            display_text = f"[?] {best} ({count}/{VOTE_THRESHOLD})"
                    else:
                        display_text = final_text if final_text else "..."
                else:
                    clean_text = plate_raw_cache.get(track_id, "")
                    final_text = (plate_history.get(track_id) or ["..."])[-1]
                    display_text = final_text
                    img_processed = preprocess_plate(img_plate_color)

                # --- Hiển thị cửa sổ so sánh ---
                DISPLAY_W, DISPLAY_H = dst_w, dst_h
                img_before_resized = cv2.resize(img_before, (DISPLAY_W, DISPLAY_H))
                
                # Resize ảnh sau xử lý để hiển thị vstack
                img_after_bgr = cv2.resize(img_processed, (DISPLAY_W, DISPLAY_H))

                cv2.putText(img_before_resized, "BEFORE", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(img_after_bgr, status_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img_after_bgr, f"RAW : {clean_text}", (5, DISPLAY_H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(img_after_bgr, f"FIX : {display_text}", (5, DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                last_comparison_window = np.vstack((img_before_resized, img_after_bgr))

                color = (0, 255, 0) if "[OK]" in display_text else (0, 0, 255)
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if last_comparison_window is not None:
            cv2.imshow("Before vs After Processing", last_comparison_window)
        cv2.imshow("Main Video", frame)

        key = cv2.waitKey(1)
        if key == 27:      
            break
        elif key == 32:    
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()