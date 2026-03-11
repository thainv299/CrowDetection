import cv2
import os

img_path = "Dataset/archive/images/train/dieu_0061.png" 
txt_path = "Dataset/archive/labels/train/dieu_0061.txt"

class_names = {0: "Nguoi", 1: "Xe_dap", 2: "O_to", 3: "Xe_may", 4: "Bien_so"}

img = cv2.imread(img_path)
h, w, _ = img.shape

with open(txt_path, "r") as f:
    lines = f.readlines()

# Duyệt từng dòng label và vẽ
for line in lines:
    data = line.strip().split()
    if len(data) == 0: continue
    
    class_id = int(data[0])
    x_c, y_c, box_w, box_h = map(float, data[1:])
    
    # Công thức giải chuẩn hóa tọa độ YOLO sang tọa độ góc của OpenCV (pixel)
    x1 = int((x_c - box_w / 2) * w)
    y1 = int((y_c - box_h / 2) * h)
    x2 = int((x_c + box_w / 2) * w)
    y2 = int((y_c + box_h / 2) * h)
    
    # Đặt màu ngẫu nhiên hoặc cố định. Ở đây dùng màu xanh lá (0, 255, 0)
    color = (0, 255, 0) if class_id != 4 else (0, 0, 255) # Biển số vẽ màu đỏ 
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, class_names.get(class_id, f"ID_{class_id}"), (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imshow("Kiem tra Auto-label", img)
cv2.waitKey(0)
cv2.destroyAllWindows()