import time
import os
import cv2
from modules.utils.interactive_telegram_bot import send_alert_with_button

class TrafficAlertManager:
    def __init__(self):
        # Cấu hình thời gian đếm ngược (giây)
        self.INTERVAL_UNACK_L1 = 180   # 3 phút
        self.INTERVAL_UNACK_L2 = 60   # 1 phút
        self.INTERVAL_UNACK_L3 = 30    # 30 giây
        self.SNOOZE_ACK_L1 = 900      # 15 phút
        self.SNOOZE_ACK_L2 = 600       # 10 phút
        self.SNOOZE_ACK_L3 = 300       # 5 phút
        
        # Các biến trạng thái quản lý
        self.current_level = 0 # 0: Thông thoáng, 1: Đông đúc, 2: Tắc nghẽn
        self.last_alert_time = 0
        self.is_acknowledged = False
        
        # Đảm bảo thư mục lưu log tồn tại
        os.makedirs("logs", exist_ok=True)

    def update_traffic_state(self, level, clean_frame, bot_token, chat_id):
        current_time = time.time()
        
        if level == 0:
            self.current_level = 0
            self.is_acknowledged = False
            self.last_alert_time = 0
            return
            
        if level > self.current_level:
            # Tăng cấp độ cảnh báo (Escalation) khi tình hình xấu đi
            self.current_level = level
            self.is_acknowledged = False
            self._trigger_alert(level, clean_frame, bot_token, chat_id)
            self.last_alert_time = current_time
            
        elif level == self.current_level:
            # Kiểm tra thời gian chờ (Cooldown check)
            if level == 1:
                cooldown = self.SNOOZE_ACK_L1 if self.is_acknowledged else self.INTERVAL_UNACK_L1
            elif level == 2:
                cooldown = self.SNOOZE_ACK_L2 if self.is_acknowledged else self.INTERVAL_UNACK_L2
            elif level == 3:
                cooldown = self.SNOOZE_ACK_L3 if self.is_acknowledged else self.INTERVAL_UNACK_L3
            else:
                return
                
            if current_time - self.last_alert_time >= cooldown:
                self._trigger_alert(level, clean_frame, bot_token, chat_id)
                self.last_alert_time = current_time
                self.is_acknowledged = False # Bắt buộc phải bấm xác nhận (Ack) lại lần nữa

    def acknowledge_alert(self):
        self.is_acknowledged = True
        print("[INFO] Người dùng đã xác nhận Cảnh báo. Hệ thống tạm chuyển sang chế độ Ngủ đông (Snooze).")

    def _trigger_alert(self, level, frame, bot_token, chat_id):
        img_path = "logs/traffic_alert.jpg"
        cv2.imwrite(img_path, frame)
        
        caption = ""
        if level == 1:
            caption = "⚠️ CẢNH BÁO: Giao thông đang Bắt Đầu Đông (Mức 1)."
        elif level == 2:
            caption = "⚠️ CẢNH BÁO: Giao thông đang RẤT ĐÔNG (Mức 2)."
        elif level == 3:
            caption = "🚨 BÁO ĐỘNG: TẮC NGHẼN nghiêm trọng (Mức 3)!"
            
        # Gửi sang Bot Telegram có đính kèm Nút nhấn tương tác
        send_alert_with_button(img_path, caption)
