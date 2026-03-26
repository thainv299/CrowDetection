import time
import os
import cv2
from modules.utils.interactive_telegram_bot import send_alert_with_button

class TrafficAlertManager:
    def __init__(self):
        # Cấu hình thời gian đếm ngược tĩnh (giây)
        self.DEBOUNCE_SECONDS = 1.0
        self.INTERVAL_UNACK = {1: 300, 2: 60, 3: 30}
        self.SNOOZE_ACK = {1: 900, 2: 600, 3: 300}
        
        # Các biến trạng thái quản lý (Debounce & Tiết chế Cảnh báo)
        self.pending_level = 0
        self.pending_start_time = 0
        self.confirmed_level = 0
        self.last_alert_level = 0
        self.snooze_until = 0
        self.is_acknowledged = False
        
        # Đảm bảo thư mục lưu log tồn tại
        os.makedirs("logs", exist_ok=True)

    def update_traffic_state(self, raw_level, clean_frame, bot_token, chat_id):
        current_time = time.time()
        
        # --- BƯỚC A: Lọc nhiễu (Debounce Logic) ---
        if raw_level != self.pending_level:
            self.pending_level = raw_level
            self.pending_start_time = current_time
            
        if current_time - self.pending_start_time >= self.DEBOUNCE_SECONDS:
            self.confirmed_level = self.pending_level
            
        # --- BƯỚC B: Cảnh báo (Alert Logic) ---
        if self.confirmed_level == 0:
            self.last_alert_level = 0
            return
            
        is_escalation = self.confirmed_level > self.last_alert_level
        timer_expired = current_time >= self.snooze_until
        
        if is_escalation or timer_expired:
            self._trigger_alert(self.confirmed_level, clean_frame, bot_token, chat_id)
            self.last_alert_level = self.confirmed_level
            # Kích hoạt trạng thái Un-Acked với thời gian ngắn
            self.snooze_until = current_time + self.INTERVAL_UNACK.get(self.confirmed_level, 60)
            self.is_acknowledged = False # Đặt lại cờ chưa xác nhận khi có cảnh báo mới phát ra

    def acknowledge_alert(self):
        """User clicked ACK button on Telegram or pressed 'A' on keyboard"""
        self.snooze_until = time.time() + self.SNOOZE_ACK.get(self.confirmed_level, 300)
        self.is_acknowledged = True
        print(f"[INFO] Người dùng đã xác nhận Cảnh báo. Hệ thống tạm chuyển sang chế độ Ngủ đông (Snooze) cho mức độ <= {self.confirmed_level}.")

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
