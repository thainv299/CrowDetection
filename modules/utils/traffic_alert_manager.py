import time
import os
import cv2
from modules.utils.interactive_telegram_bot import send_alert_with_button

class TrafficAlertManager:
    def __init__(self):
        # Configs (seconds)
        self.INTERVAL_UNACK_L1 = 180   # 3 mins
        self.INTERVAL_UNACK_L2 = 60   # 1 mins
        self.INTERVAL_UNACK_L3 = 30    # 30 secs
        self.SNOOZE_ACK_L1 = 900      # 15 mins
        self.SNOOZE_ACK_L2 = 600       # 10 mins
        self.SNOOZE_ACK_L3 = 300       # 5 mins
        
        # State variables
        self.current_level = 0 # 0: Clear, 1: Crowded, 2: Congested
        self.last_alert_time = 0
        self.is_acknowledged = False
        
        # Ensure log directory
        os.makedirs("logs", exist_ok=True)

    def update_traffic_state(self, level, clean_frame, bot_token, chat_id):
        current_time = time.time()
        
        if level == 0:
            self.current_level = 0
            self.is_acknowledged = False
            self.last_alert_time = 0
            return
            
        if level > self.current_level:
            # Escalation
            self.current_level = level
            self.is_acknowledged = False
            self._trigger_alert(level, clean_frame, bot_token, chat_id)
            self.last_alert_time = current_time
            
        elif level == self.current_level:
            # Cooldown check
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
                self.is_acknowledged = False # Force them to ack again

    def acknowledge_alert(self):
        self.is_acknowledged = True
        print("[INFO] Alert Acknowledged by user. Entering Snooze mode.")

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
            
        # Call interactive telegram bot
        send_alert_with_button(img_path, caption)
