import numpy as np
import cv2

class TrafficMonitor:
    def __init__(self, congestion_threshold=10, crowd_threshold=20, speed_threshold=10):
        self.congestion_threshold = congestion_threshold
        self.crowd_threshold = crowd_threshold
        self.speed_threshold = speed_threshold
        
        self.track_history = {}
        self.vehicle_count = 0
        self.people_count = 0
        self.current_ids_in_roi = []
        
    def reset_counters(self):
        self.vehicle_count = 0
        self.people_count = 0
        self.current_ids_in_roi = []

    def log_person(self):
        self.people_count += 1

    def log_vehicle(self, track_id, cx, cy, current_time):
        self.vehicle_count += 1
        if track_id != -1:
            self.current_ids_in_roi.append(track_id)
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((cx, cy, current_time))
            self.track_history[track_id] = [p for p in self.track_history[track_id] if current_time - p[2] <= 2.0]

    def calculate_speed_and_status(self, current_time):
        total_speed = 0.0
        valid_speed_count = 0
        for tid in list(self.track_history.keys()):
            if tid not in self.current_ids_in_roi:
                if len(self.track_history[tid]) > 0 and (current_time - self.track_history[tid][-1][2]) > 1.0:
                    del self.track_history[tid]
                continue
            points = self.track_history[tid]
            if len(points) >= 2:
                dt = points[-1][2] - points[0][2]
                if dt > 0.2: 
                    speed = np.sqrt((points[-1][0]-points[0][0])**2 + (points[-1][1]-points[0][1])**2) / dt 
                    total_speed += speed
                    valid_speed_count += 1

        avg_speed = total_speed / valid_speed_count if valid_speed_count > 0 else 0.0

        if self.vehicle_count > self.congestion_threshold or self.people_count > self.crowd_threshold:
            if valid_speed_count > 0 and avg_speed < self.speed_threshold:
                status_text, status_color = "Trang thai: TAC NGHEN!", (0, 0, 255) 
            else:
                status_text, status_color = "Trang thai: Dong duc", (0, 165, 255) 
        else:
            status_text, status_color = "Trang thai: Thong thoang", (0, 255, 0)
            
        return avg_speed, status_text, status_color

    def draw_status(self, frame, avg_speed, status_text, status_color):
        cv2.putText(frame, f"Vehicles: {self.vehicle_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Avg Speed: {int(avg_speed)} px/s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, status_text, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
