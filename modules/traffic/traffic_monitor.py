import numpy as np
import cv2

# --- CONGESTION THRESHOLDS ---
CONG_COUNT_THR = 10              # Level 1: Min vehicles to be considered "Crowded L1"
CONG_PEOPLE_THR = 20             # Level 1: Min people to be considered "Crowded L1"
CONG_AREA_PERCENT_THR = 40.0     # Level 2: Min ROI area % covered by vehicles to be "Crowded L2"
CONG_SPEED_THR = 10.0            # Level 3: Max speed (px/s) to be considered "Congested"

class TrafficMonitor:
    def __init__(self, roi_polygon=None):
        self.roi_polygon = roi_polygon
        self.roi_area = cv2.contourArea(np.array(self.roi_polygon)) if self.roi_polygon is not None else 0.0
        
        self.track_history = {}
        self.vehicle_count = 0
        self.people_count = 0
        self.current_ids_in_roi = []
        self.total_vehicle_area = 0.0
        
    def reset_counters(self):
        self.vehicle_count = 0
        self.people_count = 0
        self.current_ids_in_roi = []
        self.total_vehicle_area = 0.0

    def log_person(self):
        self.people_count += 1

    def log_vehicle(self, track_id, cx, cy, current_time, area=0.0):
        self.vehicle_count += 1
        self.total_vehicle_area += area
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

        occupancy_percent = min(100.0, (self.total_vehicle_area / self.roi_area) * 100.0) if self.roi_area > 0 else 0.0

        is_high_count = (self.vehicle_count >= CONG_COUNT_THR) or (self.people_count >= CONG_PEOPLE_THR)

        if not is_high_count and occupancy_percent < CONG_AREA_PERCENT_THR:
            traffic_level = 0
            status_text, status_color = "Trang thai: Thong thoang (MUC 0)", (0, 255, 0)
        elif is_high_count and occupancy_percent < CONG_AREA_PERCENT_THR:
            traffic_level = 1
            if self.vehicle_count >= CONG_COUNT_THR:
                status_text, status_color = f"Trang thai: Dong duc (MUC 1) - {self.vehicle_count} xe", (0, 165, 255)
            else:
                status_text, status_color = f"Trang thai: Dong duc (MUC 1) - {self.people_count} nguoi", (0, 165, 255)
        elif occupancy_percent >= CONG_AREA_PERCENT_THR and avg_speed > CONG_SPEED_THR:
            traffic_level = 2
            status_text, status_color = f"Trang thai: Rat dong (MUC 2) - {occupancy_percent:.1f}% dien tich", (0, 100, 255)
        elif occupancy_percent >= CONG_AREA_PERCENT_THR and avg_speed <= CONG_SPEED_THR:
            traffic_level = 3
            status_text, status_color = f"Trang thai: TAC NGHEN (MUC 3) - {avg_speed:.1f} px/s", (0, 0, 255)
        else:
            traffic_level = 0
            status_text, status_color = "Trang thai: Thong thoang (MUC 0)", (0, 255, 0)
            
        return avg_speed, status_text, status_color, traffic_level

    def draw_status(self, frame, avg_speed, status_text, status_color):
        occupancy_percent = min(100.0, (self.total_vehicle_area / self.roi_area) * 100.0) if self.roi_area > 0 else 0.0
        cv2.putText(frame, f"Vehicles: {self.vehicle_count} | People: {self.people_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Occupancy: {occupancy_percent:.1f}%", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Avg Speed: {int(avg_speed)} px/s", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, status_text, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
