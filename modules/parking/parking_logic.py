from __future__ import annotations
import math
from typing import Dict, Tuple

# ── Trạng thái xe ────────────────────────────────────────────────────────────
MOVING  = "moving"   # đang chạy  – dịch chuyển > nguong_di_chuyen
STOPPED = "stopped"  # đang dừng  – đứng yên nhưng < nguong_do_xe
PARKED  = "parked"   # đỗ xe      – đứng yên >= nguong_do_xe → vi phạm


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ViolationLogic:
    """
    Quản lý trạng thái từng xe theo track_id:
      MOVING  →  STOPPED  →  PARKED
    Thời gian đứng yên tính theo SỐ FRAME (không dùng time.time())
    để đảm bảo chính xác bất kể tốc độ xử lý của máy.
    """

    def __init__(
        self,
        stop_seconds: float,
        move_thr_px: float,
        cooldown_seconds: float,
        fps: float = 21.0,
    ):
        self.move_thr_px  = float(move_thr_px)

        # Đổi ngưỡng giây → số frame
        self.nguong_do_xe_frame    = int(stop_seconds    * fps)
        self.nguong_cooldown_frame = int(cooldown_seconds * fps)
        self.fps                   = float(fps)

        # track_id → dict trạng thái
        self.trang_thai: Dict[int, Dict] = {}

    # ─────────────────────────────────────────────────────────────────────────
    def update(
        self,
        track_id: int,
        center: Tuple[float, float],
        so_frame: int,
    ) -> float:
        """
        Cập nhật vị trí xe theo frame hiện tại.

        Parameters
        ----------
        track_id : int   – ID xe từ tracker
        center   : tuple – tâm bbox (x, y)
        so_frame : int   – số thứ tự frame hiện tại

        Returns
        -------
        thoi_gian_dung_yen : float – số giây xe đứng yên (quy từ frame)
        """

        # ── Lần đầu gặp xe này ──────────────────────────────────────────────
        if track_id not in self.trang_thai:
            self.trang_thai[track_id] = {
                "vi_tri_cuoi":          center,
                "dung_tu_frame":        so_frame,
                "last_violation_frame": 0,
                "vehicle_state":        MOVING,
            }
            return 0.0

        # ── Tính khoảng cách tâm bbox 2 frame liên tiếp ─────────────────────
        vi_tri_cuoi = self.trang_thai[track_id]["vi_tri_cuoi"]
        khoang_cach = dist(center, vi_tri_cuoi)

        if khoang_cach > self.move_thr_px:
            # Xe di chuyển → reset bộ đếm frame đứng yên
            self.trang_thai[track_id]["dung_tu_frame"] = so_frame
            self.trang_thai[track_id]["vehicle_state"] = MOVING

        self.trang_thai[track_id]["vi_tri_cuoi"] = center

        # ── Số frame đứng yên → đổi ra giây ─────────────────────────────────
        so_frame_dung_yen  = so_frame - self.trang_thai[track_id]["dung_tu_frame"]
        thoi_gian_dung_yen = so_frame_dung_yen / self.fps

        # ── Cập nhật trạng thái ──────────────────────────────────────────────
        if khoang_cach > self.move_thr_px:
            self.trang_thai[track_id]["vehicle_state"] = MOVING
        elif so_frame_dung_yen < self.nguong_do_xe_frame:
            self.trang_thai[track_id]["vehicle_state"] = STOPPED
        else:
            self.trang_thai[track_id]["vehicle_state"] = PARKED

        return thoi_gian_dung_yen

    # ─────────────────────────────────────────────────────────────────────────
    def should_flag_violation(
        self,
        track_id: int,
        so_frame: int,
        in_no_park: bool,
    ) -> bool:
        """
        Trả về True chỉ khi:
          1. Xe nằm trong vùng cấm đỗ.
          2. Xe đã ở trạng thái PARKED.
          3. Đã hết cooldown (tính theo frame).
        """
        if not in_no_park:
            return False

        if self.trang_thai.get(track_id, {}).get("vehicle_state") != PARKED:
            return False

        last_vio_frame = self.trang_thai.get(track_id, {}).get("last_violation_frame", 0)
        if so_frame - last_vio_frame < self.nguong_cooldown_frame:
            return False

        self.trang_thai[track_id]["last_violation_frame"] = so_frame
        return True

    # ─────────────────────────────────────────────────────────────────────────
    def get_vehicle_state(self, track_id: int) -> str:
        """Trả về trạng thái hiện tại: 'moving' | 'stopped' | 'parked'."""
        return self.trang_thai.get(track_id, {}).get("vehicle_state", MOVING)