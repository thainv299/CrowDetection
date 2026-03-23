# Walkthrough: Illegal Parking Detection Integration

## What Was Accomplished
The "Illegal Parking Detection" rules originally implemented inside the `src/` directory have been fully integrated into [main.py](file:///e:/DATN_code/main.py)!

Here are the key enhancements added to your application:
**1. [ViolationLogic](file:///e:/DATN_code/src/logic.py#15-126) Tracking**
- Reused `src.logic.ViolationLogic` to track the state of moving bounding boxes (`"MOVING"`, `"STOPPED"`, `"PARKED"`) within [main.py](file:///e:/DATN_code/main.py)'s [detect_video](file:///e:/DATN_code/main.py#177-314) function.
- Implemented state-based color indicators for the vehicle bounding boxes:
  - **DUNG (Stopped but within grace limit):** Orange outline.
  - **DO XE (Parked illegally over grace limit):** Red outline.
  - Tracking text now directly includes the time the vehicle has been stationary (e.g. `12.5s`).

**2. Violation Recording (Images & Video Context)**
- Imported `deque` and initialized a 5-second `frame_buffer` to continuously store the past 5 seconds of video frames.
- Re-implemented [save_violation_video()](file:///e:/DATN_code/src/main.py#21-46) inside [main.py](file:///e:/DATN_code/main.py)'s App class. Upon violation, it slices these 5 seconds of historical frames into `outputs/violations/violation_123_YYYY...mp4`.
- Extracted and saved the exact violation frame to `outputs/violations/violation_123_YYYY...jpg`.

**3. Application Architecture Refactor**
1. **[parking_manager.py](file:///e:/DATN_code/parking_manager.py)**: Isolated all the illegal stopping UI buttons, layout bindings, tracking state ([ViolationLogic](file:///e:/DATN_code/src/logic.py#15-126)), and Telegram notifications to completely detach it from the main code. 
2. **[ocr_manager.py](file:///e:/DATN_code/ocr_manager.py)**: Isolated the entire Optical Character Recognition threaded worker (`PaddleOCR`), queuing logic, spatial tracking matrix, regex caching, and image voting sequence.
3. **[traffic_monitor.py](file:///e:/DATN_code/traffic_monitor.py)**: Moved calculating distances between points, speeds, thresholds, and drawing `TAC NGHEN / THONG THOANG` traffic statuses into an external manager.
- *Result:* [main.py](file:///e:/DATN_code/main.py) size shrank tremendously making the central video tracking loop highly scannable and easy to further customize!

**4. Telegram Bot Compatibility**
- Reused your `src.telegram_bot` methods internally within [main.py](file:///e:/DATN_code/main.py).
- Added new parameters `self.telegram_enabled`, `self.telegram_bot_token`, and `self.telegram_chat_id`. 
- *(Note: Currently `telegram_enabled = False` by default. You can change this switch to `True` within [main.py](file:///e:/DATN_code/main.py) inside `App.__init__` and add your bot token credentials to instantly receive notifications).*

## How to Test and Verify
Since this is a GUI application (`tkinter`), automated CLI tests are not feasible. Please manually test the flow:
1. Run `python main.py` in your terminal.
2. Choose your YOLO model.
3. Import a test video (ensure there's a vehicle stopping/parking for at least 5 seconds within the view).
4. Outline your ROI (Vẽ Vùng Quan Sát).
5. Press "Bắt đầu Detect" and monitor the UI.
6. The target vehicles should correctly increment their stationary time, transition to **DUNG**, and subsequently to **DO XE** when exceeding 5 seconds.
7. Confirm that a large red banner `VI PHAM: DO XE SAI QUY DINH!` appears.
8. Validate that both an image `.jpg` and a 5-second `.mp4` video clip have been strictly generated inside the `outputs/violations/` folder in your project root!
