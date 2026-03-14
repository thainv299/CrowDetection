from ultralytics import YOLO

model = YOLO("yolo26m.pt")

results = model.train(
    data="E:/DATN_code/Dataset/COCO_Balanced/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    project="Traffic_AI",
    name="Medium_Run1",

    lr0=0.001,           # LR thấp vì đang finetune
    lrf=0.01,
    warmup_epochs=3,
    weight_decay=0.0005,

    # Augmentation
    mosaic=1.0,
    close_mosaic=10,     # Tắt mosaic 10 epoch cuối
    mixup=0.1,
    degrees=10.0,
    hsv_s=0.5,
    hsv_v=0.3,           # Giảm từ 0.4 để tránh ảnh ban đêm bị đen
    fliplr=0.5,
    flipud=0.0,
)