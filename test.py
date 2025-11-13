from ultralytics import YOLO

model = YOLO(r"model\best_detection_cccd.pt")

results = model(r"temp\temp.jpg", device="cuda")
for result in results:
    result.show()