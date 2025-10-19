from ultralytics import YOLO
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load a model
    model = YOLO("yolov8n-pose.yaml")  # build a new model from YAML
    model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="dataset\data.yaml", epochs=100, imgsz=640, device=device)