from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = YOLO(r"runs\pose\train3\weights\best.pt")
    image = r'dataset\valid\images\JSZZ1394802772779196_jpg.rf.411e87c37bd7003efab6a8b7a6e28092.jpg'

    results = model(image, device=device)

    for result in results:
        xy = result.keypoints.xy
        xyn = result.keypoints.xyn
        kpts = result.keypoints.data

        print(f"xy: {xy}")
        print(f"xyn: {xyn}")
        print(f"kpts: {kpts}")

if __name__ == "__main__":
    main()