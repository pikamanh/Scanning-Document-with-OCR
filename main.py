from ultralytics import YOLO
import torch
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = YOLO(r"model\best v1.pt")
    image = cv2.imread(r'test.jpg')
    # image = cv2.imread(r"datasets\ID Card (VN) v1\train\images\00_png_jpg.rf.2cbb14a0813594ec3831406d94bc9732.jpg")
    image = cv2.resize(image, (640, 640))
    image_copy = image.copy()

    results = model.predict(image, device=device, conf=0.8)

    try:
        for result in results:
            kpts = result.keypoints.data.cpu().numpy()
            print(kpts.shape)

            for kpt in kpts:
                for x, y, conf in kpt:
                    cv2.circle(image_copy, (int (x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

            # img_result = cv2.drawKeypoints(image=image.copy(), keypoints=keypoints, outImage=0, color=(0, 255, 0))
            cv2.imshow("test", image_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Có thể không detect được.")
        print(e)

if __name__ == "__main__":
    main()