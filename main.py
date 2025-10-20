from ultralytics import YOLO
import torch
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = YOLO(r"model\best.pt")
    cam = cv2.VideoCapture(0)

    frame_count = 0
    skip = 30

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % skip == 0:
            results = model.predict(frame, device=device, conf=0.8)

            if all(result.probs is None for result in results):
                print("Cannot detection.")
                continue
            try:
                for result in results:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    top_x = int(xyxy[0][0])
                    top_y = int(xyxy[0][1])
                    bottom_x = int(xyxy[0][2])
                    bottom_y = int(xyxy[0][3])

                    fix_top_x, fix_bottom_x, fix_top_y, fix_bottom_y = top_x - 10, bottom_x + 10, top_y - 10, bottom_y + 10

                    if all(v >= 0 for v in [fix_top_x, fix_bottom_x, fix_top_y, fix_bottom_y]):
                        cropped_image = frame[fix_top_y:fix_bottom_y, fix_top_x:fix_bottom_x]    
                    else:
                        cropped_image = frame[top_y:bottom_y, top_x:bottom_x]    

                    cv2.imwrite("temp/temp.jpg", cropped_image)
            except Exception as e:
                print(f"Error: {e}")

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()