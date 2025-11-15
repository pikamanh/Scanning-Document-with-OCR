from ultralytics import YOLO
import torch
import cv2

from pre_proccessing import ProccessingImage
from OCR import OCR

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = YOLO(r"model\best_detection_cccd.pt")
    cam = cv2.VideoCapture(r"rtsp://192.168.1.9:8080/h264.sdp")
    proccessingImage = ProccessingImage()
    ocr = OCR()
    # img_path = "temp/temp.jpg"

    frame_count = 0
    skip = 30

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % skip == 0:
            results = model.predict(frame, device=device, conf=0.8)

            if not results or len(results[0].boxes) == 0:
                print("Cannot detection.")
                continue
            try:
                for result in results:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    top_x = int(xyxy[0][0])
                    top_y = int(xyxy[0][1])
                    bottom_x = int(xyxy[0][2])
                    bottom_y = int(xyxy[0][3])

                    fix_top_x, fix_bottom_x, fix_top_y, fix_bottom_y = top_x - 50, bottom_x + 50, top_y - 50, bottom_y + 50

                    if all(v >= 0 for v in [fix_top_x, fix_bottom_x, fix_top_y, fix_bottom_y]):
                        cropped_image = frame[fix_top_y:fix_bottom_y, fix_top_x:fix_bottom_x]    
                    else:
                        cropped_image = frame[top_y:bottom_y, top_x:bottom_x]    

                    # Xoay ảnh 90 độ theo chiều kim đồng hồ
                    cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(f"temp/temp{frame_count}.jpg", cropped_image)
                # print("Finish Detection.")
                # #Align Image
                # image_proccessed = alignImage.allign(img_path)
                # print("Finish processing.")

                # #Get OCR
                # information = ocr.predict(image_proccessed)
                # print("Finish get information.")

                # print("Information:\n")
                # for field, value in information.items():
                #     print(f"{field}: {value}\n")
            except Exception as e:
                print(f"Error: {e}")

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()