import cv2

def main():
    image = cv2.imread(r"test 2.jpg")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3))

    cv2.imshow("Test", img_blur)

if __name__ == "__main__":
    main()