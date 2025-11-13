import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ProccessingImage:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.height, self.width = self.img.shape[:2]
        self.center = (self.width//2, self.height//2)


    def canny(self, min_thresh=50, max_thresh=200):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        canny_img = cv2.Canny(gray_img, min_thresh, max_thresh)

        return canny_img
    
    def get_lines(self, img_canny, threshold=180):
        lines = cv2.HoughLines(img_canny, 1, np.pi/180, threshold)

        return lines

    def line_intersection(self, line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])
        x0, y0 = np.linalg.lstsq(A, b, rcond=None)[0]
        return [int(np.round(x0)), int(np.round(y0))]

    def rotate_image(self):
        canny = self.canny()
        lines = self.get_lines(canny)
        min_horizontal_rad = np.deg2rad(45)
        max_horizontal_rad = np.deg2rad(135)

        for line in lines:
            rho, theta = line[0]
            if min_horizontal_rad < theta < max_horizontal_rad:
                horizontal_theta = theta
                break

        angle_rad = horizontal_theta - (np.pi/2)
        angle_deg = np.rad2deg(angle_rad)
        M = cv2.getRotationMatrix2D(center=self.center, angle=angle_deg, scale=1)

        rotated_image = cv2.warpAffine(
            src=self.img.copy(),
            M=M, 
            dsize=(self.width, self.height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated_image
    
    def focus_image(self):
        horizontal_lines = []
        vertical_lines = []

        canny = self.canny()
        lines = self.get_lines(canny)

        