import cv2
import numpy as np
from sklearn.cluster import KMeans

class AllignImage:
    def __init__(self):
        self.img = None

    def convert_to_canny(self, threshold_min_canny=10, threshold_max_canny=150):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        canny = cv2.Canny(img_blur, threshold_min_canny, threshold_max_canny, L2gradient=True)
        return canny
    
    def intersection(self, line1, line2):
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)  # Giải hệ phương trình tuyến tính
        return int(np.round(x0)), int(np.round(y0))
    
    def get_points_intersections(self, img_canny):
        horizontal, vertical, intersections = [], [], []
        lines = cv2.HoughLines(img_canny, 1, np.pi/180, 100)
    
        for line in lines:
            rho, theta = line[0]
            if theta < np.pi / 4 or theta > 3 * np.pi / 4:
                vertical.append(line)
            else:
                horizontal.append(line)

        for v in vertical:
            for h in horizontal:
                pt = self.intersection(v, h)
                if pt is not None:
                    intersections.append(pt)

        return intersections
    
    def get_corners(self, intersections):
        points = np.array(intersections, dtype=np.float32)
        kmeans = KMeans(4, random_state=0).fit(points)
        center = kmeans.cluster_centers_

        s = center.sum(axis=1)
        diff = np.diff(center, axis=1)
        return np.array([
            center[np.argmin(s)],      # top-left
            center[np.argmin(diff)],   # top-right
            center[np.argmax(s)],      # bottom-right
            center[np.argmax(diff)]    # bottom-left
        ], dtype=np.float32)
    
    def allign(self, img_path):
        self.img = img_path
        img_canny = self.convert_to_canny()
        intersections = self.get_points_intersections(img_canny)
        corners = self.get_corners(intersections)

        (tl, tr, br, bl) = corners

        # Chiều rộng (max giữa cạnh trên và dưới)
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        # Chiều cao (max giữa cạnh trái và phải)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        img_rotate = self.img.copy()

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Tính ma trận transform
        M = cv2.getPerspectiveTransform(corners, dst)

        # Áp dụng warp để xoay lại ảnh ID card
        warped = cv2.warpPerspective(img_rotate, M, (maxWidth, maxHeight))

        return warped