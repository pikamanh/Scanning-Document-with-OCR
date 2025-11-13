from PIL import Image
import cv2

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class OCR:
    def __init__(self, model_name="vgg_transformer", device="cuda"):
        config = Cfg.load_config_from_name(model_name)
        config['device'] = device

        self.detector = Predictor(config=config)
        self.results = {}
        self.img = None
        self.rois = {
            "id_number":        (270, 340, 250, 490),
            "name":             (360, 416, 170, 470),
            "dob":              (410, 450, 370, 500),
            "gender":           (450, 490, 310, 360),
            "national":         (450, 485, 520, 620),
            "place_orgin":      (515, 575, 180, 630),
            "place_of_residence1": (565, 605, 435, 620),
            "place_of_residence2": (600, 640, 185, 620)
        }

    def predict(self, img):
        self.img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        for field, (y1, y2, x1, x2) in self.rois.items():
            roi_img = self.img[y1:y2, x1:x2]
            roi_img = Image.fromarray(roi_img)
            text = self.detector(roi_img)
            self.results[field] = text

        return self.results