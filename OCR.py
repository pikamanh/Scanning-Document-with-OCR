from PIL import Image
import cv2
import matplotlib.pyplot as plt

from pre_proccessing import ProccessingImage
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
            "id_number":        (275, 326, 230, 480),
            "name":             (357, 405, 171, 640),
            "dob":              (405, 439, 358, 483),
            "gender":           (440, 478, 292, 345),
            "national":         (430, 478, 502, 609),
            "place_orgin":      (512, 553, 171, 640),
            "place_of_residence1": (551, 592, 423, 620),
            "place_of_residence2": (588, 630, 171, 620)
        }

    def predict(self, img):
        self.img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        # for field, (y1, y2, x1, x2) in self.rois.items():
        #     roi_img = self.img[y1:y2, x1:x2]
        #     roi_img = Image.fromarray(roi_img)
        #     text = self.detector(roi_img)
        #     self.results[field] = text

        return self.img
    
if __name__ == "__main__":
    proccessingImage = ProccessingImage()
    ocr = OCR()
    cropped_image = proccessingImage.focus_image(r"temp/temp17.jpg")
    test = ocr.predict(cropped_image)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(test)
    axs[1].imshow(test[588:630, 171:620])
    plt.tight_layout()
    plt.show()