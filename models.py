from typing import Tuple
import numpy as np
from ultralytics import YOLO

class YOLOMODEL:
    def __init__(self, weights_path: str,  device: str = 'cpu'):
        model = YOLO(weights_path)

    def yolo2bbox(self, bboxes) -> Tuple:
        xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
        xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
        return xmin, ymin, xmax, ymax



    def predict(self, image: np.array) -> Tuple:
        results = self.model.predict(image)

        return selected
