from typing import Tuple
import numpy as np
from ultralytics import YOLO

class YOLOMODEL:
    def __init__(self, weights_path: str,  device: str = 'cpu'):
        model = YOLO(weights_path)
    def predict(self, image: np.array) -> Tuple:
        results = self.model.predict(image)
        output = []
        for r in results:
            boxes = r.boxes.xywh.numpy()
            cls = r.boxes.cls.numpy()
            scores = r.boxes.conf.numpy()
            label_dict = r.names
            for i in range(boxes.shape[0]):
                x, y, w, h = boxes[i]
                xmin, ymin = x - w / 2, y - h / 2
                xmax, ymax = x + w / 2, y + h / 2
                label = label_dict[cls[i]]
                score = scores[i]
                if label == 'window':
                    label = 1
                elif label == 'empty':
                    label = 2
                elif label == 'filled':
                    label = 3
                output_format = f"{label} {score} {xmin} {ymin} {xmax} {ymax}"
                output.append(output_format)
        return output

