from typing import Tuple
import numpy as np
from ultralytics import YOLO

class YOLOMODEL:
    def __init__(self, weights_path: str, task: str):
        self.model = YOLO(model=weights_path, task=task)

    def predict(self, image: np.array) -> Tuple:
        results = self.model.predict(image)
        bboxes, labels, scores = [], [], []
        for r in results:
            boxes = r.boxes.xywh.numpy()
            cls = r.boxes.cls.numpy()
            scores_ = r.boxes.conf.numpy()
            label_dict = r.names
            for i in range(boxes.shape[0]):
                x, y, w, h = boxes[i]
                xmin, ymin = x - w / 2, y - h / 2
                xmax, ymax = x + w / 2, y + h / 2
                label = label_dict[cls[i]]
                score = scores_[i]
                bbox = (xmin, ymin, xmax, ymax)
                if label == 'window':
                    label = 1
                elif label == 'empty':
                    label = 2
                elif label == 'filled':
                    label = 3
                bboxes.append(bbox)
                labels.append(label)
                scores.append(score)
        return bboxes, labels, scores
