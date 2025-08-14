from ultralytics import YOLO
import numpy as np
from ..core.ports import Detector, Detection, Frame

class UltralyticsDetector(Detector):
    def __init__(self, engine_path: str, conf: float, imgsz: int, vid_stride: int):
        self.model = YOLO(engine_path, task="detect")
        self.conf = conf
        self.imgsz = imgsz
        self.vid_stride = vid_stride

    def detect(self, frame: Frame):
        r = self.model.predict(
            source=frame.image, imgsz=self.imgsz, conf=self.conf,
            vid_stride=self.vid_stride, device=0, verbose=False
        )[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy()
        ids  = boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None else None
        dets = []
        for i in range(len(xyxy)):
            x1,y1,x2,y2 = [int(v) for v in xyxy[i]]
            dets.append(Detection(
                xyxy=(x1,y1,x2,y2), conf=float(conf[i]), cls_id=int(cls[i]),
                track_id=int(ids[i]) if ids is not None else None
            ))
        return dets
