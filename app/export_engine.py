import os
from ultralytics import YOLO

# Choose model via env; default is yolov8n
MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
ENGINE = os.getenv("YOLO_ENGINE", "yolov8n.engine")

def main():
    print(f"[export] loading {MODEL}")
    m = YOLO(MODEL)  # auto-downloads if not present
    print(f"[export] exporting TensorRT engine -> {ENGINE} (FP16)")
    m.export(format="engine", half=True, device=0, imgsz=640, dynamic=False)
    # Ultralytics writes engine in CWD; rename if needed
    # but by default it will be {stem}.engine
    print("[export] done")

if __name__ == "__main__":
    main()
