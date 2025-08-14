import os
from ultralytics import YOLO

# Choose model via env; default is yolov8n
MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
ENGINE = os.getenv("YOLO_ENGINE", "yolov8n.engine")

def main():
    print(f"[export] loading {MODEL}")
    m = YOLO(MODEL)
    print(f"[export] exporting TensorRT engine -> {ENGINE} (FP16, 640)")
    m.export(format="engine", half=True, device=0, imgsz=640, dynamic=False)
    # Move/rename to shared volume so alert/preview can find it
    import os, shutil
    os.makedirs("/workspace/work", exist_ok=True)
    src = os.path.splitext(MODEL)[0] + ".engine"
    dst = os.path.join("/workspace/work", ENGINE)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"[export] saved: {dst}")
    else:
        print(f"[export] expected not found: {src}")

if __name__ == "__main__":
    main()
