# Thin wrapper to swap-in later if you build a custom TensorRT runtime detector.
# For now, reuse Ultralytics engine loader so Detector is interchangeable.
from .detector_ultra import UltralyticsDetector as TensorRTDetector
