# Optional: If you want explicit tracking separate from Ultralytics, implement here.
# Current Ultralytics YOLO can already return .id when using tracker config.
# This stub shows protocol compliance; replace with actual tracker as needed.
from typing import Sequence
from ..core.ports import ITracker, Detection, Frame

class PassthroughTracker(ITracker):
    def update(self, frame: Frame, dets: Sequence[Detection]) -> Sequence[Detection]:
        return dets
