import numpy as np
from src.core.hand_tracker import HandTracker
from src.utils.smoothing import ExponentialMovingAverageSmoother


def test_handtracker_no_hand():
    tracker = HandTracker(smoother=ExponentialMovingAverageSmoother(alpha=0.9))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(frame)
    assert result["landmarks"] is None
    assert isinstance(result["gestures"], dict)
    tracker.close()
