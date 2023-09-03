from typing import Any, Callable
import importlib
import os
import time
import urllib.request
import numpy as np

try:
    import cv2
except Exception as exc:
    raise ImportError(
        "OpenCV (cv2) is required for HandTracker. Install with `pip install opencv-python`") from exc

try:
    import mediapipe as mp
except Exception:
    mp = None

from config import MEDIAPIPE

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def _ensure_hand_model(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """Download the MediaPipe hand landmark model if it is not already present."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


Point = tuple[int, int]
LandmarkList = list[Point]
GestureDict = dict[str, bool]
SmootherCallable = Callable[[float, float], tuple[float, float]]


# MediaPipe hand tracking and gesture detection pipeline.
class HandTracker:

    def __init__(self, smoother: SmootherCallable | None = None) -> None:
        self._smoother: SmootherCallable | None = smoother
        self._mp_hands: Any = None
        self._mp_drawing: Any = None
        self._hands: Any = None
        self._landmarker: Any = None
        self._task_image_class: Any = None
        self._task_image_format: Any = None
        self._backend_error: str | None = None

        if mp is None:
            self._backend_error = "MediaPipe is not installed or importable."
            print(f"[WARN] {self._backend_error}")
            return

        if hasattr(mp, "solutions"):
            try:
                # Access the legacy solutions API dynamically — mediapipe ships
                # no type stubs for it, so static analysis cannot see it.
                solutions = getattr(mp, "solutions", None)
                if solutions is not None:
                    self._mp_hands = getattr(solutions, "hands", None)
                    self._mp_drawing = getattr(
                        solutions, "drawing_utils", None)
                    self._hands = self._mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=MEDIAPIPE.max_num_hands,
                        min_detection_confidence=MEDIAPIPE.min_detection_confidence,
                        min_tracking_confidence=MEDIAPIPE.min_tracking_confidence,
                    )
            except AttributeError:
                self._backend_error = (
                    "MediaPipe legacy 'solutions' API is not available in this environment. "
                    "Falling back to the current Tasks API."
                )
                print(f"[WARN] {self._backend_error}")

        if self._hands is None and not hasattr(mp, "solutions"):
            try:
                vision = importlib.import_module(
                    "mediapipe.tasks.python.vision")
                hand_landmarker_module = importlib.import_module(
                    "mediapipe.tasks.python.vision.hand_landmarker")
                image_module = importlib.import_module(
                    "mediapipe.tasks.python.vision.core.image")
                python_module = importlib.import_module(
                    "mediapipe.tasks.python")
                model_path = _ensure_hand_model()
                options = hand_landmarker_module.HandLandmarkerOptions(
                    base_options=python_module.BaseOptions(
                        model_asset_path=model_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=MEDIAPIPE.max_num_hands,
                    min_hand_detection_confidence=MEDIAPIPE.min_detection_confidence,
                    min_hand_presence_confidence=MEDIAPIPE.min_tracking_confidence,
                    min_tracking_confidence=MEDIAPIPE.min_tracking_confidence,
                )
                self._landmarker = hand_landmarker_module.HandLandmarker.create_from_options(
                    options)
                self._task_image_class = image_module.Image
                self._task_image_format = image_module.ImageFormat
            except Exception as exc:
                self._backend_error = f"MediaPipe hand detection backend could not initialize: {exc}"
                print(f"[WARN] {self._backend_error}")

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            finally:
                self._hands = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            finally:
                self._landmarker = None

    def _normalized_to_pixel(self, nx: float, ny: float, frame_shape: tuple[int, int, int]) -> Point:
        # Convert normalized MediaPipe coordinates to pixel coordinates
        h, w = frame_shape[0], frame_shape[1]
        x = int(min(max(nx * w, 0), w - 1))
        y = int(min(max(ny * h, 0), h - 1))
        return x, y

    def _fingers_up(self, landmarks: list[Point]) -> dict[str, bool]:
        # Determine extended fingers from landmark positions
        fingers = {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
        }

        tip_ids = {"thumb": 4, "index": 8,
                   "middle": 12, "ring": 16, "pinky": 20}
        pip_ids = {"thumb": 2, "index": 6,
                   "middle": 10, "ring": 14, "pinky": 18}

        try:
            for finger in ("index", "middle", "ring", "pinky"):
                tip_y = landmarks[tip_ids[finger]][1]
                pip_y = landmarks[pip_ids[finger]][1]
                fingers[finger] = tip_y < pip_y

            wrist_x = landmarks[0][0]
            thumb_tip_x = landmarks[tip_ids["thumb"]][0]
            fingers["thumb"] = abs(thumb_tip_x - wrist_x) > 30
        except Exception:
            return {key: False for key in fingers.keys()}

        return fingers

    def _classify_gestures(self, fingers: dict[str, bool]) -> GestureDict:
        # Classify gesture flags from extended-finger states.
        # Thumb is intentionally ignored — its position is unreliable when the
        # palm faces the camera, which previously blocked the open-hand erase.
        up = sum(1 for finger in ("index", "middle", "ring", "pinky")
                 if fingers.get(finger, False))
        return {
            "index_up": up == 1 and fingers.get("index", False),
            "selection": up == 2 and fingers.get("index", False) and fingers.get("middle", False),
            "erase": up >= 3,
            "fist": up == 0,
        }

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        # Process a single BGR frame and return landmarks and gesture flags.
        if frame is None:
            raise ValueError("Input frame is None")

        empty_result = {
            "landmarks": None,
            "handedness": None,
            "gestures": {"index_up": False, "selection": False, "erase": False, "fist": False},
            "raw_landmarks": None,
        }

        if self._hands is None and self._landmarker is None:
            return empty_result

        if self._landmarker is not None and self._task_image_class is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._task_image_class(
                image_format=self._task_image_format.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(
                mp_image, int(time.time_ns() / 1_000_000))
            if not result.hand_landmarks:
                return empty_result

            hand_landmarks = result.hand_landmarks[0]
            handedness_label = None
            if getattr(result, "handedness", None):
                first_handedness = result.handedness[0]
                if first_handedness:
                    handedness_label = getattr(
                        first_handedness[0], "category_name", None)

            raw_pixel_landmarks: LandmarkList = []
            pixel_landmarks: LandmarkList = []
            for lm in hand_landmarks:
                px, py = self._normalized_to_pixel(lm.x, lm.y, frame.shape)
                raw_pixel_landmarks.append((px, py))
                if self._smoother is not None:
                    sx, sy = self._smoother(float(px), float(py))
                    pixel_landmarks.append((int(sx), int(sy)))
                else:
                    pixel_landmarks.append((px, py))

            fingers = self._fingers_up(pixel_landmarks)
            gestures: GestureDict = self._classify_gestures(fingers)
            return {
                "landmarks": pixel_landmarks,
                "handedness": handedness_label,
                "gestures": gestures,
                "raw_landmarks": raw_pixel_landmarks,
            }

        if self._hands is None:
            return empty_result

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return empty_result

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_label = None
        if results.multi_handedness:
            try:
                handedness_label = results.multi_handedness[0].classification[0].label
            except Exception:
                handedness_label = None

        raw_pixel_landmarks = []
        pixel_landmarks = []
        for lm in hand_landmarks.landmark:
            px, py = self._normalized_to_pixel(lm.x, lm.y, frame.shape)
            raw_pixel_landmarks.append((px, py))
            if self._smoother is not None:
                sx, sy = self._smoother(float(px), float(py))
                pixel_landmarks.append((int(sx), int(sy)))
            else:
                pixel_landmarks.append((px, py))

        fingers = self._fingers_up(pixel_landmarks)
        gestures = self._classify_gestures(fingers)

        return {
            "landmarks": pixel_landmarks,
            "handedness": handedness_label,
            "gestures": gestures,
            "raw_landmarks": raw_pixel_landmarks,
        }

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()
