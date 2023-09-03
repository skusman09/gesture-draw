# Centralized configuration for Air Canvas

from typing import Tuple, List, Dict
from dataclasses import dataclass

# Color represented as BGR tuples (OpenCV uses BGR)
Color = Tuple[int, int, int]

HEADER_HEIGHT = 100
BUTTON_W = 80
BUTTON_H = 40
GAP_SMALL = 10
GAP_LARGE = 40
PALETTE_SWATCH_SIZE = 40
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 40
ERASER_RADIUS = 25
FIST_CLEAR_DEBOUNCE_FRAMES = 18
LOST_TRACKING_RESET_FRAMES = 5
TEMPLATE_BUTTON_W = 120
TEMPLATE_BUTTON_GAP = 18
TEMPLATE_CLOSE_PADDING = 10
TEMPLATE_CLOSE_SIZE = 22


@dataclass(frozen=True)
class UIConfig:
    # UI layout, sizing, and styling constants

    window_name: str = "Air Canvas"
    topbar_height: int = HEADER_HEIGHT
    palette_position: Tuple[int, int] = (18, 18)
    palette_spacing: int = GAP_SMALL
    palette_item_size: int = PALETTE_SWATCH_SIZE
    font: int = 0
    font_scale: float = 0.6
    font_thickness: int = 1
    show_fps: bool = True


@dataclass(frozen=True)
class BrushConfig:
    # Brush geometry, palette defaults, and undo limits

    default_color: Color = (0, 0, 255)
    eraser_color: Color = (255, 255, 255)
    min_brush_size: int = 2
    max_brush_size: int = 50
    default_brush_size: int = 12
    max_undo: int = 20


# Vertical slider rail geometry (left edge, below header)
SLIDER_X = 30
SLIDER_Y = 150
SLIDER_H = 300
SLIDER_W = 10


@dataclass(frozen=True)
class CameraConfig:
    # Camera device capture parameters

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    mirror: bool = True


@dataclass(frozen=True)
class MediaPipeConfig:
    # MediaPipe hand tracking thresholds

    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6


PALETTE_COLORS: Dict[str, Color] = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "black": (10, 10, 10),
    "eraser": (255, 255, 255),
}

PALETTE: List[Color] = list(PALETTE_COLORS.values())

UI = UIConfig()
BRUSH = BrushConfig()
CAMERA = CameraConfig()
MEDIAPIPE = MediaPipeConfig()
