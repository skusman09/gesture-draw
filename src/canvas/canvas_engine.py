from typing import Tuple, List, Optional
import numpy as np
import cv2

from config import BRUSH

Color = Tuple[int, int, int]

# Off-screen RGBA canvas engine for stroke drawing, layer compositing, and undo/redo history
class CanvasEngine:

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._canvas = np.zeros((height, width, 4), dtype=np.uint8)
        self._template: Optional[np.ndarray] = None
        self._template_mask: Optional[np.ndarray] = None
        self._template_alpha: float = 0.0
        self._template_visible: bool = True
        self.undo_stack: List[np.ndarray] = []
        self.redo_stack: List[np.ndarray] = []
        self._current_color: Color = BRUSH.default_color
        self._current_size: int = BRUSH.default_brush_size
        self._tool_mode: str = "brush"
        self._is_drawing: bool = False
        self._last_point: Optional[Tuple[int, int]] = None
        self._line_start: Optional[Tuple[int, int]] = None

    def set_color(self, color: Color) -> None:
        self._current_color = color

    def set_tool_mode(self, mode: str) -> None:
        # Set active drawing tool ('brush' or 'line')
        if mode in {"brush", "line"}:
            self._tool_mode = mode

    def get_tool_mode(self) -> str:
        return self._tool_mode

    def set_brush_size(self, size: int) -> None:
        self._current_size = max(BRUSH.min_brush_size, min(BRUSH.max_brush_size, size))

    def get_brush_size(self) -> int:
        return self._current_size

    def _push_undo(self) -> None:
        # Save current canvas snapshot to undo stack
        if len(self.undo_stack) >= BRUSH.max_undo:
            self.undo_stack.pop(0)
        self.undo_stack.append(self._canvas.copy())
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self._canvas.copy())
        self._canvas = self.undo_stack.pop()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self._canvas.copy())
        self._canvas = self.redo_stack.pop()

    def start_stroke(self, point: Tuple[int, int]) -> None:
        self._is_drawing = True
        self._last_point = point
        self._line_start = point
        if self._tool_mode == "brush":
            self._draw_point(point)

    def update_stroke(self, point: Tuple[int, int]) -> None:
        if not self._is_drawing or self._last_point is None:
            self.start_stroke(point)
            return

        dx = point[0] - self._last_point[0]
        dy = point[1] - self._last_point[1]
        if dx * dx + dy * dy < 4:
            return

        if self._tool_mode == "line":
            self._last_point = point
            return

        steps = max(1, int(np.hypot(dx, dy) / 2.0))
        for i in range(1, steps + 1):
            t = i / steps
            x = int(round(self._last_point[0] + dx * t))
            y = int(round(self._last_point[1] + dy * t))
            self._draw_point((x, y))
        self._last_point = point

    def end_stroke(self) -> None:
        # Finalize stroke and commit canvas state to undo stack
        if self._is_drawing:
            if self._tool_mode == "line" and self._line_start is not None and self._last_point is not None:
                self._draw_line(self._line_start, self._last_point)
            self._push_undo()
        self._is_drawing = False
        self._last_point = None
        self._line_start = None

    def clear(self) -> None:
        # Clear the canvas and push previous state to undo stack
        self._push_undo()
        self._canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def _draw_point(self, point: Tuple[int, int]) -> None:
        x, y = point
        radius = self.get_brush_size()
        color_bgr = self._current_color
        cv2.circle(self._canvas, (x, y), radius, (*color_bgr, 255), thickness=-1, lineType=cv2.LINE_AA)

    def _draw_line(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
        thickness = max(1, self.get_brush_size() // 2)
        color_bgr = self._current_color
        cv2.line(self._canvas, p1, p2, (*color_bgr, 255), thickness=thickness, lineType=cv2.LINE_AA)

    def erase_stroke(self, p1: Tuple[int, int], p2: Tuple[int, int], thickness: Optional[int] = None) -> None:
        # Erase on the canvas layer only by painting transparent pixels, not black on the webcam frame
        if thickness is None:
            thickness = max(1, self.get_brush_size() // 2)
        cv2.line(self._canvas, p1, p2, (0, 0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)

    def set_template(self, template: np.ndarray, alpha: float = 0.9) -> None:
        if template is None:
            self.clear_template()
            return

        if template.ndim == 2:
            template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
            alpha_mask = np.ones((template.shape[0], template.shape[1]), dtype=np.uint8) * 255
        elif template.shape[2] == 4:
            template, alpha_mask = template[..., :3], template[..., 3]
        else:
            template = template[..., :3]
            alpha_mask = np.ones(template.shape[:2], dtype=np.uint8) * 255

        if template.shape[0] != self.height or template.shape[1] != self.width:
            template = cv2.resize(template, (self.width, self.height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            alpha_mask = cv2.resize(alpha_mask.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        self._template = template.astype(np.uint8)
        self._template_mask = alpha_mask.astype(np.uint8)
        self._template_alpha = max(0.0, min(1.0, float(alpha)))
        self._template_visible = True

    def clear_template(self) -> None:
        self._template = None
        self._template_mask = None
        self._template_alpha = 0.0
        self._template_visible = False

    def composite_on_frame(self, frame: np.ndarray) -> np.ndarray:
        # Blend template layer and canvas over a camera frame
        out = frame.copy().astype(np.float32)

        if self._template is not None and self._template_visible and self._template_alpha > 0.0:
            template = self._template
            mask = self._template_mask
            if mask is None:
                mask = np.ones((template.shape[0], template.shape[1]), dtype=np.uint8) * 255
            if template.shape[0] != frame.shape[0] or template.shape[1] != frame.shape[1]:
                template = cv2.resize(template, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            template_float = template.astype(np.float32)
            mask_float = (mask.astype(np.float32) / 255.0)[..., None]
            blend_alpha = self._template_alpha
            out = (template_float * blend_alpha * mask_float + out * (1.0 - blend_alpha * mask_float)).astype(np.uint8)

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            canvas = cv2.resize(self._canvas, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            canvas = self._canvas

        bgr_canvas = canvas[..., :3].astype(np.uint8)
        alpha_channel = canvas[..., 3].astype(np.float32) / 255.0
        alpha_map = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=-1)

        frame_float = out.astype(np.float32)
        bgr_canvas_float = bgr_canvas.astype(np.float32)

        out = (bgr_canvas_float * alpha_map + frame_float * (1 - alpha_map)).astype(np.uint8)
        return out

    def get_canvas_image(self) -> np.ndarray:
        return self._canvas.copy()
