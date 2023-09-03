from typing import Any
import time
import sys
from pathlib import Path

import cv2
import numpy as np

from config import (
    BRUSH,
    CAMERA,
    UI,
    HEADER_HEIGHT,
    ERASER_RADIUS,
    FIST_CLEAR_DEBOUNCE_FRAMES,
    LOST_TRACKING_RESET_FRAMES,
    TEMPLATE_BUTTON_W,
    TEMPLATE_BUTTON_GAP,
    TEMPLATE_CLOSE_PADDING,
    TEMPLATE_CLOSE_SIZE,
    SLIDER_X,
    SLIDER_Y,
    SLIDER_H,
    SLIDER_W,
)
from src.core.hand_tracker import HandTracker
from src.canvas.canvas_engine import CanvasEngine
from src.ui.overlay import OverlayRenderer
from src.utils.smoothing import ExponentialMovingAverageSmoother

# Main controller coordinating video capture, tracking, and UI.


class AppController:

    def __init__(self, camera_index: int = CAMERA.camera_index) -> None:
        self.camera_index: int = camera_index
        self.cap: cv2.VideoCapture | None = None
        self.width: int = CAMERA.frame_width
        self.height: int = CAMERA.frame_height
        self.fps_target: int = CAMERA.fps
        self.mirror: bool = CAMERA.mirror

        self.smoother: ExponentialMovingAverageSmoother = ExponentialMovingAverageSmoother(alpha=0.35)
        self.tracker: HandTracker = HandTracker(smoother=self.smoother)
        self.overlay: OverlayRenderer = OverlayRenderer()
        self.canvas: CanvasEngine | None = None

        self.active_color_index: int = 0
        self.current_tool: str = "DRAW"
        self.current_mode: str = "BRUSH"
        self.is_running: bool = False
        self._selection_target: str | None = None
        self._line_preview_start: tuple[int, int] | None = None
        self._line_preview_end: tuple[int, int] | None = None
        self._eraser_last_point: tuple[int, int] | None = None
        self._clear_fist_frames: int = 0
        self._lost_tracking_frames: int = 0
        self.template_names: list[str] = ["fish", "house"]
        self.template_index: int = 0
        self.show_template: bool = False
        self.show_image: bool = False
        self.active_template: str | None = None
        self.template_window_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._template_image: np.ndarray | None = None
        self._template_alpha_mask: np.ndarray | None = None
        self.draw_tool: str = "brush"
        self.current_brush_size: int = BRUSH.default_brush_size
        self.template_paths: dict[str, Path] = {
            "fish": self._resolve_template_path("fish"),
            "house": self._resolve_template_path("house"),
        }

    # Resolve template file path from assets directory or project root.
    def _resolve_template_path(self, name: str) -> Path:
        candidates = [
            Path(__file__).resolve().parent /
            "assets" / "templates" / f"{name}.png",
            Path(__file__).resolve().parent /
            "assets" / "templates" / f"{name}.jpg",
            Path(__file__).resolve().parent.parent / f"{name}.png",
            Path(__file__).resolve().parent.parent / f"{name}.jpg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    # Sync overlay state and canvas template visibility.
    def _set_template_visibility(self, visible: bool) -> None:
        self.show_image = bool(visible)
        self.show_template = self.show_image
        if self.canvas is not None:
            if hasattr(self.canvas, "_template_visible"):
                self.canvas._template_visible = self.show_image

    # Load and prepare template image with alpha mask for canvas overlay
    def _load_template(self, template_name: str) -> None:
        if self.canvas is None:
            return

        template_path = self.template_paths.get(template_name)
        if template_path is None or not template_path.exists():
            template_path = self._resolve_template_path(template_name)
            self.template_paths[template_name] = template_path
        if not template_path.exists():
            return

        image = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return

        if image.ndim == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            alpha_mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
        elif image.shape[2] == 4:
            image_bgr = image[..., :3]
            alpha_mask = image[..., 3]
        else:
            image_bgr = image[..., :3]
            alpha_mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255

        h, w = self.height, self.width
        split_x = w // 2
        panel_x = split_x + 30
        panel_y = UI.topbar_height + 30
        max_w = max(1, w - panel_x - 30)
        max_h = max(1, h - panel_y - 30)

        img_h, img_w = image_bgr.shape[:2]
        scale = min(max_w / img_w, max_h / img_h)
        display_w = max(1, int(img_w * scale))
        display_h = max(1, int(img_h * scale))
        resized_bgr = cv2.resize(
            image_bgr, (display_w, display_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(
            alpha_mask, (display_w, display_h), interpolation=cv2.INTER_AREA)

        paste_x2 = min(panel_x + display_w, w - 30)
        paste_y2 = min(panel_y + display_h, h - 30)
        if paste_x2 > panel_x and paste_y2 > panel_y:
            final_w = paste_x2 - panel_x
            final_h = paste_y2 - panel_y
            resized_bgr = cv2.resize(
                resized_bgr, (final_w, final_h), interpolation=cv2.INTER_AREA)
            resized_mask = cv2.resize(
                resized_mask, (final_w, final_h), interpolation=cv2.INTER_AREA)
        else:
            final_w = display_w
            final_h = display_h

        self._template_image = resized_bgr
        self._template_alpha_mask = resized_mask
        self.template_window_rect = (panel_x, panel_y, final_w, final_h)
        self.show_image = True
        self.show_template = True
        self.active_template = template_name

        canvas_template = np.zeros((h, w, 4), dtype=np.uint8)
        if final_w > 0 and final_h > 0:
            canvas_template[panel_y:panel_y + final_h,
                            panel_x:panel_x + final_w, :3] = resized_bgr
            canvas_template[panel_y:panel_y + final_h,
                            panel_x:panel_x + final_w, 3] = resized_mask
        self.canvas.set_template(canvas_template, alpha=1.0)

    # Return top-right close button bounding rect for active template window.
    def _get_template_close_button_rect(self) -> tuple[int, int, int, int] | None:
        if not self.show_image or self.active_template is None:
            return None

        x, y, w, h = self.template_window_rect
        if w <= 0 or h <= 0:
            return None

        rect = (x + w - TEMPLATE_CLOSE_SIZE - TEMPLATE_CLOSE_PADDING, y +
                TEMPLATE_CLOSE_PADDING, TEMPLATE_CLOSE_SIZE, TEMPLATE_CLOSE_SIZE)
        return rect

    # Return selection coordinate in frame space.
    def _normalize_selection_point(self, x: int, y: int) -> tuple[int, int]:
        return (x, y)

    # Cycle to next available template.
    def _next_template(self) -> None:
        if not self.template_names:
            return
        self.template_index = (self.template_index +
                               1) % len(self.template_names)
        self._load_template(self.template_names[self.template_index])

    # Blend template onto frame using PNG alpha channel.
    def _paste_template_to_frame(self, frame: np.ndarray) -> None:
        if not self.show_image or self.active_template is None or self._template_image is None:
            return

        x, y, w, h = self.template_window_rect
        if w <= 0 or h <= 0:
            return

        template = self._template_image
        if template.shape[:2] != (h, w):
            template = cv2.resize(
                template, (w, h), interpolation=cv2.INTER_AREA)

        alpha = self._template_alpha_mask
        if alpha is not None:
            if alpha.shape[:2] != (h, w):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
            alpha_float = alpha.astype(np.float32) / 255.0
            alpha_3 = np.stack(
                [alpha_float, alpha_float, alpha_float], axis=-1)
            roi = frame[y:y + h, x:x + w].astype(np.float32)
            overlay = template.astype(np.float32)
            result = roi * (1.0 - alpha_3) + overlay * alpha_3
            frame[y:y + h, x:x + w] = result.astype(np.uint8)
        else:
            roi = frame[y:y + h, x:x + w]
            frame[y:y + h, x:x +
                  w] = cv2.addWeighted(roi, 1.0, template, 1.0, 0)

        close_box = self._get_template_close_button_rect()
        if close_box is None:
            return
        close_x, close_y, close_w, close_h = close_box
        cv2.rectangle(frame, (close_x, close_y), (close_x + close_w,
                      close_y + close_h), (40, 40, 40), thickness=-1)
        cv2.rectangle(frame, (close_x, close_y), (close_x + close_w,
                      close_y + close_h), (255, 255, 255), thickness=2)
        cv2.line(frame, (close_x + 6, close_y + 6), (close_x + close_w - 6,
                 close_y + close_h - 6), (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(frame, (close_x + 6, close_y + close_h - 6), (close_x + close_w -
                 6, close_y + 6), (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # No-op hook for template panel rendering.
    def _render_template_panel(self, frame: np.ndarray) -> None:
        return

    # Handle template selection and closure from right panel.
    def _handle_template_selection(self, fingertip: tuple[int, int] | None) -> None:
        if fingertip is None:
            return

        x, y = fingertip
        w = self.width
        split_x = w // 2
        top_y = UI.topbar_height
        close_x1 = w - 64
        close_y1 = top_y + 10
        close_x2 = w - 20
        close_y2 = top_y + 56

        if x >= split_x and y >= top_y:
            close_box = self._get_template_close_button_rect()
            if close_box is not None:
                close_x, close_y, close_w, close_h = close_box
                if close_x < x < close_x + close_w and close_y < y < close_y + close_h:
                    self.active_template = None
                    self.show_image = False
                    self.show_template = False
                    self._template_image = None
                    self._template_alpha_mask = None
                    self.template_window_rect = (0, 0, 0, 0)
                    if self.canvas is not None:
                        self.canvas.clear_template()
                    return
            if not self.show_image:
                self._set_template_visibility(True)
            else:
                self._next_template()

    # Toggle between brush and line drawing modes.
    def _handle_draw_tool_selection(self, fingertip: tuple[int, int] | None) -> None:
        if fingertip is None:
            return
        x, y = fingertip
        h = self.height
        bottom_y = h - 78
        button_x = self.width // 2 - 120
        button_w = 80

        if bottom_y <= y <= h and button_x <= x <= button_x + button_w * 2 + 30:
            if x <= button_x + button_w:
                self.draw_tool = "brush"
            else:
                self.draw_tool = "line"
            if self.canvas is not None:
                self.canvas.set_tool_mode(self.draw_tool)

    # Render bottom center tool buttons.
    def _render_bottom_tools(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        btn_w = 120
        btn_h = 42
        gap = 20
        total_w = 2 * btn_w + gap
        x_start = (w - total_w) // 2
        y0 = h - 62

        buttons = [("BRUSH", 0), ("LINE", 1)]
        for label, idx in buttons:
            x1 = x_start + idx * (btn_w + gap)
            y1 = y0
            x2 = x1 + btn_w
            y2 = y1 + btn_h

            fill = (60, 60, 60)
            border = (255, 255, 255) if self.draw_tool == label.lower() else (
                150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, thickness=-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness=2)

            text_size, _ = cv2.getTextSize(label, UI.font, 0.45, 1)
            tx = x1 + (btn_w - text_size[0]) // 2
            ty = y1 + (btn_h + text_size[1]) // 2 - 2
            cv2.putText(frame, label, (tx, ty), UI.font,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Initialize camera capture and canvas engine.
    def _open_camera(self) -> None:
        self.cap = cv2.VideoCapture(
            self.camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open camera index {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.fps_target))

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(
                "Camera opened but failed to read initial frame")

        h, w = frame.shape[:2]
        self.width = w
        self.height = h
        self.canvas = CanvasEngine(width=w, height=h)

    # Extract index fingertip coordinates from landmark data.
    def _get_index_tip_point(
        self,
        landmarks: list[tuple[int, int]] | None,
        raw_landmarks: list[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        if raw_landmarks and len(raw_landmarks) > 8:
            return raw_landmarks[8]
        if landmarks and len(landmarks) > 8:
            return landmarks[8]
        return None

    # Process active gestures for selection, drawing, and clearing.
    def _handle_gestures(self, gestures: dict[str, bool], fingertip: tuple[int, int] | None) -> None:
        if self.canvas is None:
            return

        if gestures.get("erase", False) and not gestures.get("selection", False):
            # Open hand: temporarily switch to eraser and paint at the fingertip.
            if fingertip is None:
                return
            if self.canvas._is_drawing:
                self.canvas.end_stroke()
            if self.current_tool != "ERASE":
                self.current_tool = "ERASE"
                self.current_mode = "BRUSH"
            x1, y1 = fingertip
            if y1 > HEADER_HEIGHT:
                if not self.canvas._is_drawing:
                    self.canvas.start_stroke(fingertip)
                    # Sweep a transparent disc so accidental opaque-white marks get wiped too.
                    self.canvas.erase_stroke(
                        fingertip, fingertip, thickness=ERASER_RADIUS * 2)
                else:
                    self.canvas.update_stroke(fingertip)
                    xp, yp = self._eraser_last_point if self._eraser_last_point is not None else (
                        x1, y1)
                    self.canvas.erase_stroke(
                        (xp, yp), (x1, y1), thickness=ERASER_RADIUS * 2)
                self._eraser_last_point = (x1, y1)
            return

        if gestures.get("fist", False):
            if self.canvas._is_drawing:
                self._clear_fist_frames = 0
                self._selection_target = None
                return

            self._clear_fist_frames += 1
            if self._clear_fist_frames >= FIST_CLEAR_DEBOUNCE_FRAMES:
                self._clear_fist_frames = 0
                self._line_preview_start = None
                self._line_preview_end = None
                self.canvas.clear()
                self._selection_target = None
            return

        self._clear_fist_frames = 0

        if gestures.get("selection", False):
            if fingertip is None:
                return

            x, y = self._normalize_selection_point(*fingertip)

            # Slider hit-test — check before any other UI element
            slider_margin = 20
            slider_cx = SLIDER_X + SLIDER_W // 2
            if (abs(x - slider_cx) <= SLIDER_W // 2 + slider_margin
                    and SLIDER_Y - slider_margin <= y <= SLIDER_Y + SLIDER_H + slider_margin):
                clamped_y = max(SLIDER_Y, min(SLIDER_Y + SLIDER_H, y))
                t = 1.0 - (clamped_y - SLIDER_Y) / max(1, SLIDER_H)
                new_size = int(BRUSH.min_brush_size + t *
                               (BRUSH.max_brush_size - BRUSH.min_brush_size))
                self.current_brush_size = max(
                    BRUSH.min_brush_size, min(BRUSH.max_brush_size, new_size))
                if self.canvas is not None:
                    self.canvas.set_brush_size(self.current_brush_size)
                self._selection_target = "slider"
                return

            close_box = self._get_template_close_button_rect()
            if self.show_image and close_box is not None:
                cx, cy, cw, ch = close_box
                if cx - 30 < x < cx + cw + 30 and cy - 30 < y < cy + ch + 30:
                    self.active_template = None
                    self.show_image = False
                    self.show_template = False
                    self._template_image = None
                    self._template_alpha_mask = None
                    self.template_window_rect = (0, 0, 0, 0)
                    if self.canvas is not None:
                        self.canvas.clear_template()
                    self._eraser_last_point = (0, 0)
                    return

            ui_items = self.overlay.build_ui_registry(self.width, self.height)
            for item in ui_items:
                rx, ry, rw, rh = item["rect"]
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    item_type = item["type"]
                    if item_type == "palette":
                        idx = int(item["name"].split("_")[-1])
                        self.active_color_index = idx
                        self.canvas.set_color(self.overlay.palette[idx])
                        if self.current_tool == "ERASE":
                            self.current_tool = "DRAW"
                            self.current_mode = "BRUSH"
                            self.canvas.set_tool_mode("brush")
                        else:
                            self.canvas.set_tool_mode(
                                "line" if self.current_mode == "LINE" else "brush")
                        self._line_preview_start = None
                        self._line_preview_end = None
                        self._selection_target = "palette"
                        return

                    if item_type == "tool":
                        label = item["tool"]
                        if label == "BRUSH":
                            self.current_tool = "DRAW"
                            self.current_mode = "BRUSH"
                            self.canvas.set_tool_mode("brush")
                        else:
                            self.current_tool = "LINE"
                            self.current_mode = "LINE"
                            self.canvas.set_tool_mode("line")
                        self._line_preview_start = None
                        self._line_preview_end = None
                        self._selection_target = label.lower()
                        return

                    if item_type == "action":
                        label = item["action"]
                        if label == "UNDO":
                            self.canvas.undo()
                            self._selection_target = "undo"
                        elif label == "REDO":
                            self.canvas.redo()
                            self._selection_target = "redo"
                        elif label == "CLEAR":
                            self.canvas.clear()
                            self._selection_target = "clear"
                        elif label == "DRAW":
                            self.current_tool = "LINE" if self.current_mode == "LINE" else "DRAW"
                            self.canvas.set_tool_mode(
                                "line" if self.current_mode == "LINE" else "brush")
                            self._selection_target = "draw"
                        elif label == "ERASE":
                            self.active_color_index = len(
                                self.overlay.palette) - 1
                            self.current_tool = "ERASE"
                            self.current_mode = "ERASE"
                            self.canvas.set_color(
                                self.overlay.palette[self.active_color_index])
                            self.canvas.set_tool_mode("brush")
                            self._selection_target = "erase"
                        self._line_preview_start = None
                        self._line_preview_end = None
                        return

            if y < UI.topbar_height:
                template_total_w = TEMPLATE_BUTTON_W * 2 + TEMPLATE_BUTTON_GAP
                template_x = max(220, (self.width // 2) -
                                 (template_total_w // 2))
                template_y = 18
                for idx, name in enumerate(self.template_names):
                    x1 = template_x + idx * \
                        (TEMPLATE_BUTTON_W + TEMPLATE_BUTTON_GAP)
                    x2 = x1 + TEMPLATE_BUTTON_W
                    y2 = template_y + 42
                    if x1 <= x <= x2 and template_y <= y <= y2:
                        self._load_template(name)
                        self.active_template = name
                        self._selection_target = "template"
                        return

            return

        self._selection_target = None

        if gestures.get("index_up", False):
            # Single index finger always resumes normal drawing.
            if self.current_tool == "ERASE":
                self.current_tool = "DRAW"
                self.canvas.set_color(
                    self.overlay.palette[self.active_color_index])
            if fingertip is not None:
                x1, y1 = fingertip
                xp, yp = self._eraser_last_point if self._eraser_last_point is not None else (
                    0, 0)
                if self._lost_tracking_frames > 0 and self._lost_tracking_frames <= LOST_TRACKING_RESET_FRAMES:
                    xp, yp = self._eraser_last_point if self._eraser_last_point is not None else (
                        x1, y1)

                if y1 > HEADER_HEIGHT:
                    if self.current_tool == "ERASE":
                        # Erase via the canvas engine so pixels become transparent,
                        # not opaque white (compositing only respects the alpha channel).
                        self.canvas.set_brush_size(self.current_brush_size)
                        self.canvas.set_tool_mode("brush")
                        if not self.canvas._is_drawing:
                            self.canvas.start_stroke(fingertip)
                            self.canvas.erase_stroke(
                                fingertip, fingertip, thickness=ERASER_RADIUS * 2)
                        else:
                            self.canvas.update_stroke(fingertip)
                            self.canvas.erase_stroke(
                                (xp, yp), (x1, y1), thickness=ERASER_RADIUS * 2)
                        self._eraser_last_point = (x1, y1)
                        xp, yp = x1, y1
                    elif self.current_tool == "LINE":
                        self.canvas.set_color(
                            self.overlay.palette[self.active_color_index])
                        self.canvas.set_brush_size(self.current_brush_size)
                        self.canvas.set_tool_mode("line")
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        self._line_preview_start = (xp, yp)
                        self._line_preview_end = (x1, y1)
                        self.canvas._is_drawing = True
                        self.canvas._last_point = (x1, y1)
                    elif self.current_tool in ["BRUSH", "DRAW"]:
                        self.canvas.set_color(
                            self.overlay.palette[self.active_color_index])
                        self.canvas.set_brush_size(self.current_brush_size)
                        self.canvas.set_tool_mode("brush")
                        if not self.canvas._is_drawing:
                            self.canvas.start_stroke(fingertip)
                        else:
                            self.canvas.update_stroke(fingertip)
                else:
                    if self._lost_tracking_frames >= LOST_TRACKING_RESET_FRAMES:
                        self._eraser_last_point = (0, 0)
                        xp, yp = 0, 0
                    self._line_preview_start = None
                    self._line_preview_end = None
                    self.canvas._is_drawing = False
                    self.canvas._last_point = None
        else:
            xp, yp = self._eraser_last_point if self._eraser_last_point is not None else (
                0, 0)
            if self.current_tool == "LINE" and self._line_preview_start is not None and self._line_preview_end is not None and xp != 0 and yp != 0:
                line_start = self._line_preview_start
                line_end = self._line_preview_end
                if line_start != line_end:
                    cv2.line(self.canvas._canvas, line_start, line_end, (*self.overlay.palette[self.active_color_index], 255), thickness=max(
                        1, self.canvas.get_brush_size() // 2), lineType=cv2.LINE_AA)
                    self.canvas._push_undo()
                self._line_preview_start = None
                self._line_preview_end = None
                self._eraser_last_point = (0, 0)
                self.canvas._is_drawing = False
                self.canvas._last_point = None
                xp, yp = 0, 0
            elif self.canvas._is_drawing:
                if self.current_tool == "LINE":
                    self.canvas.set_tool_mode("line")
                self.canvas.end_stroke()
                self._line_preview_start = None
                self._line_preview_end = None
                self._eraser_last_point = (0, 0)
            else:
                self._line_preview_start = None
                self._line_preview_end = None
                self._eraser_last_point = (0, 0)

        if self.current_tool in ["DRAW", "BRUSH"]:
            self.canvas.set_tool_mode("brush")
        elif self.current_tool == "LINE":
            self.canvas.set_tool_mode("line")

    # Run the main event loop.
    def run(self) -> None:
        try:
            self._open_camera()
        except Exception as exc:
            sys.stderr.write(f"[ERROR] Camera error: {exc}\n")
            return

        if self.cap is None:
            sys.stderr.write("[ERROR] Camera not initialized\n")
            return

        if self.canvas is None:
            sys.stderr.write("[ERROR] Canvas not initialized\n")
            return

        cap = self.cap
        if self.active_template is not None:
            self._load_template(self.active_template)

        cv2.namedWindow(UI.window_name, cv2.WINDOW_NORMAL)
        self.is_running = True
        prev_time = time.time()
        fps = 0.0

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    sys.stderr.write("[WARN] Frame read failed; stopping\n")
                    break

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                tracking_scale = 0.75 if max(frame.shape[:2]) > 900 else 1.0
                tracking_frame = frame
                if tracking_scale != 1.0:
                    tracking_frame = cv2.resize(
                        frame,
                        (max(1, int(frame.shape[1] * tracking_scale)),
                         max(1, int(frame.shape[0] * tracking_scale))),
                        interpolation=cv2.INTER_LINEAR,
                    )

                try:
                    result = self.tracker.process_frame(tracking_frame)
                except Exception:
                    result = {"landmarks": None, "gestures": {
                        "index_up": False, "selection": False, "erase": False, "fist": False}}

                landmarks: Any = result.get("landmarks")
                raw_landmarks: Any = result.get("raw_landmarks")
                gestures: Any = result.get("gestures", {})
                fingertip = None
                if landmarks:
                    if tracking_scale != 1.0:
                        landmarks = [
                            (int(float(x) / tracking_scale), int(float(y) / tracking_scale)) for x, y in landmarks]
                        if raw_landmarks is not None:
                            raw_landmarks = [(int(float(
                                x) / tracking_scale), int(float(y) / tracking_scale)) for x, y in raw_landmarks]
                    fingertip = self._get_index_tip_point(
                        landmarks, raw_landmarks)

                if fingertip is not None:
                    self._lost_tracking_frames = 0
                else:
                    self._lost_tracking_frames += 1

                self._handle_gestures(gestures, fingertip)

                if self.show_image:
                    self._paste_template_to_frame(frame)

                now = time.time()
                dt = now - prev_time
                prev_time = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

                if self.current_tool == "LINE" and self._line_preview_start is not None and self._line_preview_end is not None:
                    draw_color = self.overlay.palette[self.active_color_index]
                    thickness = max(2, self.canvas.get_brush_size() // 2)
                    cv2.line(frame, self._line_preview_start, self._line_preview_end,
                             draw_color, thickness=thickness, lineType=cv2.LINE_AA)

                canvas = self.canvas.get_canvas_image()[..., :3]
                img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, img_mask = cv2.threshold(
                    img_gray, 0, 255, cv2.THRESH_BINARY)
                img_inv = cv2.bitwise_not(img_mask)
                frame = cv2.bitwise_and(frame, frame, mask=img_inv)
                frame = cv2.bitwise_or(frame, canvas)
                composed = frame

                self._render_template_panel(composed)
                self.overlay.render_topbar(
                    composed, fps=fps, active_template=self.active_template)
                self.overlay.render_palette(
                    composed, active_index=self.active_color_index)
                self.overlay.render_mode_buttons(
                    composed, active_tool=self.current_tool)
                self.overlay.render_slider(
                    composed,
                    self.current_brush_size,
                    color=self.overlay.palette[self.active_color_index],
                )
                self.overlay.render_controls(
                    composed,
                    clear_active=False,
                    erase_active=(self.current_tool == "ERASE"),
                    undo_active=False,
                    redo_active=False,
                    draw_active=(self.current_tool in {"DRAW", "BRUSH"}),
                )
                if self.current_mode == "LINE" and self._line_preview_start is not None and self._line_preview_end is not None:
                    preview_color = self.overlay.palette[self.active_color_index]
                    thickness = max(2, self.canvas.get_brush_size() // 2)
                    cv2.line(composed, self._line_preview_start, self._line_preview_end,
                             preview_color, thickness=thickness, lineType=cv2.LINE_AA)
                    cv2.circle(composed, self._line_preview_start, 3,
                               preview_color, thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(composed, self._line_preview_end, 3,
                               preview_color, thickness=-1, lineType=cv2.LINE_AA)
                if fingertip:
                    self.overlay.render_cursor(
                        composed,
                        fingertip,
                        self.overlay.palette[self.active_color_index],
                        self.canvas.get_brush_size(),
                        eraser=(self.current_tool == "ERASE"),
                    )

                cv2.imshow(UI.window_name, composed)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.is_running = False
                elif key == ord('u'):
                    self.canvas.undo()
                elif key == ord('r'):
                    self.canvas.redo()
                elif key == ord('c'):
                    self.canvas.clear()
                elif key in (ord('f'), ord('F')):
                    self._load_template("fish")
                elif key in (ord('h'), ord('H')):
                    self._load_template("house")
                elif key in (ord('+'), ord('=')):
                    self.current_brush_size = min(
                        BRUSH.max_brush_size, self.current_brush_size + 2)
                    self.canvas.set_brush_size(self.current_brush_size)
                elif key in (ord('-'), ord('_')):
                    self.current_brush_size = max(
                        BRUSH.min_brush_size, self.current_brush_size - 2)
                    self.canvas.set_brush_size(self.current_brush_size)
                elif key in (ord('q'), ord('Q')):
                    self.is_running = False

        finally:
            self.shutdown()

    # Release camera and destroy OpenCV windows.
    def shutdown(self) -> None:
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            self.tracker.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
