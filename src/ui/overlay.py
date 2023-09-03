from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

from config import (
    UI,
    BRUSH,
    PALETTE,
    HEADER_HEIGHT,
    BUTTON_W,
    BUTTON_H,
    GAP_SMALL,
    GAP_LARGE,
    PALETTE_SWATCH_SIZE,
    SLIDER_X,
    SLIDER_Y,
    SLIDER_H,
    SLIDER_W,
)

Color = Tuple[int, int, int]


# UI overlay rendering for toolbar, palette, action buttons, and cursor.
class OverlayRenderer:

    def __init__(self) -> None:
        self.topbar_height = UI.topbar_height
        self.palette = PALETTE
        self.palette_item_size = UI.palette_item_size
        self.palette_pos = UI.palette_position
        self.palette_spacing = UI.palette_spacing
        self.header_height = HEADER_HEIGHT
        self.button_w = BUTTON_W
        self.button_h = BUTTON_H
        self.gap_small = GAP_SMALL
        self.gap_large = GAP_LARGE
        self.palette_swatch_size = PALETTE_SWATCH_SIZE

    # Compute coordinate anchors for topbar controls.
    def _get_layout_bounds(self, frame_width: int) -> Tuple[int, int, int, int, int]:
        palette_x, palette_y = self.palette_pos
        palette_group_end = palette_x + len(self.palette) * (self.palette_item_size + self.palette_spacing)

        tool_x_start = palette_group_end + 18
        action_group_total_w = 5 * self.button_w + 4 * self.gap_small
        action_x_start = max(0, frame_width - action_group_total_w - 18)
        action_y = palette_y

        tool_group_end = tool_x_start + 2 * (self.button_w + self.gap_small)
        if action_x_start < tool_group_end + self.gap_large:
            action_y = palette_y + self.button_h + self.gap_small
            action_x_start = max(0, frame_width - action_group_total_w - 18)

        return palette_x, palette_y, tool_x_start, action_x_start, action_y

    # Build bounding rect registry for hit-testing and rendering.
    def build_ui_registry(self, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        palette_x, palette_y, tool_x_start, action_x_start, action_y = self._get_layout_bounds(frame_width)
        registry: List[Dict[str, Any]] = []

        for idx, color in enumerate(self.palette):
            rect = (palette_x + idx * (self.palette_item_size + self.palette_spacing), palette_y, self.palette_item_size, self.palette_item_size)
            registry.append({"name": f"palette_{idx}", "type": "palette", "rect": rect, "color": color})

        for idx, label in enumerate(("BRUSH", "LINE")):
            rect = (tool_x_start + idx * (self.button_w + self.gap_small), palette_y, self.button_w, self.button_h)
            registry.append({"name": label, "type": "tool", "rect": rect, "tool": label})

        for idx, label in enumerate(("UNDO", "REDO", "CLEAR", "DRAW", "ERASE")):
            rect = (action_x_start + idx * (self.button_w + self.gap_small), action_y, self.button_w, self.button_h)
            registry.append({"name": label, "type": "action", "rect": rect, "action": label})

        return registry

    # Return UI component at the given coordinate, if any.
    def get_hit_target(self, x: int, y: int, frame_width: int, frame_height: int) -> Optional[Dict[str, Any]]:
        for item in self.build_ui_registry(frame_width, frame_height):
            rx, ry, rw, rh = item["rect"]
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return item
        return None

    # Render top navigation bar and title text.
    def render_topbar(self, frame: np.ndarray, fps: float = 0.0, active_template: Optional[str] = "fish") -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, self.topbar_height), (30, 30, 30), thickness=-1)
        title = UI.window_name
        cv2.putText(frame, title, (16, 28), UI.font, UI.font_scale + 0.2, (255, 255, 255), UI.font_thickness + 1, cv2.LINE_AA)
        if UI.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 28), UI.font, UI.font_scale, (200, 200, 200), UI.font_thickness, cv2.LINE_AA)
        self.render_template_buttons(frame, active_template)

    # Render template selection buttons.
    def render_template_buttons(self, frame: np.ndarray, active_template: Optional[str] = "fish") -> None:
        h, w = frame.shape[:2]
        start_x = 20
        y = 20

        palette_group_end = start_x + len(self.palette) * self.palette_item_size + (len(self.palette) - 1) * self.gap_small
        tools_start_x = palette_group_end + self.gap_large
        tools_group_end = tools_start_x + 2 * self.button_w + self.gap_small
        templates_start_x = tools_group_end + self.gap_large

        for idx, label in enumerate(("FISH", "HOUSE")):
            x1 = templates_start_x + idx * (self.button_w + self.gap_small)
            y1 = y
            x2 = x1 + self.button_w
            y2 = y1 + self.button_h
            active = label.lower() == active_template
            fill = (70, 70, 70)
            border = (255, 255, 255) if active else (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, thickness=-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness=2)
            text_size, _ = cv2.getTextSize(label, UI.font, 0.45, 1)
            text_x = x1 + (self.button_w // 2) - (text_size[0] // 2)
            text_y = y1 + (self.button_h // 2) + (text_size[1] // 2) - 2
            cv2.putText(frame, label, (text_x, text_y), UI.font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Render action control buttons (undo, redo, clear, draw, erase).
    def render_controls(self, frame: np.ndarray, clear_active: bool = False, erase_active: bool = False, undo_active: bool = False, redo_active: bool = False, draw_active: bool = True) -> None:
        _, _, _, action_x_start, action_y = self._get_layout_bounds(frame.shape[1])

        actions = [
            ("UNDO", undo_active, (80, 80, 80)),
            ("REDO", redo_active, (90, 90, 90)),
            ("CLEAR", clear_active, (110, 110, 110)),
            ("DRAW", draw_active, (70, 120, 200)),
            ("ERASE", erase_active, (180, 180, 180)),
        ]

        for idx, (label, active, fill) in enumerate(actions):
            x1 = action_x_start + idx * (self.button_w + self.gap_small)
            y1 = action_y
            x2 = x1 + self.button_w
            y2 = y1 + self.button_h
            border = (255, 255, 255) if active else (120, 120, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, thickness=-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness=2)
            text_size, _ = cv2.getTextSize(label, UI.font, 0.5, 1)
            text_x = x1 + (self.button_w // 2) - (text_size[0] // 2)
            text_y = y1 + (self.button_h // 2) + (text_size[1] // 2) - 2
            cv2.putText(frame, label, (text_x, text_y), UI.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Render brush and line mode buttons.
    def render_mode_buttons(self, frame: np.ndarray, active_tool: str = "DRAW") -> None:
        _, y, tool_x_start, _, _ = self._get_layout_bounds(frame.shape[1])
        active_brush = active_tool in {"DRAW", "BRUSH"}
        active_line = active_tool == "LINE"

        for idx, (label, tool_name, active) in enumerate((
            ("BRUSH", "DRAW", active_brush),
            ("LINE", "LINE", active_line),
        )):
            x1 = tool_x_start + idx * (self.button_w + self.gap_small)
            y1 = y
            x2 = x1 + self.button_w
            y2 = y1 + self.button_h
            fill = (80, 120, 200) if tool_name == "LINE" and active else (70, 70, 70)
            border = (255, 255, 255) if active else (140, 140, 140)
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, thickness=-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness=2)
            text_size, _ = cv2.getTextSize(label, UI.font, 0.5, 1)
            text_x = x1 + (self.button_w // 2) - (text_size[0] // 2)
            text_y = y1 + (self.button_h // 2) + (text_size[1] // 2) - 2
            cv2.putText(frame, label, (text_x, text_y), UI.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Render color palette chips.
    def render_palette(self, frame: np.ndarray, active_index: int = 0) -> None:
        for i, color in enumerate(self.palette):
            x, y = self.palette_pos
            rect = (x + i * (self.palette_item_size + self.palette_spacing), y, self.palette_item_size, self.palette_item_size)
            cv2.rectangle(frame, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), color, thickness=-1)
            border_color = (255, 255, 255) if i == active_index else (120, 120, 120)
            cv2.rectangle(frame, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), border_color, thickness=2)

    # Render pointer cursor with contrast-adaptive border.
    def render_cursor(self, frame: np.ndarray, point: Tuple[int, int], color: Color, size: int, eraser: bool = False) -> None:
        x, y = point
        radius = max(2, size // 2)

        h, w = frame.shape[:2]
        pad = max(3, radius + 4)
        x0 = max(0, x - pad)
        x1 = min(w, x + pad + 1)
        y0 = max(0, y - pad)
        y1 = min(h, y + pad + 1)
        patch = frame[y0:y1, x0:x1]
        if patch.size:
            brightness = float(np.mean(patch[:, :, 0]) + np.mean(patch[:, :, 1]) + np.mean(patch[:, :, 2])) / 3.0 / 255.0
        else:
            brightness = 0.5

        cursor_ring = (255, 255, 255) if eraser else ((0, 0, 0) if brightness > 0.55 else (255, 255, 255))
        inner_dot = (0, 0, 0) if eraser else color

        cv2.circle(frame, (x, y), radius, cursor_ring, thickness=2, lineType=cv2.LINE_AA)
        if eraser:
            cv2.circle(frame, (x, y), radius, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), max(1, radius // 3), inner_dot, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(1, radius // 5), cursor_ring, thickness=1, lineType=cv2.LINE_AA)

    def render_status(self, frame: np.ndarray, text: str, pos: Tuple[int, int] = (16, 56)) -> None:
        cv2.putText(frame, text, pos, UI.font, UI.font_scale, (220, 220, 220), UI.font_thickness, cv2.LINE_AA)

    # Render vertical brush size slider on the left edge.
    def render_slider(self, frame: np.ndarray, current_size: int, color: Color = (200, 200, 200)) -> None:
        rail_x = SLIDER_X
        rail_y = SLIDER_Y
        rail_h = SLIDER_H
        rail_w = SLIDER_W
        knob_radius = 14
        min_s = BRUSH.min_brush_size
        max_s = BRUSH.max_brush_size

        # Rail background — rounded dark track
        rail_cx = rail_x + rail_w // 2
        cv2.rectangle(frame, (rail_x, rail_y), (rail_x + rail_w, rail_y + rail_h), (50, 50, 50), thickness=-1)
        cv2.circle(frame, (rail_cx, rail_y), rail_w // 2, (50, 50, 50), thickness=-1)
        cv2.circle(frame, (rail_cx, rail_y + rail_h), rail_w // 2, (50, 50, 50), thickness=-1)

        # Filled portion — gradient accent from knob to bottom (bottom = min)
        t = max(0.0, min(1.0, (current_size - min_s) / max(1, max_s - min_s)))
        knob_y = rail_y + int((1.0 - t) * rail_h)
        cv2.rectangle(frame, (rail_x, knob_y), (rail_x + rail_w, rail_y + rail_h), (180, 140, 60), thickness=-1)

        # Rail border
        cv2.rectangle(frame, (rail_x, rail_y), (rail_x + rail_w, rail_y + rail_h), (80, 80, 80), thickness=1)

        # Knob glow (subtle outer ring)
        cv2.circle(frame, (rail_cx, knob_y), knob_radius + 3, (80, 70, 40), thickness=-1, lineType=cv2.LINE_AA)
        # Knob body
        cv2.circle(frame, (rail_cx, knob_y), knob_radius, (220, 190, 90), thickness=-1, lineType=cv2.LINE_AA)
        # Knob inner highlight
        cv2.circle(frame, (rail_cx, knob_y), knob_radius - 4, (240, 215, 140), thickness=-1, lineType=cv2.LINE_AA)
        # Knob border
        cv2.circle(frame, (rail_cx, knob_y), knob_radius, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Size label — right of the knob
        label = str(current_size)
        label_x = rail_x + rail_w + knob_radius + 8
        label_y = knob_y + 5
        cv2.putText(frame, label, (label_x, label_y), UI.font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Min/Max tick labels at rail ends
        cv2.putText(frame, str(max_s), (rail_x + rail_w + 10, rail_y + 5), UI.font, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, str(min_s), (rail_x + rail_w + 10, rail_y + rail_h + 5), UI.font, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

        # Brush preview dot — below the rail
        preview_y = rail_y + rail_h + 40
        preview_r = max(2, current_size // 2)
        cv2.circle(frame, (rail_cx, preview_y), preview_r, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (rail_cx, preview_y), preview_r, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
