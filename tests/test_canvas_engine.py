import numpy as np
from config import PALETTE, FIST_CLEAR_DEBOUNCE_FRAMES
from src.canvas.canvas_engine import CanvasEngine
from src.controller import AppController


def test_palette_black_color_is_not_pure_black():
    black_color = PALETTE[4]
    assert black_color != (0, 0, 0)
    assert all(channel > 0 for channel in black_color)


def test_canvas_init_and_draw():
    w, h = 320, 240
    engine = CanvasEngine(width=w, height=h)
    assert engine.get_canvas_image().shape == (h, w, 4)
    engine.set_color((0, 255, 0))
    engine.start_stroke((10, 10))
    engine.update_stroke((20, 20))
    engine.end_stroke()
    img = engine.get_canvas_image()
    assert img[..., 3].sum() > 0
    engine.undo()
    assert engine.get_canvas_image().shape == (h, w, 4)


def test_canvas_template_layer():
    w, h = 320, 240
    engine = CanvasEngine(width=w, height=h)
    template = np.ones((h, w, 3), dtype=np.uint8) * 255
    template[50:150, 50:150] = 0
    engine.set_template(template, alpha=1.0)

    out = engine.composite_on_frame(np.zeros((h, w, 3), dtype=np.uint8))
    assert out.shape == (h, w, 3)
    assert np.any(out < 255)


def test_template_loads_only_on_right_side(tmp_path):
    w, h = 320, 240
    controller = AppController(camera_index=0)
    controller.width = w
    controller.height = h
    controller.canvas = CanvasEngine(width=w, height=h)

    template_path = tmp_path / "paint.png"
    template = np.ones((80, 120, 3), dtype=np.uint8) * 255
    template[:, 20:80] = 0
    import cv2
    cv2.imwrite(str(template_path), template)

    controller.template_paths["paint"] = template_path
    controller._load_template("paint")

    right_template = controller.canvas._template
    assert right_template is not None
    assert np.all(right_template[:, : w // 2] == 0)
    assert np.any(right_template[:, w // 2 :] != 0)


def test_template_alpha_keeps_camera_visible_on_left():
    engine = CanvasEngine(width=200, height=100)
    template = np.zeros((100, 200, 4), dtype=np.uint8)
    template[:, 100:, :3] = 255
    template[:, 100:, 3] = 255
    template[:, :100, 3] = 0
    engine.set_template(template, alpha=1.0)

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :100] = 200
    out = engine.composite_on_frame(frame)

    assert np.allclose(out[:, :100], 200, atol=2)
    assert np.all(out[:, 100:] == 255)


def test_show_image_toggle_controls_template_visibility():
    controller = AppController(camera_index=0)
    controller.width = 640
    controller.height = 480
    controller.canvas = CanvasEngine(width=640, height=480)

    assert controller.show_image is False

    controller.show_image = True
    controller.active_template = "fish"
    controller._load_template("fish")
    assert controller.show_image is True
    assert controller.active_template == "fish"

    controller.show_image = False
    controller._handle_template_selection((600, 30))
    assert controller.show_image is False


def test_controller_pointer_uses_index_tip_alignment():
    controller = AppController(camera_index=0)
    landmarks = [(0, 0)] * 21
    raw_landmarks = [(0, 0)] * 21
    landmarks[5] = (100, 100)
    landmarks[8] = (100, 40)
    raw_landmarks[5] = (100, 100)
    raw_landmarks[8] = (100, 20)

    pointer = controller._get_index_tip_point(landmarks, raw_landmarks)
    assert pointer is not None
    assert pointer == raw_landmarks[8]


def test_selection_mode_sets_current_mode_and_actions():
    controller = AppController(camera_index=0)
    controller.width = 640
    controller.height = 480
    controller.canvas = CanvasEngine(width=640, height=480)

    palette_x = controller.overlay.palette_pos[0]
    palette_y = controller.overlay.palette_pos[1]
    palette_point = (palette_x + 60, palette_y + 20)
    controller._handle_gestures({"selection": True}, palette_point)
    assert controller.current_mode == "BRUSH"
    assert controller.active_color_index == 1

    erase_item = next(item for item in controller.overlay.build_ui_registry(controller.width, controller.height) if item["type"] == "action" and item["action"] == "ERASE")
    erase_rect = erase_item["rect"]
    erase_center = (erase_rect[0] + erase_rect[2] // 2, erase_rect[1] + erase_rect[3] // 2)
    controller._handle_gestures({"selection": True}, erase_center)
    assert controller.current_mode == "ERASE"
    assert controller.active_color_index == len(controller.overlay.palette) - 1

    controller.current_mode = "BRUSH"
    controller.canvas.set_color((255, 0, 0))
    controller._handle_gestures({"index_up": True}, (200, 200))
    assert controller.canvas._is_drawing is True
    controller._handle_gestures({"index_up": False}, (200, 200))
    assert controller.canvas._is_drawing is False


def test_palette_selection_keeps_line_mode_and_only_switches_from_erase():
    controller = AppController(camera_index=0)
    controller.width = 640
    controller.height = 480
    controller.canvas = CanvasEngine(width=640, height=480)

    palette_x = controller.overlay.palette_pos[0]
    palette_y = controller.overlay.palette_pos[1]
    palette_point = (palette_x + 60, palette_y + 20)

    controller.current_tool = "LINE"
    controller.current_mode = "LINE"
    controller.canvas.set_tool_mode("line")
    controller._handle_gestures({"selection": True}, palette_point)
    assert controller.current_tool == "LINE"
    assert controller.current_mode == "LINE"
    assert controller.active_color_index == 1

    controller.current_tool = "ERASE"
    controller.current_mode = "ERASE"
    controller._handle_gestures({"selection": True}, palette_point)
    assert controller.current_tool == "DRAW"
    assert controller.current_mode == "BRUSH"
    assert controller.active_color_index == 1


def test_fist_clear_uses_clear_button_logic():
    controller = AppController(camera_index=0)
    controller.width = 640
    controller.height = 480
    controller.canvas = CanvasEngine(width=640, height=480)
    controller.canvas.set_color((0, 0, 255))
    controller.canvas.start_stroke((50, 50))
    controller.canvas.update_stroke((100, 100))
    controller.canvas.end_stroke()

    before = controller.canvas.get_canvas_image().copy()
    for _ in range(FIST_CLEAR_DEBOUNCE_FRAMES - 1):
        controller._handle_gestures({"fist": True}, (50, 50))
        assert controller.canvas.get_canvas_image().shape == before.shape
        assert controller.canvas.get_canvas_image()[..., 3].sum() > 0

    controller._handle_gestures({"fist": True}, (50, 50))
    assert controller.canvas.get_canvas_image().shape == before.shape
    assert controller.canvas.get_canvas_image()[..., 3].sum() == 0


def test_header_tool_buttons_set_line_and_brush_modes():
    controller = AppController(camera_index=0)
    controller.width = 1280
    controller.height = 720
    controller.canvas = CanvasEngine(width=1280, height=720)

    palette_w = len(controller.overlay.palette) * (controller.overlay.palette_item_size + controller.overlay.palette_spacing)
    tool_x = controller.overlay.palette_pos[0] + palette_w + 30
    brush_point = (tool_x + 10, controller.overlay.palette_pos[1] + 10)
    line_point = (tool_x + 90, controller.overlay.palette_pos[1] + 10)

    controller._handle_gestures({"selection": True}, brush_point)
    assert controller.current_mode == "BRUSH"
    assert controller.current_tool == "DRAW"

    controller._handle_gestures({"selection": True}, line_point)
    assert controller.current_mode == "LINE"
    assert controller.current_tool == "LINE"


def test_draw_button_is_not_replaced_by_brush_and_line_starts_immediately():
    controller = AppController(camera_index=0)
    controller.width = 640
    controller.height = 480
    controller.canvas = CanvasEngine(width=640, height=480)

    palette_x, palette_y = controller.overlay.palette_pos
    action_items = controller.overlay.build_ui_registry(controller.width, controller.height)
    draw_item = next(item for item in action_items if item["type"] == "action" and item["action"] == "DRAW")
    draw_rect = draw_item["rect"]
    center = (draw_rect[0] + draw_rect[2] // 2, draw_rect[1] + draw_rect[3] // 2)

    controller._handle_gestures({"selection": True}, center)
    assert controller.current_tool == "DRAW"
    assert controller.current_mode == "BRUSH"

    controller.current_mode = "LINE"
    controller.current_tool = "LINE"
    controller.canvas._is_drawing = False
    controller._handle_gestures({"index_up": True}, (200, 200))
    assert controller._line_preview_start == (200, 200)
    assert controller._line_preview_end == (200, 200)


def test_erase_stroke_only_updates_canvas():
    engine = CanvasEngine(width=200, height=100)
    engine.set_color((0, 255, 0))
    engine.start_stroke((10, 10))
    engine.update_stroke((40, 20))
    engine.end_stroke()

    before = engine.get_canvas_image().copy()
    engine.erase_stroke((10, 10), (80, 20), thickness=12)

    after = engine.get_canvas_image()
    assert after[..., 3].sum() < before[..., 3].sum()
    assert np.any(after[..., 3] == 0)
