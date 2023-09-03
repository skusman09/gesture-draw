# Air Canvas

A computer vision drawing application that uses webcam input and MediaPipe hand tracking to paint on a virtual canvas in real time. Built with Python, OpenCV, and MediaPipe.

## Requirements

- Python 3.10 or higher
- A working webcam

## Installation

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

The MediaPipe hand-landmark model (`models/hand_landmarker.task`) is downloaded automatically on first run, so no manual setup is needed.

## Usage

```bash
python -m src.app
```

## Gestures

| Gesture | Action |
| --- | --- |
| ☝️ Only index finger up | Draw with the active tool (brush or line) and selected color |
| ✌️ Index + middle up, ring/pinky down | Select / UI interaction — hover over toolbar items to pick colors, switch tools, adjust the brush-size slider, load templates, or trigger Undo / Redo / Clear / Draw / Erase |
| ✋ 3 or more fingers up (thumb ignored) | Erase strokes at the fingertip; raise index finger alone to resume drawing |
| ✊ All fingers down (fist, held ~0.5 sec) | Clear the entire canvas (debounced to prevent accidental wipes) |

## Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `ESC` / `q` | Quit |
| `u` | Undo last stroke |
| `r` | Redo last stroke |
| `c` | Clear canvas |
| `+` / `-` | Increase / decrease brush size |
| `f` | Load fish template |
| `h` | Load house template |

## Project Structure

```
gesture-draw/
├── config.py                    # Camera, UI, brush, and MediaPipe settings
├── src/
│   ├── app.py                   # Entry point
│   ├── controller.py            # Main loop: capture, gestures, canvas, UI
│   ├── core/hand_tracker.py     # Landmark detection + gesture classification
│   ├── canvas/canvas_engine.py  # Off-screen canvas, stroke history, undo/redo
│   ├── ui/overlay.py            # Toolbar, palette, buttons, cursor rendering
│   └── utils/smoothing.py       # Jitter reduction (EMA / Kalman)
├── models/                      # Auto-downloaded MediaPipe model
└── tests/                       # Pytest suite
```

## Testing

```bash
pytest
```

## Troubleshooting

- **"Python was not found" on Windows**: use `python` (not `python3`), or install Python from python.org and restart the terminal.
- **Black window / no camera**: close other apps using the webcam, or change `camera_index` in `config.py`.
