# VisionQuest Architecture

Last updated: 2026-06-05

## Purpose

VisionQuest is split into five main layers:

```text
network -> vision -> board tracking / AR -> game -> UI
```

`main.py` wires those layers together and owns the real-time OpenCV loop.

## End-to-End Flow

```text
phone camera
  -> web/app.js captures JPEG
  -> ws://<pc-ip>:8765
  -> network.WebSocketFrameServer
  -> network.FrameReceiver
  -> main.py
  -> vision.HandTracker
  -> vision.GestureDetector
  -> ar.PlaneTracker
  -> game.GameManager
  -> ar.ARRenderer
  -> ui.HUD
  -> OpenCV window
```

## `network/`

Responsibilities:

- serve the mobile camera page
- receive JPEG frames
- keep latest frame only
- expose frame freshness
- generate QR connection URL

Important files:

- `websocket_server.py`
- `frame_receiver.py`
- `qr_generator.py`

Design notes:

- Do not queue many frames; real-time freshness matters more than completeness.
- HTTP and WebSocket ports are separate.
- QR URL carries the WebSocket port as `?ws_port=8765`.

## `web/`

Responsibilities:

- request phone camera permission
- show connection status
- downscale camera frames
- send frames to PC WebSocket
- reconnect after disconnection

Important files:

- `index.html`
- `app.js`

## `vision/`

Responsibilities:

- run MediaPipe hand landmark detection
- normalize hand landmarks
- classify gestures
- support dataset capture

Important files:

- `hand_tracker.py`
- `gesture_detector.py`
- `dataset_capture.py`
- `dataset_capture_both.py`

Gesture contract:

```python
{
    "gesture": str,
    "confidence": float,
    "smoothed_gesture": str,
}
```

Current gesture mapping:

```text
Fist       Strike
Open_Palm  Guard
V_Sign     ranged attack
Gun_Sign   ranged attack
OK_Sign    setup confirmation only
```

## `models/`

Responsibilities:

- train gesture classifier
- test model independently
- store model artifacts

Important files:

- `train.py`
- `train_landmarks.py`
- `train_cnn.py`
- `test_gesture_model.py`
- `gesture_model.pkl`
- `hand_landmarker.task`

Runtime model format:

```text
21 landmarks x (x, y, z) = 63 float32 values
```

## `ar/plane_tracker.py`

Responsibilities:

- detect A4 board
- detect dark corner marks
- estimate homography
- smooth homography
- recover missing markers
- reject low-confidence tracking

Core flow:

```text
detect corner marks
  -> assign TL/TR/BR/BL marker slots
  -> compute homography
  -> estimate reprojection error/confidence
  -> smooth accepted H
  -> return corners, marker predictions, debug state
```

Partial occlusion rule:

- Observed markers are used for measurement.
- Missing marker screen positions are predicted from known A4 world marker coordinates and current homography.
- Last screen-space marker positions should not be used as the final missing-marker display/AR source.

## `ar/homography.py`

Responsibilities:

- direct normalized DLT homography
- point projection
- grid drawing helper

This file intentionally keeps core homography math project-owned instead of delegating the whole calculation to a high-level OpenCV wrapper.

## `ar/ar_renderer.py`

Responsibilities:

- draw A4 floor/grid
- derive approximate camera pose with `solvePnP`
- render ground and enemy models
- apply fallback OpenCV geometry if textured rendering fails

Current rendering policy:

- Player model is not rendered.
- Decorative board corner pillars are not rendered.
- Ground model is centered on the A4 board.
- Enemy model is above the ground and slowly bobs.

## `ar/pyrender_renderer.py`

Responsibilities:

- load GLB/GLTF/OBJ through `trimesh`
- preload configured enemy and ground assets in the background
- preserve GLB mesh/material structure where possible
- normalize model size
- convert GLB/GLTF Y-up assets into the board's Z-up coordinate system
- render RGBA through `pyrender.OffscreenRenderer`
- alpha blend into the OpenCV frame

Rendering pipeline:

```text
solvePnP rvec/tvec
  -> pyrender IntrinsicsCamera
  -> offscreen RGBA
  -> OpenCV BGR alpha blend
```

Fallback behavior:

- If `pyrender`, `trimesh`, OpenGL, or model loading fails, `ARRenderer` falls back to simpler OpenCV model/primitive rendering.
- If a preloaded model is still loading, the frame does not wait for it; fallback rendering is used until the cache is ready.

## `game/`

Responsibilities:

- wave progression
- player/enemy stats
- turn timing
- action resolution
- event queue

Important files:

- `wave_manager.py`
- `game_manager.py`
- `battle_system.py`
- `enemy.py`
- `player.py`
- `skills.py`

Enemy type fields:

```python
EnemyType(
    name=str,
    base_hp=int,
    base_damage=int,
    color=tuple,
    action_weights=dict,
    model_path=str | None,
    ground_model_path=str | None,
)
```

## `ui/`

Responsibilities:

- HUD panels
- battle status
- gesture/action display
- floating combat feedback
- debug-friendly overlays when enabled

Important files:

- `hud.py`
- `damage_text.py`

## Main Loop State

Important flags in `main.py`:

- `plane_registered`: A4 board has been registered
- `game_started`: battle has started
- `debug_mode`: debug overlays are visible
- `freshness_grace_until`: prevents short processing pauses from returning to QR setup
- `setup_gesture_counter`: stable setup `OK_Sign` count

Expected main loop order:

```text
read latest fresh frame
  -> detect hands/gestures on interval
  -> track A4 board
  -> complete setup when OK hold reaches the threshold
  -> update game state
  -> render AR
  -> render HUD/debug
  -> process keyboard input
```

## Debug Policy

Before game start:

- board highlight and diagnostics are visible to help registration

After game start:

- debug overlays are hidden by default
- press `D` to toggle:
  - A4 outline
  - marker circles/X predictions
  - homography confidence text
  - FPS/hand count diagnostics

## Asset Policy

Preferred:

```text
assets/models/*.glb
```

Supported:

```text
.glb
.gltf
.obj
```

Notes:

- `.glb` is preferred for material/texture preservation.
- `.obj` is accepted but visual quality is limited.
- Real animation playback is not implemented yet.
- Current movement is transform-based, such as enemy bobbing.
