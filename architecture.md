# VisionQuest Architecture

Last updated: 2026-06-07

## Layer Overview

VisionQuest is organized as:

```text
network -> web/mobile -> vision -> board tracking / AR -> game -> UI
```

`main.py` is the composition root. It starts the servers, reads the latest camera frame, schedules vision work, updates game state, renders AR/UI, sends preview frames to the phone, and handles cleanup.

## End-to-End Runtime Flow

```text
phone camera
  -> web/app.js JPEG capture
  -> network.WebSocketFrameServer
  -> network.FrameReceiver
  -> main.py
  -> vision.HandTracker
  -> vision.GestureDetector
  -> ar.PlaneTracker
  -> game.GameManager
  -> ar.ARRenderer / PyrenderModelRenderer
  -> ui.ActionCardRenderer / HUD / FloatingText
  -> desktop OpenCV window
  -> rendered phone preview
```

The system keeps only the newest frame. This is intentional: low latency is more important than processing every frame.

## `network/`

Responsibilities:

- Serve the mobile camera page.
- Receive binary JPEG frames.
- Track client connection state.
- Expose latest-frame freshness.
- Generate QR URL.
- Send rendered preview frames back to the phone.

Important files:

- `network/websocket_server.py`
- `network/frame_receiver.py`
- `network/qr_generator.py`

Key techniques:

- HTTP server and WebSocket server run separately.
- URL includes `?ws_port=8765` to keep frontend/backend port config synchronized.
- Rendered preview JPEG encoding runs in a background thread.
- Preview output is rate-limited and downscaled.

## `web/`

Responsibilities:

- Request phone camera permission.
- Capture frames through a reusable canvas.
- Send JPEG frames to PC.
- Show connection/frame status.
- Receive rendered preview frames from PC.
- Reconnect after WebSocket close/error.

Important files:

- `web/index.html`
- `web/app.js`

## `vision/`

Responsibilities:

- Hand landmark detection.
- Gesture feature construction.
- Gesture classification.
- Dataset capture for landmark and CNN paths.

Important files:

- `vision/hand_tracker.py`
- `vision/gesture_detector.py`
- `vision/gesture_features.py`
- `vision/dataset_capture_both.py`

Gesture output contract:

```python
{
    "gesture": str,
    "confidence": float,
    "smoothed_gesture": str,
    "margin": float,
    "second_confidence": float,
}
```

Gesture filtering uses high thresholds for entering a gesture, lower thresholds for maintaining it, and stricter thresholds for Shot gestures (`V_Sign`, `Gun_Sign`).

## `models/`

Responsibilities:

- Train gesture classifiers.
- Test gesture model independently.
- Store runtime artifacts.

Important files:

- `models/train.py`
- `models/train_landmarks.py`
- `models/train_cnn.py`
- `models/test_gesture_model.py`
- `models/gesture_model.pkl`
- `models/hand_landmarker.task`

Runtime uses landmark vectors:

```text
21 landmarks x (x, y, z) = 63 values
```

The project keeps PNG/CNN and landmark datasets separate so a CNN version remains possible later.

## `ar/plane_tracker.py`

Default mode:

```python
PlaneTracker(detector_mode="door_marker")
```

Responsibilities:

- Detect the 150 mm gate marker.
- Validate marker symbol structure.
- Estimate board homography.
- Smooth accepted homographies.
- Track registered board features.
- Re-detect after tracking loss.
- Keep legacy L-corner marker code as a fallback/comparison path.

Primary gate-marker detection:

```text
threshold / contour candidates
  -> quadrilateral filtering
  -> canonical marker patch
  -> outer border continuity
  -> central ring validation
  -> direction stem validation
  -> confidence score
  -> board homography
```

Registered tracking:

```text
last feature points
  -> LK optical flow
  -> forward/backward error check
  -> RANSAC homography
  -> reprojection/confidence check
  -> smoothing or short last-pose hold
```

Robustness details:

- Detection is downscaled for performance.
- Frame-edge-like quads are rejected.
- Marker validation happens after perspective normalization.
- Resolution/orientation changes clear frame-size-dependent caches.
- Re-detection interval becomes shorter when confidence is low.

## `ar/ar_renderer.py`

Responsibilities:

- Estimate camera intrinsics from homography samples.
- Build `solvePnP` pose from board corners.
- Draw fallback floor/primitive geometry.
- Warp cached ground texture onto the board.
- Render enemy GLB through `PyrenderModelRenderer`.
- Hide enemy during wave clear and reward selection.

Current policy:

- Player model is not rendered.
- Corner pillars are removed.
- Ground model is centered on the board and scaled up to cover the marker.
- Enemy model size is relative to board side length.
- Enemy render is downscaled and reused between frames for performance.

## `ar/pyrender_renderer.py`

Responsibilities:

- Load GLB/GLTF/OBJ with `trimesh`.
- Preload meshes in a background thread.
- Normalize mesh bounds and lift bottom to local `z=0`.
- Convert GLB/GLTF Y-up assets to board Z-up.
- Render RGBA through `pyrender.OffscreenRenderer`.
- Alpha-blend RGBA into OpenCV BGR frame.
- Render/caches top-down ground textures.

Axis conversion:

```text
game_x = asset_x
game_y = asset_z
game_z = asset_y
```

Camera conversion:

```text
OpenCV camera -> OpenGL camera
cv_to_gl = diag(1, -1, -1, 1)
```

Only board-Z yaw is currently exposed. Per-model axis correction is a recommended next step.

## `game/`

Responsibilities:

- Player and enemy runtime state.
- Wave selection and difficulty.
- Simultaneous card battle rules.
- Reward generation/application.
- Augment hooks.
- Event queue for UI/effects.

Important files:

- `game/game_manager.py`
- `game/battle_system.py`
- `game/wave_manager.py`
- `game/enemy.py`
- `game/player.py`
- `game/reward_system.py`
- `game/augment_system.py`
- `game/skills.py`

Main battle states:

```text
WAITING -> WAVE_INTRO -> PLAYER_TURN -> ROUND_REVEAL
  -> PLAYER_TURN / WAVE_CLEAR / DEFEAT
WAVE_CLEAR -> REWARD_SELECT -> WAVE_INTRO
```

Timing:

- Setup OK hold: 2.0 seconds.
- Player action hold: 2.0 seconds.
- Round reveal: 1.35 seconds.
- Reward hold: 2.0 seconds.
- Defeat restart OK hold: 2.0 seconds.

## Enemy and Wave Data

`EnemyType` fields:

```python
name: str
base_hp: int
base_damage: int
color: tuple
full_health_action_weights: dict
zero_health_action_weights: dict
action_weight_random_delta: float
min_wave: int
model_path: str | None
ground_model_path: str | None
```

Difficulty:

```text
global_difficulty_multiplier = 1.15 ** (current_wave - 1)
```

Dragon currently has `min_wave=4`.

## `ui/`

Responsibilities:

- Setup HUD and defeat HUD.
- AR-space player panel.
- AR-space action cards.
- AR-space reward cards.
- Enemy HP/action probability hint.
- Gesture probability panel.
- Augment badges.
- Floating combat feedback.

Important files:

- `ui/hud.py`
- `ui/action_cards.py`
- `ui/damage_text.py`

UI approach:

- Gameplay panels/cards are rendered as high-resolution RGBA images.
- They are projected or warped using the active homography.
- Sharp UI is drawn after AR rendering for both desktop and phone preview.

## Main Loop Order

```text
read latest fresh frame
  -> update gesture recognition on interval
  -> track board
  -> complete OK setup/start if needed
  -> process gesture card or reward input
  -> update game timers/state
  -> consume events
  -> render AR ground/enemy
  -> draw tracking attention/debug if needed
  -> build phone and desktop display frames
  -> draw sharp UI overlays
  -> publish phone preview
  -> handle Q/D/R keys
```

## Debug Policy

Before registration, setup overlays are visible. During gameplay, debug overlays are hidden unless tracking needs attention. Press `D` for full diagnostics:

- marker candidates
- board outline
- homography/tracking text
- hand overlay
- FPS/hand count

## Asset Policy

Preferred:

```text
assets/models/*.glb
assets/cards/*.png
```

Supported model formats:

```text
.glb .gltf .obj
```

`.glb` is preferred because materials and textures survive better than OBJ/MTL in the current path.
