# VisionQuest Architecture

Last updated: 2026-06-05

## Purpose

VisionQuest is a computer vision project that uses a simple AR battle game as the demonstration layer.

The architecture should optimize for:

- clear separation between camera/network, vision, AR, game logic, and HUD rendering
- fast debugging of each pipeline stage
- minimal game complexity until the CV pipeline is reliable
- repeatable demo setup

## High-Level Runtime Flow

```text
phone camera
  -> web/app.js captures frames as JPEG
  -> WebSocket ws://<pc-ip>:8765
  -> network.WebSocketFrameServer decodes JPEG
  -> network.FrameReceiver stores latest frame
  -> main.py reads latest fresh frame
  -> vision.HandTracker detects landmarks
  -> vision.GestureDetector classifies normalized landmark vector
  -> game.GameManager maps gesture to battle action
  -> ar.PlaneTracker tracks registered plane
  -> ar.ARRenderer draws battlefield overlay
  -> ui.HUD draws state and diagnostics
  -> OpenCV desktop window
```

`main.py` is the application composition root. It starts the network servers, owns the OpenCV loop, and coordinates all modules.

## Module Boundaries

### `network/`

Responsibility:

- serve the mobile camera web page
- receive JPEG frames over WebSocket
- keep only the latest frame for the desktop loop
- generate QR code URL for the phone

Key files:

- `websocket_server.py`: starts HTTP server and WebSocket frame server
- `frame_receiver.py`: thread-safe latest-frame storage and stale-frame checks
- `qr_generator.py`: local LAN IP detection and QR generation

Rules:

- Do not add OpenCV game or gesture logic here.
- Keep this layer focused on transport and frame delivery.
- The receiver should not queue unlimited frames; freshness is more important than completeness.

### `web/`

Responsibility:

- request mobile camera access
- display connection/frame status
- downscale frames
- send JPEG frames to the PC WebSocket server

Key files:

- `index.html`: mobile page shell
- `app.js`: camera capture, reconnect logic, WebSocket sender

Rules:

- Keep the mobile page lightweight.
- Prefer clear diagnostics: connected/disconnected, frame count, camera permission state.

### `vision/`

Responsibility:

- detect hands and landmarks
- convert landmarks into a normalized 63-value model input vector
- classify gestures
- support dataset capture/debugging

Key files:

- `hand_tracker.py`: MediaPipe HandLandmarker wrapper
- `gesture_detector.py`: landmark-vector gesture model wrapper, preferring `models/gesture_model.pkl`
- `dataset_capture.py`: camera landmark-vector dataset capture helper
- `dataset_collector.py`: landmark-vector dataset helper

Rules:

- No battle or AR state should live here.
- Gesture output should remain simple: gesture name, confidence, smoothed gesture.
- Keep model input format consistent between training and runtime.
- Current model input is `21 landmarks x 3 values = 63 float32 values`.

### `models/`

Responsibility:

- store trained models
- provide training and model-only testing utilities

Key files:

- `train.py`: lightweight landmark-vector classifier training script
- `test_gesture_model.py`: standalone camera-based gesture model tester
- `gesture_model.pkl`: trained landmark gesture classifier
- `hand_landmarker.task`: MediaPipe hand landmark model

Rules:

- Test `models/test_gesture_model.py` before blaming game or AR code.
- If predictions are bad here, fix dataset/model/training first.

### `ar/`

Responsibility:

- detect the centered white A4 paper board
- compute board-to-screen homography with direct DLT
- render 3D-style AR battlefield elements rising from the A4 floor

Key files:

- `plane_tracker.py`: centered white A4 detection using thresholding and connected components
- `homography.py`: direct normalized DLT homography and point projection helpers
- `ar_renderer.py`: perspective floor grid and raised player/enemy units

Rules:

- AR code should not know battle calculations.
- Registration remains manual: center a white A4 sheet and press `SPACE`.
- AR overlay is only expected after plane registration and game start.
- The renderer approximates camera pose from the board homography and a virtual camera matrix.

### `game/`

Responsibility:

- player/enemy state
- turn transitions
- action resolution
- gesture-to-action mapping through the manager/skills layer

Key files:

- `battle_system.py`: battle states and turn resolution
- `game_manager.py`: bridge from gesture result to game action
- `player.py`, `enemy.py`, `skills.py`

Rules:

- No OpenCV, MediaPipe, or networking logic should be added here.
- Keep mechanics simple until CV demo quality is stable.
- `OK_Sign` is a start gesture only and should not consume a battle turn.
- `GameManager` owns turn delays and repeated-input guards.

### `ui/`

Responsibility:

- draw OpenCV HUD text and bars
- show diagnostics useful during the demo

Key files:

- `hud.py`: HP, state, gesture, instruction drawing

Rules:

- UI should display current system state clearly.
- Broken encoded strings should be replaced with stable ASCII or verified UTF-8.

## Main Loop State

Important runtime flags in `main.py`:

- `plane_registered`: true after successful `PlaneTracker.register_plane(frame)`
- `game_started`: true after battle starts
- `start_gesture_counter`: counts stable `OK_Sign` frames for battle start

Expected flow:

```text
start main.py
  -> wait for fresh phone frame
  -> show camera stream
  -> center white A4 sheet and press SPACE to register board
  -> hold OK_Sign to start, or press SPACE again as keyboard fallback
  -> process player gestures during PLAYER_TURN
  -> render AR battlefield when homography is valid
```

## Gesture Contract

Runtime gesture classes:

```text
Fist      -> Attack
Open_Palm -> Defend
V_Sign    -> Skill
OK_Sign   -> Start game only
```

`GestureDetector.detect_gesture()` returns:

```python
{
    "gesture": str,
    "confidence": float,
    "smoothed_gesture": str,
}
```

The battle system should use `smoothed_gesture` with a confidence threshold.

Training and inference both use:

```text
MediaPipe landmarks -> wrist-relative normalization -> scale normalization -> 63-value vector
```

## Debugging Order

Use this order when something breaks:

1. Mobile page loads from `http://<pc-ip>:8000/?ws_port=8765`.
2. Phone page shows camera preview.
3. Phone page shows `WebSocket: connected`.
4. Phone page `Frames sent` increases.
5. PC OpenCV window shows phone camera stream.
6. `models/test_gesture_model.py` detects hands and gestures from a local camera.
7. Full `main.py` detects hands from the phone stream.
8. Press `SPACE` and verify plane registration.
9. Start battle and verify AR overlay.

When no fresh camera frame is available, `main.py` should render a connection setup screen with:

- QR code generated by `network.WebSocketFrameServer`
- exact mobile URL
- short firewall/Wi-Fi troubleshooting hint

## Current Technical Risks

- Mobile camera access over LAN HTTP may require browser flags or HTTPS/WSS.
- Old PNG hand-crop datasets in `dataset/` are reserved for possible future CNN experiments.
- Current landmark-vector training data lives in `dataset_landmarks/`.
- Runtime hand tracking installs a tiny `tensorflow.tools.docs.doc_controls` stub before importing MediaPipe Tasks. This avoids a TensorFlow Python DLL load caused by MediaPipe's documentation-only optional import path. MediaPipe still uses its own TensorFlow Lite runtime internally.
- MediaPipe processing can be slow if frame resolution is too high.
- Feature-poor surfaces can fail ORB plane registration.
- Some code comments/UI strings have encoding damage.
- AR overlay is currently gated by both plane registration and game start.

## Design Invariants

- Network layer only transports frames.
- Vision layer only detects/tracks/classifies visual input.
- Game layer only resolves gameplay state.
- AR/UI layers render state but should not own battle logic.
- Documentation should be updated whenever a milestone, known limitation, or run workflow changes.

## Gesture Pipeline Options

Two dataset/training paths are intentionally kept separate.

Default capture can populate both paths at once:

```text
vision/dataset_capture.py --mode both
  -> dataset/*.png
  -> dataset_landmarks/*.npy
```

### Landmark MLP

Current runtime default.

```text
vision/dataset_capture_landmarks.py -> dataset_landmarks/*.npy
models/train_landmarks.py -> models/gesture_model.pkl
vision/gesture_detector.py -> main.py
```

Wrapper commands:

```bash
python vision/dataset_capture.py --mode landmarks
python models/train.py --mode landmarks
```

### PNG CNN

Preserved for future image-based experiments.

```text
vision/dataset_capture_cnn.py -> dataset/*.png
models/train_cnn.py -> models/gesture_model_cnn.keras
```

PNG samples must be clean hand crops from the raw camera frame. Do not save UI text, MediaPipe landmark drawings, HUD elements, or AR overlays into CNN training images.

Wrapper commands:

```bash
python vision/dataset_capture.py --mode cnn
python models/train.py --mode cnn
```

The default landmark runtime uses scikit-learn pickle models and does not require TensorFlow. The CNN model is not currently consumed by `main.py`; add a CNN detector or detector mode switch before using `models/gesture_model_cnn.keras` in gameplay.
## A4 Board AR Tracking

The AR target has shifted from markerless arbitrary-plane tracking to a standard white A4 paper game board.

Runtime behavior:

```text
center white A4 sheet in camera view
  -> press SPACE
  -> PlaneTracker detects the centered white A4 candidate
  -> HomographyEstimator computes board-to-screen homography with direct DLT
  -> ARRenderer estimates an approximate pose from the homography
  -> ARRenderer draws a perspective floor grid and raised 3D units
```

Implementation constraints:

- Use OpenCV for camera input, color conversion, display, and basic image operations.
- Do not use `cv2.findHomography`, `cv2.getPerspectiveTransform`, `cv2.aruco`, or marker libraries for the core board solve.
- Current `ar/homography.py` computes homography with normalized DLT using NumPy SVD.
- Current `ar/plane_tracker.py` first detects four black marks near the inside corners of the A4 sheet, maps them to inset board coordinates, and computes the full A4 homography.
- After registration, marker detection is constrained to small ROIs predicted by the previous homography, so background changes are less likely to steal the track.
- Pixel morphology and component labeling use OpenCV's optimized C implementations; the project-owned core remains the marker geometry, validation, and direct DLT homography logic.
- If predicted marker ROIs are not found, `PlaneTracker` falls back to patch tracking, then short hold-last-corners behavior.
- Current `ar/ar_renderer.py` uses a screen-size-based virtual camera matrix to project simple 3D geometry onto the A4 board.

Known limitations:

- Plain white A4 detection can fail on white desks, bright backgrounds, glare, or weak contrast.
- The current detector uses a bounding rectangle of the white component, so extreme perspective angles are less accurate than a printed border or corner marker design.
- This tradeoff is intentional for the ordinary A4 with no special preparation requirement.
