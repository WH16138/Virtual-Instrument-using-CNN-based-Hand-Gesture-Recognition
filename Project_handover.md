# VisionQuest Project Handover

Last updated: 2026-06-05

## Project Intent

VisionQuest is a computer vision project presented as an AR dungeon battle simulator.

The game layer should stay simple. The core value of the project is demonstrating an end-to-end CV interaction pipeline:

1. capture camera frames
2. detect hands
3. classify gestures with a trained landmark-vector model
4. register and track a planar surface
5. render an AR-style battlefield
6. map gestures to turn-based battle actions

Avoid spending major time on game complexity until the CV pipeline is stable.

## Current Architecture

### Runtime Flow

```text
phone browser camera
  -> JPEG frames over WebSocket
  -> network.FrameReceiver
  -> main.py OpenCV loop
  -> vision.HandTracker
  -> vision.GestureDetector
  -> game.GameManager
  -> ar.PlaneTracker / ar.ARRenderer
  -> ui.HUD
```

`main.py` owns the full application loop and starts the HTTP/WebSocket servers automatically.

### Network Layer

Files:

- `network/websocket_server.py`
- `network/frame_receiver.py`
- `network/qr_generator.py`
- `web/index.html`
- `web/app.js`

Status:

- HTTP server serves the mobile page from `web/` on port `8000`.
- WebSocket server receives JPEG frames on port `8765`.
- QR URL includes `?ws_port=8765` so the web client does not rely on a hidden port assumption.
- `FrameReceiver` keeps only the newest frame.
- Stale frame detection exists so the desktop loop does not process an old frozen frame forever.
- The mobile sender downsizes frames to a maximum width of `640` for performance.

Known risks:

- Mobile camera access over LAN HTTP may require browser flags or HTTPS.
- Firewall can allow `8000` while blocking `8765`.
- HTTPS/WSS support is not implemented yet.

### Vision Layer

Files:

- `vision/hand_tracker.py`
- `vision/gesture_detector.py`
- `vision/dataset_capture.py`
- `vision/dataset_collector.py`

Status:

- `HandTracker` wraps MediaPipe HandLandmarker.
- `GestureDetector` loads `models/gesture_model.keras`.
- Gesture input is now a normalized 63-value vector from 21 MediaPipe landmarks.
- Runtime gesture classes are:
  - `Fist`
  - `Open_Palm`
  - `V_Sign`
- A standalone tester exists at `models/test_gesture_model.py`.

Important issue to validate:

- The code path has been unified around normalized landmark vectors saved as `.npy` files.
- Old PNG hand-crop datasets in `dataset/` are preserved for possible future CNN experiments, but are no longer valid for the current `models/train.py`.
- Recollect gesture samples with `python vision/dataset_capture.py`, then retrain with `python models/train.py`.
- Current landmark-vector samples are stored in `dataset_landmarks/`.

### AR Layer

Files:

- `ar/plane_tracker.py`
- `ar/homography.py`
- `ar/ar_renderer.py`

Status:

- Plane registration uses ORB features from the current frame.
- Tracking uses BFMatcher, Lowe ratio filtering, and RANSAC homography.
- AR renderer can draw grid/player/enemy overlays when a valid homography exists.

How to use:

- Press `SPACE` once to register a plane.
- After registration, start the game manually with `SPACE` or use the two-hand start flow if gesture recognition is reliable.

Known risks:

- Feature-poor surfaces fail registration.
- Phone camera motion blur can break homography.
- AR overlay only appears when both `plane_registered` and `game_started` are true.

### Game and UI Layer

Files:

- `game/battle_system.py`
- `game/game_manager.py`
- `game/player.py`
- `game/enemy.py`
- `game/skills.py`
- `ui/hud.py`

Status:

- Turn states exist: waiting, player turn, enemy turn, victory, defeat.
- Gesture mapping:
  - `Fist` -> Attack
  - `Open_Palm` -> Defend
  - `V_Sign` -> Skill
- HP display exists.
- Some HUD/code strings may still be corrupted from earlier encoding problems and should be cleaned incrementally.

## Recent Work Completed

- Stabilized smartphone camera streaming path.
- Added WebSocket port query parameter to the QR/mobile URL.
- Added WebSocket reconnection logic in the browser client.
- Added stale frame detection in `FrameReceiver`.
- Added server `stop()` handling.
- Added a standalone gesture model test runner:

```bash
python models/test_gesture_model.py
```

- Reduced mobile frame upload size to improve desktop processing performance.
- Removed the mid-loop debug `imshow/waitKey` that caused visual flicker.
- Restored missing `cv2` import in `vision/hand_tracker.py`.
- Replaced image-based gesture training/inference with normalized landmark-vector training/inference.
- Added frame/drop control in `main.py` by running heavy vision processing every `VISION_INTERVAL_FRAMES` frames while keeping display updates live.

## How to Validate the Current Build

### 1. Test Model Only

```bash
python models/test_gesture_model.py
```

Expected:

- camera window opens
- hand landmarks are drawn
- left/right prediction panels update
- confidence changes as gestures change

If this fails, fix vision/model issues before debugging the AR game.

### 2. Test Full Camera Streaming

```bash
python main.py
```

Phone page should show:

- `WebSocket: connected`
- increasing `Frames sent`

Desktop window should show:

- phone camera frame
- FPS
- camera resolution and hand count

### 3. Test AR Path

1. Aim the phone at a textured flat surface.
2. Press `SPACE` in the desktop OpenCV window.
3. Confirm plane registration succeeds in the console.
4. Start the battle.
5. Confirm the AR overlay appears.

## Next Recommended Tasks

## Gesture Training Options

There are now two separate dataset/training paths.

Default capture saves both formats in one session:

```bash
python vision/dataset_capture.py
```

This writes:

```text
dataset/<class>/*.png
dataset_landmarks/<class>/*.npy
```

Landmark MLP path, used by the current runtime:

```bash
python vision/dataset_capture.py --mode landmarks
python models/train.py --mode landmarks
```

Direct scripts:

```bash
python vision/dataset_capture_landmarks.py
python models/train_landmarks.py
```

Data and model:

```text
dataset_landmarks/
models/gesture_model.pkl
```

Legacy PNG/CNN path, preserved for future experiments:

```bash
python vision/dataset_capture.py --mode cnn
python models/train.py --mode cnn
```

Direct scripts:

```bash
python vision/dataset_capture_cnn.py
python models/train_cnn.py
```

Data and model:

```text
dataset/
models/gesture_model_cnn.keras
```

Current `main.py` expects the landmark-vector `.pkl` model and should not import TensorFlow on startup. Using the CNN model in gameplay later requires TensorFlow to load correctly plus a CNN runtime detector or detector mode switch.

## Next Recommended Tasks

1. Recollect `.npy` landmark-vector data for all gesture classes in `dataset_landmarks/`.
2. Retrain `models/gesture_model.keras`.
3. Verify `models/test_gesture_model.py` in real camera conditions.
4. Clean corrupted strings in remaining modules.
4. Add an explicit debug mode in `main.py`:
   - camera-only
   - hand-tracking-only
   - gesture-only
   - full AR/game
5. Consider HTTPS/WSS support if browser flags are not acceptable for demo.
6. Tune plane registration and AR overlay once gesture recognition is stable.

## Demo Readiness

The project is not yet final-demo ready.

It is close to an integrated MVP, but these must be verified before presentation:

- phone camera stream is stable for several minutes
- hand detection works through the streamed phone camera
- gesture classes are recognized reliably enough for gameplay
- plane registration succeeds on the chosen demo surface
- AR overlay appears after plane registration and game start
- broken UI strings are cleaned or hidden
