# VisionQuest

VisionQuest is a computer-vision based AR dungeon battle simulator controlled by hand gestures.

The project uses a game as the demonstration layer, but the main goal is to show an integrated computer vision pipeline:

- smartphone camera streaming to a PC
- hand landmark detection
- landmark-vector based gesture classification
- standard white A4 paper board detection and homography tracking
- perspective 3D-style battlefield overlay
- turn-based battle interaction

## Current Status

Implemented:

- Smartphone camera streaming through a local HTTP page and WebSocket frame server.
- QR code generation for the mobile camera page.
- Latest-frame receiver that discards stale frames.
- Camera stream display in the OpenCV desktop window.
- MediaPipe HandLandmarker integration for hand landmark extraction.
- Lightweight gesture model loading from `models/gesture_model.pkl`.
- Gesture classes:
  - `Fist`
  - `Open_Palm`
  - `V_Sign`
  - `OK_Sign` (reserved for starting the game)
- Basic turn-based battle system.
- Centered white A4 paper detection for the game board.
- Direct normalized DLT homography implementation for board-to-screen projection.
- 3D-style AR battlefield renderer that treats the A4 sheet as the virtual floor.
- Standalone gesture model test program at `models/test_gesture_model.py`.

In progress / needs validation:

- Gesture recognition quality is not yet confirmed to be reliable in the full game loop.
- The model input path has been unified around normalized MediaPipe landmark vectors, so old PNG image datasets should be recollected as `.npy` feature samples.
- The game starts from a held `OK_Sign` after A4 board registration. `SPACE` remains as a keyboard fallback.
- Some older comments or strings in the code may still have encoding damage and should be cleaned gradually.

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The expected model files are:

```text
models/hand_landmarker.task
models/gesture_model.pkl
```

## Running the Game

Start the full application:

```bash
python main.py
```

`main.py` starts both servers automatically:

- HTTP mobile page server: `http://<PC_IP>:8000/?ws_port=8765`
- WebSocket frame server: `ws://<PC_IP>:8765`

After startup:

1. Make sure the PC and phone are on the same network.
2. Scan the QR code shown in the PC OpenCV waiting window, open the generated `qr_code.png`, or use the printed URL.
3. Allow camera access on the phone.
4. Confirm the phone page shows:
   - `WebSocket: connected`
   - increasing `Frames sent`
5. The PC OpenCV window should show the phone camera stream.

For stable A4 board tracking, draw black marks near the four inside corners of the A4 sheet. A simple L shape or filled dot with a dark pen is enough. Keep each mark roughly 1-2 cm inside the paper corner. The tracker uses these high-contrast marks first, then falls back to white-paper boundary detection.

Controls in the PC OpenCV window:

- `Q`: quit
- `SPACE`: register the centered A4 board
- `OK_Sign`: start the battle after A4 registration
- `SPACE` again after registration: start the battle manually as fallback
- `R`: reset the game

## Mobile Camera Notes

Mobile browsers often block camera access on insecure LAN HTTP pages.

For quick testing in Chrome on Android, you can use:

```text
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```

Add the project URL, for example:

```text
http://<PC_IP>:8000
```

Then restart Chrome and reload the page.

If the page loads but WebSocket stays disconnected:

- check Windows Firewall for Python inbound access
- ensure port `8765` is reachable from the phone
- confirm the URL includes `?ws_port=8765`

## Capturing and Training Gesture Data

There are two gesture training pipelines. Dataset capture can save both formats at once.

### Recommended Capture: Both Formats

Use this by default so one recording session can support both future CNN experiments and the current landmark MLP runtime:

```bash
python vision/dataset_capture.py
```

Equivalent explicit command:

```bash
python vision/dataset_capture.py --mode both
```

This saves:

```text
dataset/<class>/*.png
dataset_landmarks/<class>/*.npy
```

Both files share the same timestamp, so samples can be matched later if needed.

### Option A: Landmark MLP (current runtime default)

The gesture pipeline uses normalized 63-value landmark feature vectors:

```text
21 landmarks x (x, y, z) = 63 values
```

Collect landmark training samples:

```bash
python vision/dataset_capture.py --mode landmarks
```

Equivalent direct script:

```bash
python vision/dataset_capture_landmarks.py
```

Controls:

- `1`: save `Fist`
- `2`: save `Open_Palm`
- `3`: save `V_Sign`
- `4`: save `OK_Sign`
- `Q`: quit

Train the model:

```bash
python models/train.py --mode landmarks
```

Equivalent direct script:

```bash
python models/train_landmarks.py
```

The trainer reads `.npy` feature files from:

```text
dataset_landmarks/Fist/
dataset_landmarks/Open_Palm/
dataset_landmarks/V_Sign/
dataset_landmarks/OK_Sign/
```

The older `dataset/` folder can be kept for future CNN/image experiments. It is not used by the current landmark-vector trainer.

The landmark pipeline saves a scikit-learn model by default:

```text
models/gesture_model.pkl
```

This is the default runtime model used by `main.py`, and it avoids TensorFlow native DLL issues.

`main.py` also avoids MediaPipe's documentation-only TensorFlow import path during hand-tracker startup. Seeing a TensorFlow Lite XNNPACK delegate log from MediaPipe is normal; the failing TensorFlow Python DLL runtime is not required for the default landmark pipeline.

### Option B: PNG CNN (kept for future experiments)

This path preserves the older image/CNN idea.

Capture PNG hand crops:

```bash
python vision/dataset_capture.py --mode cnn
```

Equivalent direct script:

```bash
python vision/dataset_capture_cnn.py
```

Train the CNN:

```bash
python models/train.py --mode cnn
```

Equivalent direct script:

```bash
python models/train_cnn.py
```

The CNN trainer reads PNG files from:

```text
dataset/Fist/
dataset/Open_Palm/
dataset/V_Sign/
dataset/OK_Sign/
```

CNN PNG samples are saved from the clean camera frame before UI text or MediaPipe landmark drawings are added.

The CNN output defaults to:

```text
models/gesture_model_cnn.keras
```

Note: the current `main.py` runtime uses the landmark-vector `.pkl` detector. Using the CNN `.keras` model in the game later will require TensorFlow to load correctly plus a separate CNN runtime detector or detector mode switch.

## Testing the Gesture Model Only

To test the trained model without the AR/game loop:

```bash
python models/test_gesture_model.py
```

Optional camera index:

```bash
python models/test_gesture_model.py --camera 1
```

Optional model path:

```bash
python models/test_gesture_model.py --model models/gesture_model.pkl
```

The test window shows:

- detected hand landmarks
- left/right hand predicted class
- confidence
- smoothed class
- detected hand count
- FPS

Use this script before debugging the full game. If this test cannot recognize hands or gestures reliably, the issue is in the vision/model pipeline rather than AR or battle logic.

## Project Structure

```text
ar/
  plane_tracker.py      centered white A4 board detection
  homography.py         direct DLT homography and coordinate projection
  ar_renderer.py        3D-style AR battlefield overlay on the A4 floor

game/
  battle_system.py      turn state and combat resolution
  game_manager.py       bridge between gesture input and battle system
  player.py
  enemy.py
  skills.py

models/
  train.py              CNN training script
  test_gesture_model.py standalone gesture model test runner
  gesture_model.pkl     trained landmark gesture model
  hand_landmarker.task  MediaPipe hand landmark model

network/
  frame_receiver.py     thread-safe latest-frame storage
  websocket_server.py   HTTP server and WebSocket JPEG frame receiver
  qr_generator.py       LAN URL QR code generation

ui/
  hud.py                OpenCV HUD drawing helpers

vision/
  hand_tracker.py       MediaPipe hand landmark wrapper
  gesture_detector.py   Keras gesture classifier wrapper
  dataset_capture.py    camera-based dataset capture helper
  dataset_collector.py  landmark-image dataset helper

web/
  index.html            mobile camera page
  app.js                mobile camera capture and WebSocket sender
```

## MVP Completion Checklist

- [x] Smartphone camera input path
- [x] Real-time desktop display of streamed frames
- [x] Hand landmark detection integration
- [x] Landmark-vector gesture model loading
- [x] Basic gesture-to-action mapping
- [x] Turn-based battle states
- [x] HP display
- [x] Manual centered A4 board registration
- [x] Direct DLT homography-based AR overlay path
- [ ] Recollected `.npy` landmark dataset in `dataset_landmarks/` for all four gestures
- [ ] Reliable gesture recognition in the full game loop
- [ ] Verified A4 board overlay stability under phone camera motion
- [ ] Clean all broken encoded UI strings
- [ ] Document final demo workflow after validation
