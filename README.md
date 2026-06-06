# VisionQuest

VisionQuest is a computer-vision based AR dungeon battle simulator controlled by hand gestures.

The game is the demonstration layer. The core project goal is an end-to-end computer vision pipeline:

```text
smartphone camera
  -> WebSocket JPEG frames
  -> OpenCV desktop loop
  -> MediaPipe hand landmarks
  -> landmark-vector gesture classifier
  -> A4 board marker tracking
  -> homography / solvePnP pose
  -> AR battlefield compositing
  -> turn-based battle interaction
```

## Current Status

Implemented:

- Smartphone camera streaming through a local HTTP page and WebSocket frame server.
- QR code generation shown in the OpenCV waiting screen.
- Latest-frame receiver with stale-frame detection and short registration/start grace handling.
- MediaPipe hand landmark detection.
- Landmark-vector gesture classifier.
- Gesture classes:
  - `Fist`: Strike
  - `Open_Palm`: Guard
  - `V_Sign`: ranged attack
  - `Gun_Sign`: ranged attack, trained separately from `V_Sign`
  - `OK_Sign`: board registration and game start during setup
- A4 board registration using dark corner marks drawn near the paper corners.
- Homography confidence, reprojection error checks, temporal smoothing, and missing marker prediction.
- Partial marker occlusion handling using current homography projection from A4 world coordinates.
- Infinite wave battle system with enemy types and difficulty scaling.
- Textured model rendering through `trimesh` + `pyrender` with OpenCV alpha blending.
- GLB/GLTF model support for enemy and ground models.
- Debug overlay toggle with `D`.
- Standalone gesture model tester at `models/test_gesture_model.py`.

Recently changed:

- The player model is no longer rendered on the board.
- The AR board now focuses on terrain/ground model plus the current enemy model.
- Old decorative corner pillars were removed.
- During gameplay, A4 boxes, homography text, marker debug graphics, FPS diagnostics, and other debug overlays are hidden by default.
- Press `D` to show or hide debug overlays.
- Enemy models float slowly up and down to avoid a static, frozen look.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Important packages:

- `opencv-python`
- `mediapipe`
- `numpy`
- `scikit-learn`
- `websockets`
- `qrcode`
- `trimesh`
- `pyrender`
- `PyOpenGL`

Expected model files:

```text
models/hand_landmarker.task
models/gesture_model.pkl
```

## Running

Start the full application:

```bash
python main.py
```

`main.py` starts both servers:

```text
HTTP camera page: http://<PC_IP>:8000/?ws_port=8765
WebSocket frames: ws://<PC_IP>:8765
```

Startup flow:

1. Connect PC and phone to the same network.
2. Run `python main.py`.
3. Scan the QR code shown in the desktop OpenCV waiting screen.
4. Allow camera access on the phone.
5. Confirm the phone page shows `WebSocket: connected` and increasing frame count.
6. Aim the phone at the A4 board.
7. Hold `OK_Sign` while the detected board highlight is visible.
8. Keep holding until the progress outline completes; the board registers and the game starts together.

Controls:

```text
Q      quit
R      reset game
D      toggle debug overlays
```

## A4 Board Setup

Use a normal white A4 sheet. Draw dark marks near the four inside corners.

Recommended:

- Use a black pen or marker.
- Draw an L-shaped mark or a bold dot near each inside corner.
- Keep each mark roughly 1-2 cm inside the paper edge.
- Keep the A4 sheet near the center of the camera view during initial registration.

The tracker uses the corner marks first. Plain white-paper boundary detection exists only as a fallback and is less reliable.

## Board Tracking Design

The board tracker borrows practical ideas from fiducial marker systems while keeping the marker simple enough to draw by hand.

Runtime structure:

```text
L/dark corner mark detection
  -> 4 marker slots, or 3 observed + 1 predicted
  -> homography calculation
  -> reprojection error
  -> confidence estimation
  -> reject low-confidence H or hold previous H briefly
  -> EMA homography smoothing
  -> AR rendering
```

Important details:

- Marker slots correspond to A4 world coordinates, not arbitrary screen points.
- If one marker is hidden, its screen position is predicted by projecting its known A4 world coordinate through the current/previous homography.
- Missing marker debug X marks are now homography-projected predictions, not stale screen-space memories.
- Hand occlusion masks are used to reject marker candidates overlapping the detected hand area.
- Homography confidence uses visible corner count, detector score, temporal consistency, board sanity, and reprojection error.
- Low-confidence results are rejected to prevent the AR plane from jumping.

References behind the tracker design:

- STag: A Stable Fiducial Marker System
- Designing Highly Reliable Fiducial Markers
- Planar Fiducial Markers: A Comparative Study
- Fiducial Markers for Pose Estimation: overview/comparison literature

These were used as design inspiration for confidence scoring, temporal smoothing, partial occlusion handling, and re-detection after tracking loss.

## AR Rendering

Current AR rendering path:

```text
board corners / homography
  -> solvePnP pose
  -> trimesh loads GLB/GLTF/OBJ
  -> pyrender renders textured model to RGBA
  -> alpha blend onto OpenCV frame
```

Model behavior:

- Enemy and ground models are set per enemy type.
- Models should be placed under `assets/models/`.
- Preferred format is `.glb`.
- `.obj` remains supported as a fallback, but visual quality is lower.
- GLB/GLTF Y-up assets are converted to the board's Z-up coordinate system.
- Enemy models are rendered slightly above the ground and slowly bob up/down.

Enemy model configuration lives in:

```text
game/wave_manager.py
```

Example:

```python
EnemyType(
    name="Slime",
    base_hp=50,
    base_damage=6,
    color=(50, 200, 50),
    action_weights={"Attack": 0.50, "Defend": 0.30, "Skill": 0.20},
    model_path=str(Path("assets") / "models" / "Slime.glb"),
    ground_model_path=str(Path("assets") / "models" / "Grass.glb"),
)
```

If `pyrender` cannot render a model, the renderer falls back to simpler OpenCV geometry.

## Game Flow

State flow:

```text
camera setup
  -> OK hold board registration/start
  -> wave intro
  -> player turn
  -> enemy turn
  -> wave clear
  -> next wave or defeat
```

Combat gestures:

```text
Fist       Strike
Open_Palm  Guard
V_Sign     ranged attack
Gun_Sign   ranged attack
OK_Sign    setup confirmation only
```

The run is endless. The goal is to reach the highest wave possible.

## Gesture Data

Recommended capture:

```bash
python vision/dataset_capture.py --mode both
```

This can save both:

```text
dataset/<class>/*.png
dataset_landmarks/<class>/*.npy
```

The current runtime uses landmark vectors, not PNG image classification.

Train landmark model:

```bash
python models/train.py --mode landmarks
```

Test model only:

```bash
python models/test_gesture_model.py
```

If gestures are poor in the tester, fix dataset/model quality before debugging the full game.

## Mobile Camera Notes

Mobile browsers may block camera access on insecure LAN HTTP pages.

For quick Android Chrome testing:

```text
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```

Add:

```text
http://<PC_IP>:8000
```

Then restart Chrome.

If the phone page loads but WebSocket is disconnected:

- check Windows Firewall for Python inbound access
- confirm phone and PC are on the same network
- confirm port `8765` is reachable
- confirm the URL includes `?ws_port=8765`

## Known Limitations

- HTTPS/WSS is not implemented.
- Camera calibration is approximate; pose quality depends on stable board homography.
- `pyrender` offscreen rendering depends on local OpenGL support.
- GLB animation clips are not played yet; only static first-frame model geometry/materials are used.
- The gesture classifier still depends heavily on collected landmark dataset quality.
