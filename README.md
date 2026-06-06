# VisionQuest

VisionQuest is a computer-vision based AR dungeon battle simulator controlled by hand gestures.

The game is the demonstration layer. The core project goal is an end-to-end computer vision pipeline:

```text
smartphone camera
  -> WebSocket JPEG frames
  -> OpenCV desktop loop
  -> MediaPipe hand landmarks
  -> landmark-vector gesture classifier
  -> 150 mm gate-board marker tracking
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
- Default 150 x 150 mm hand-drawn gate-board registration using an outer frame, central ring, and short direction stem.
- Perspective-normalized gate marker validation, homography confidence, reprojection checks, and temporal smoothing.
- Registered-board tracking using multi-feature optical flow, forward/backward validation, RANSAC homography, and periodic gate re-detection.
- Infinite wave battle system with enemy types and difficulty scaling.
- Textured model rendering through `trimesh` + `pyrender` with OpenCV alpha blending.
- GLB/GLTF model support for enemy and ground models.
- Debug overlay toggle with `D`.
- Standalone gesture model tester at `models/test_gesture_model.py`.

Recently changed:

- The player model is no longer rendered on the board.
- The AR board now focuses on terrain/ground model plus the current enemy model.
- Old decorative corner pillars were removed.
- During gameplay, board outlines, homography text, marker debug graphics, FPS diagnostics, and other debug overlays are hidden by default.
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
6. Aim the phone at the complete square gate marker.
7. Hold `OK_Sign` while the detected board highlight is visible.
8. Keep holding until the progress outline completes; the board registers and the game starts together.

Controls:

```text
Q      quit
R      reset game
D      toggle debug overlays
```

## Gate Board Setup

Use a normal white A4 sheet, but draw one large gate marker at the center. The marker itself defines the playable board.

Recommended marker:

- Draw a bold hollow black square, about `15 cm x 15 cm`.
- Draw a hollow central ring inside the square.
- Draw a short dark stem downward from the ring so the tracker can infer orientation.
- Keep the inside of the square mostly white.
- The outer square corners are the AR board corners.

The older A4 corner L-marker detector is kept only as a legacy comparison path in code. The default runtime detector is the single gate marker because it is easier to draw, more thematic, and provides stronger shape validation than four small hand-drawn corner marks.

## Board Tracking Design

The board tracker borrows practical ideas from fiducial marker systems while keeping the marker simple enough to draw by hand.

Runtime structure:

```text
gate-marker square contour detection
  -> perspective-normalized patch
  -> outer frame + central ring + stem validation
  -> 150 mm square homography
  -> optical-flow feature tracking after registration
  -> periodic low-cost re-detection when confidence drops
  -> AR rendering
```

Important details:

- The gate marker is the board, so its four square corners directly define the AR plane.
- The hollow square is easy for thresholding/contours, while the central ring and stem reject ordinary rectangular noise.
- The stem resolves board orientation, so enemy/ground placement stays consistent when the paper rotates.
- Registered tracking uses internal feature points, forward/backward optical-flow checks, and RANSAC homography.
- Low-confidence results are rejected or held briefly to prevent the AR plane from jumping.

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
