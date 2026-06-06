# VisionQuest

VisionQuest is a computer-vision based AR dungeon battle game controlled by hand gestures.

Last updated: 2026-06-07

## Project Goal

The game is the demo layer for an end-to-end real-time CV pipeline:

```text
phone camera -> WebSocket JPEG frames -> OpenCV loop
  -> MediaPipe hand landmarks -> gesture classifier
  -> 150 mm gate marker tracking -> homography / solvePnP
  -> AR ground/enemy rendering -> gesture card battle
  -> optional rendered preview back to phone
```

## Current Status

Implemented:

- Local HTTP/WebSocket phone camera streaming with QR setup.
- Latest-frame receiver, stale-frame detection, and graceful setup/start handling.
- PC-rendered game frame preview sent back to the phone.
- MediaPipe hand landmarks plus landmark-vector gesture classifier.
- Gesture confidence filtering using probability and class-margin thresholds.
- Single 150 mm gate marker board tracking.
- Legacy L-corner marker code preserved as a comparison path.
- Homography confidence, reprojection checks, smoothing, optical-flow tracking, RANSAC recovery, and periodic re-detection.
- GLB/GLTF ground and enemy rendering with `trimesh`, `pyrender`, and OpenCV alpha blending.
- AR-space player info, action cards, reward cards, enemy HP, enemy action hints, gesture probability bars, and augment badges.
- Simultaneous card reveal combat.
- Infinite waves with multiplicative difficulty scaling.
- Run-limited rewards and hook-style augments.
- Defeat restart by holding `OK_Sign` for 2 seconds.
- Debug overlays toggled by `D`.

## Requirements

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
- `Pillow`

Expected runtime files:

```text
models/hand_landmarker.task
models/gesture_model.pkl
```

The `.pkl` gesture model is preferred at runtime because it avoids TensorFlow native DLL loading in `main.py`.

## Running

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
3. Scan the QR code shown in the OpenCV waiting screen.
4. Allow camera access on the phone.
5. Aim the camera at the 150 mm gate marker.
6. Hold `OK_Sign` for 2 seconds to register the board and start the run.
7. During combat, hold an action gesture for 2 seconds.
8. After defeat, hold `OK_Sign` for 2 seconds to restart the run without re-registering the board.

Controls:

```text
Q      quit program
R      hard reset, including board registration
D      toggle debug overlays
```

Normal gameplay after startup is intended to be gesture-only.

## Gate Marker

Use a white A4 sheet and draw one large gate marker. The marker itself is the game board.

Recommended marker:

- Hollow black square, about `15 cm x 15 cm`.
- Hollow central ring.
- Short downward stem from the ring, ending before the bottom border.
- Mostly white interior.

Why this marker is the default:

- The large square gives stable four-corner geometry.
- The central ring rejects ordinary rectangles and table edges.
- The short stem resolves orientation.
- It is easier to draw and more immersive than a chessboard.
- It is more stable than four small L markers.

## Tracking Technique

The board tracker is inspired by fiducial-marker papers such as STag and comparative marker studies, but uses a hand-drawable custom marker.

```text
gate square contour candidates
  -> canonical perspective-normalized patch
  -> border continuity validation
  -> central ring validation
  -> direction stem validation
  -> 150 mm board homography
  -> solvePnP pose for 3D rendering
  -> optical-flow tracking after registration
```

Key techniques:

- Downscaled detection for speed.
- Quad geometry and frame-edge rejection.
- Symbol validation in normalized marker space.
- Confidence scoring and EMA smoothing.
- LK optical flow with forward/backward validation.
- RANSAC homography for tracked points.
- Last-pose hold and periodic re-detection when confidence drops.
- Tracking cache reset when phone resolution changes.

## Gestures

```text
Fist       Strike
Open_Palm  Guard
V_Sign     Shot
Gun_Sign   Shot
OK_Sign    setup / restart only
```

`V_Sign` and `Gun_Sign` are trained separately for accuracy but map to the same Shot card in game. Shot gestures use stricter confidence and margin thresholds because accidental Shot actions are costly.

Test the gesture model independently:

```bash
python models/test_gesture_model.py
```

## AR Rendering

Current rendering path:

```text
board homography -> solvePnP pose -> trimesh GLB/GLTF load
  -> pyrender RGBA -> OpenCV alpha blend
```

Model notes:

- Preferred model format: `.glb` in `assets/models/`.
- `.obj` remains supported as a lower-quality fallback.
- Enemy and ground model paths are configured in `game/wave_manager.py`.
- Ground is rendered as a cached top-down texture and warped to the board plane.
- Enemy is rendered above the board, slowly bobbing up and down.
- Player model is not rendered; player information is AR-space UI.

GLB/GLTF axis conversion:

```text
game_x = asset_x
game_y = asset_z
game_z = asset_y
```

This converts common Y-up assets into the board's Z-up space. Model-specific facing direction may still require per-asset correction.

## Game Flow

```text
camera setup -> OK hold board registration/start -> wave intro
  -> player card hold -> simultaneous reveal -> round resolution
  -> next turn or wave clear -> reward select -> next wave
  -> defeat -> OK hold run restart
```

Player base stats:

```text
Max HP: 100
Attack power: 15
Strike damage: attack_power + strike_bonus
Shot damage: (attack_power + shot_bonus) * 2
Guard heal: max(5, missing_hp * (10% + guard ratio bonus)) plus flat bonus
```

Difficulty:

```text
Wave N multiplier = 1.15 ** (N - 1)
```

Enemy HP and damage use this multiplier.

## Rewards and Augments

After each wave clear, three reward cards appear. Reward categories are:

- `stat`
- `heal`
- `card_upgrade`
- `augment`

Already-owned augments are removed from future reward pools.

Implemented augments:

- Double Attack
- Cull the Weak
- Deep Rest
- Counter Guard
- Chicken Game
- Vampire
- Prepared
- Insurance
- First Strike

## Known Limitations

- HTTPS/WSS is not implemented.
- Camera calibration is approximate.
- `pyrender` depends on local OpenGL/offscreen support.
- GLB animation clips are not played yet.
- Per-model pitch/roll/yaw correction is not yet stored in `EnemyType`.
- Gesture quality depends on balanced landmark data.
- Some Korean augment labels in code need encoding cleanup before final presentation.
