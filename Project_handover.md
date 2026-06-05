# VisionQuest Project Handover

Last updated: 2026-06-05

## Project Intent

VisionQuest is a computer vision project presented as an AR dungeon battle demo.

The game should remain simple enough that the CV pipeline is visible:

1. phone camera streaming
2. hand landmark detection
3. gesture classification
4. A4 board registration/tracking
5. homography and pose estimation
6. AR model compositing
7. gesture-driven turn-based combat

Avoid adding complex game systems unless they improve the demonstration.

## Current Runtime Flow

```text
phone browser camera
  -> web/app.js JPEG frames
  -> WebSocket server
  -> FrameReceiver latest frame
  -> main.py OpenCV loop
  -> HandTracker
  -> GestureDetector
  -> PlaneTracker
  -> GameManager
  -> ARRenderer / PyrenderModelRenderer
  -> HUD
```

`main.py` is the composition root. It starts HTTP and WebSocket servers automatically.

## Network Layer

Files:

- `network/websocket_server.py`
- `network/frame_receiver.py`
- `network/qr_generator.py`
- `web/index.html`
- `web/app.js`

Status:

- HTTP page runs on port `8000`.
- WebSocket frame stream runs on port `8765`.
- QR URL includes `?ws_port=8765`.
- Browser reconnect logic exists.
- `FrameReceiver` stores only the newest frame.
- Stale frame detection prevents processing frozen frames.
- `main.py` includes a short stale-frame grace period after board registration/start to avoid returning to the QR screen during transient processing pauses.

Known risks:

- HTTP LAN camera permission can be blocked by mobile browsers.
- Firewall may allow `8000` but block `8765`.
- HTTPS/WSS is not implemented.

## Vision Layer

Files:

- `vision/hand_tracker.py`
- `vision/gesture_detector.py`
- `vision/dataset_capture.py`
- `vision/dataset_capture_both.py`
- `models/train.py`
- `models/train_landmarks.py`
- `models/test_gesture_model.py`

Status:

- MediaPipe provides hand landmarks.
- Runtime gesture input is a normalized 63-value landmark vector.
- Active gesture classes:
  - `Fist`
  - `Open_Palm`
  - `V_Sign`
  - `Gun_Sign`
  - `OK_Sign`
- `OK_Sign` is start-only.
- `V_Sign` and `Gun_Sign` are trained as separate classes but both map to the same in-game ranged action.
- Model-only testing exists at `models/test_gesture_model.py`.

Recommendation:

- Validate gestures in `models/test_gesture_model.py` before testing the full AR game.
- Recollect landmark samples if `Open_Palm`, `V_Sign`, `Gun_Sign`, or `OK_Sign` is confused with another class.

## Board Tracking Layer

Files:

- `ar/plane_tracker.py`
- `ar/homography.py`

Current design:

- A4 sheet is the board.
- Dark hand-drawn corner marks are the primary detection target.
- White A4 boundary detection is only a fallback.
- Tracker supports partial occlusion:
  - 4 visible markers: direct marker homography
  - 3 visible markers: missing marker predicted from homography/world coordinates
  - tracking loss: last accepted homography held briefly
- Homography quality is checked with confidence and reprojection error.
- Accepted homographies are smoothed with EMA.
- Hand occlusion mask is used to reject candidates overlapping the detected hand.

Important recent fix:

- Missing marker debug points now come from `H(A4_world_marker)` projection.
- They are no longer stale screen-space points from the previous frame.

## AR Rendering Layer

Files:

- `ar/ar_renderer.py`
- `ar/pyrender_renderer.py`
- `ar/model_loader.py`

Current design:

- The A4 floor/grid uses the board homography.
- 3D textured assets use `solvePnP` pose derived from the A4 board.
- `trimesh` loads `.glb`, `.gltf`, and `.obj`.
- `pyrender` renders RGBA offscreen.
- OpenCV alpha blends the rendered model onto the camera frame.
- GLB/GLTF Y-up assets are converted to the board's Z-up coordinate system.
- Enemy models float up/down using a time-based sine offset.
- Player model rendering was removed.
- Decorative corner pillars were removed.
- If `pyrender` fails, OpenCV primitive fallback remains.

Model locations:

```text
assets/models/
```

Enemy/ground model configuration:

```text
game/wave_manager.py
```

Known risks:

- Offscreen rendering depends on OpenGL support.
- GLB animations are not played yet.
- Pose accuracy still depends on board tracking quality.

## Game and UI Layer

Files:

- `game/wave_manager.py`
- `game/game_manager.py`
- `game/battle_system.py`
- `game/enemy.py`
- `game/player.py`
- `game/skills.py`
- `ui/hud.py`
- `ui/damage_text.py`

Status:

- Infinite wave loop exists.
- Enemy type controls:
  - name
  - base HP
  - base damage
  - action weights
  - fallback color
  - enemy model path
  - ground model path
- Combat event queue feeds HUD/floating text.
- During gameplay, debug overlays are hidden by default.
- Press `D` to toggle debug overlays.

Controls:

```text
Q      quit
SPACE  board register / manual start
R      reset
D      debug overlays
```

## Validation Checklist

1. Gesture model:

```bash
python models/test_gesture_model.py
```

2. Full app:

```bash
python main.py
```

3. Phone page:

```text
WebSocket: connected
Frames sent increasing
```

4. Board:

- draw dark marks near A4 inside corners
- press `SPACE`
- confirm registration message
- hold `OK_Sign` or press `SPACE` again

5. AR:

- ground model appears on board center
- enemy model appears above ground
- enemy slowly floats
- `D` shows/hides marker and diagnostic overlays

## Next Recommended Tasks

- Add clearer on-screen error when `pyrender` is unavailable.
- Add optional per-enemy model scale/yaw/height settings in `EnemyType`.
- Add simple attack/idle transform animations before implementing real GLB animation playback.
- Add HTTPS/WSS support only if mobile camera permission remains a demo blocker.
- Continue improving gesture dataset quality.
