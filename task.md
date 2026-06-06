# VisionQuest Task Notes

Last updated: 2026-06-07

## Current Focus

Prepare a stable live demo with:

- reliable 150 mm gate marker tracking
- smooth AR ground/enemy rendering
- readable AR-space UI
- gesture-only gameplay after startup
- clear wave, reward, augment, and restart flow

## Completed Recently

### Camera / Network

- `main.py` starts the HTTP camera page and WebSocket server.
- QR URL includes the WebSocket port.
- Latest-frame freshness prevents frozen-frame processing.
- PC-rendered game preview is sent back to the phone.
- Preview encoding is threaded and rate-limited.

### Board Tracking

- Single 150 mm gate marker is now the primary board detector.
- Marker validation uses outer square, central ring, and short direction stem.
- Frame-edge rejection reduces full-screen false overlays.
- Registered tracking uses LK optical flow and RANSAC homography.
- Confidence controls smoothing, pose hold, and re-detection.
- Phone resolution/orientation changes reset tracking caches.
- Legacy L-marker code remains but is not the default path.

### AR Rendering

- GLB/GLTF/OBJ loading through `trimesh`.
- Offscreen `pyrender` RGBA rendering and OpenCV alpha blending.
- GLB/GLTF Y-up to board Z-up conversion.
- Ground model top-down texture caching and homography warp.
- Player model and decorative corner pillars removed.
- Enemy model floats and is hidden during wave clear/reward selection.
- Enemy render is downscaled/reused to reduce FPS cost.

### Game Flow

- Combat changed to simultaneous card reveal.
- Player holds a gesture to choose Strike/Guard/Shot.
- Enemy action is selected from HP-based probabilities.
- Player/enemy cards reveal together before resolution.
- Wave clear opens reward selection.
- Defeat supports OK-sign 2 second run restart without board reset.

### Waves / Rewards / Augments

- Difficulty scales by `1.15 ** (wave - 1)`.
- Enemy types support `min_wave`.
- Dragon appears from wave 4 onward.
- Reward catalog includes stat, card upgrade, and augment rewards.
- Owned augments are excluded from future reward choices.
- Hook-style augment system implemented.

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

### UI

- Core gameplay UI moved into board-relative AR space.
- Player panel, action cards, reward cards, gesture graph, and augment badges enlarged.
- High-resolution card/panel rendering is used before perspective warp.
- Enemy HP and action probability hint are shown near enemy position.
- Floating combat text has small random offsets.
- Defeat panel includes restart instruction and border-progress highlight.
- Debug overlays are hidden during normal gameplay and toggled with `D`.

## Validation Checklist

Static check:

```bash
python -m py_compile main.py game/*.py ui/*.py ar/*.py network/*.py vision/*.py models/*.py
```

Gesture-only check:

```bash
python models/test_gesture_model.py
```

Full app:

```bash
python main.py
```

Manual checks:

- Phone page loads from QR.
- WebSocket connects and frames increase.
- Rendered preview appears on phone.
- Gate marker registers after OK hold.
- Game starts without keyboard input.
- Fist/Open Palm/V/Gun select correct cards.
- Player and enemy cards reveal together.
- Reward selection appears after wave clear.
- Owned augments are not offered again.
- Dragon does not appear before wave 4.
- Defeat screen restarts after OK hold.
- `D` toggles debug overlays.
- `R` hard-resets board registration and game state.

## Known Issues / Risks

- HTTPS/WSS is not implemented.
- `pyrender` depends on local OpenGL/offscreen support.
- GLB animations are not played.
- Camera calibration is approximate.
- Board pose quality depends on marker visibility and homography stability.
- Model facing direction varies by asset; only yaw is currently exposed.
- Korean augment labels in `game/augment_system.py` need encoding cleanup.
- Gesture accuracy depends on landmark dataset quality.

## Next Useful Work

High priority:

- Add per-enemy model orientation and scale settings.
- Clean Korean text encoding in `game/augment_system.py`.
- Add warning overlay if `pyrender` or model loading fails.
- Tune gate marker thresholds with live camera samples.

Medium priority:

- Add simple attack/hit/idle transform animations.
- Add persistent best-wave storage.
- Add marker drawing guide image.
- Add final demo checklist.

Low priority:

- HTTPS/WSS support.
- Real GLB animation playback.
- Camera calibration workflow.
- Revisit CNN runtime gesture path.
