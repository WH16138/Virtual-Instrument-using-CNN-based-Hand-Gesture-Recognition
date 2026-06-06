# VisionQuest Project Handover

Last updated: 2026-06-07

## Project Intent

VisionQuest is a computer vision project presented as an AR dungeon battle game.

The live demo should make the pipeline visible:

1. smartphone camera streaming
2. hand landmark detection
3. gesture classification
4. hand-drawn gate marker tracking
5. homography and pose estimation
6. textured AR compositing
7. gesture-driven card combat
8. rewards and run progression

Keep game additions explainable and stable. The game should support the CV demonstration, not hide it.

## Current Runtime Flow

```text
phone browser camera
  -> web/app.js JPEG frames
  -> WebSocketFrameServer
  -> FrameReceiver latest frame
  -> main.py OpenCV loop
  -> HandTracker / GestureDetector
  -> PlaneTracker
  -> GameManager
  -> ARRenderer / PyrenderModelRenderer
  -> ActionCardRenderer / HUD / FloatingText
  -> desktop display and phone preview
```

`main.py` starts all runtime services and owns cleanup.

## Current User Flow

1. Run `python main.py`.
2. Scan QR code with phone.
3. Allow camera access.
4. Show the 150 mm gate marker.
5. Hold `OK_Sign` for 2 seconds.
6. Play with gestures only:
   - `Fist`: Strike
   - `Open_Palm`: Guard
   - `V_Sign` or `Gun_Sign`: Shot
7. Select rewards after wave clear.
8. On defeat, hold `OK_Sign` for 2 seconds to restart the run without board reset.

Keyboard controls:

```text
Q      quit
D      debug overlays
R      hard reset including board registration
```

## Network Layer

Files:

- `network/websocket_server.py`
- `network/frame_receiver.py`
- `network/qr_generator.py`
- `web/index.html`
- `web/app.js`

Status:

- HTTP page: port `8000`.
- WebSocket frames: port `8765`.
- QR URL includes `?ws_port=8765`.
- Browser reconnect logic exists.
- Frame receiver keeps only the newest frame.
- Stale-frame handling prevents processing frozen input.
- Rendered preview frames are sent back to the phone.

Risks:

- Mobile browser camera permission may fail over LAN HTTP.
- Firewall can block WebSocket port even if HTTP loads.
- HTTPS/WSS is not implemented.

## Vision Layer

Files:

- `vision/hand_tracker.py`
- `vision/gesture_detector.py`
- `vision/gesture_features.py`
- `models/test_gesture_model.py`
- `models/train.py`
- `models/train_landmarks.py`
- `models/train_cnn.py`

Status:

- Runtime uses normalized landmark vectors.
- Runtime prefers `models/gesture_model.pkl`.
- Confidence and margin thresholds reduce ambiguous actions.
- Shot gestures have stricter thresholds.
- `V_Sign` and `Gun_Sign` are separate model classes but map to one game action.

Recommendation:

- Always test gestures in `models/test_gesture_model.py` before full-game debugging.
- Improve dataset balance if the full game misreads Open Palm or Fist as Shot.

## Board Tracking Layer

Files:

- `ar/plane_tracker.py`
- `ar/homography.py`

Status:

- Primary board is a single 150 mm gate marker.
- Marker validates square border, central ring, and direction stem.
- Legacy L-marker code is preserved but no longer primary.
- Initial detection uses canonical patch validation.
- Registered tracking uses LK optical flow and RANSAC homography.
- Confidence controls smoothing, hold-last-pose, and re-detection.

Demo advice:

- Use a bold black marker on white A4.
- Keep the whole square in view during initial registration.
- Use debug mode only for diagnosis; normal gameplay should stay clean.

## AR Rendering Layer

Files:

- `ar/ar_renderer.py`
- `ar/pyrender_renderer.py`
- `ar/model_loader.py`

Status:

- Ground GLB is rendered to a cached top-down texture and warped onto the board.
- Enemy GLB is rendered with pyrender using `solvePnP` pose.
- Enemy render is downscaled and reused between frames.
- Enemy floats using transform-based sine bobbing.
- Enemy is hidden during wave clear and reward select.
- Player model is removed.

Risks:

- Requires working local OpenGL/offscreen rendering.
- GLB animations are not played yet.
- Mixed-source models may face the wrong direction.

Recommended next render improvement:

- Add per-enemy render settings to `EnemyType`:
  - `model_scale`
  - `ground_scale`
  - `yaw_degrees`
  - `pitch_degrees`
  - `roll_degrees`
  - `height_offset`
  - `bob_amplitude`
  - `bob_speed`

## Game Layer

Files:

- `game/game_manager.py`
- `game/battle_system.py`
- `game/wave_manager.py`
- `game/enemy.py`
- `game/player.py`
- `game/reward_system.py`
- `game/augment_system.py`
- `game/skills.py`

Status:

- Combat is simultaneous card reveal, not alternating enemy/player turns.
- Player holds a gesture to choose a card.
- Enemy chooses an action from HP-based probabilities.
- Round resolves damage, heal, block, miss, crit, and augment events.
- Wave clear enters reward selection.
- Defeat updates best wave and supports OK-hold run restart.

Player base stats:

- Max HP: `100`
- Attack power: `15`
- Strike: attack power plus Strike bonus
- Shot: attack power plus Shot bonus, doubled
- Guard: missing-HP based healing, minimum 5

Wave system:

- Difficulty multiplier: `1.15 ** (wave - 1)`.
- Enemy HP and damage use the multiplier.
- Dragon has `min_wave=4`.

Reward and augment system:

- Rewards are defined in `game/reward_system.py`.
- Augment logic is in `game/augment_system.py`.
- Already-owned augments are excluded from future reward choices.
- Augments are hook-style and can modify heal, shot chance, round resolution, or wave start.

Known issue:

- Some Korean augment labels in `game/augment_system.py` are mojibake and should be cleaned before final presentation.

## UI Layer

Files:

- `ui/action_cards.py`
- `ui/hud.py`
- `ui/damage_text.py`

Status:

- Main gameplay UI is board-relative AR UI.
- Player info and action cards are below the board.
- Reward cards appear near the former enemy position.
- Enemy HP/probability hint is near the enemy.
- Gesture probability and augment badges are on the board side.
- Defeat panel is screen-space and shows OK-hold border progress.

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

Manual test points:

- QR page opens on phone.
- WebSocket connects and frame count increases.
- Rendered preview appears on phone.
- Gate marker registers after OK hold.
- Game starts without pressing Space.
- Fist/Open Palm/V/Gun select expected cards.
- Simultaneous reveal occurs.
- Wave clear opens reward selection.
- Owned augments are not offered again.
- Dragon does not appear before wave 4.
- Defeat screen restarts run after OK hold.
- `D` toggles debug overlays.
- `R` performs hard reset.

## Next Recommended Tasks

High priority:

- Add per-enemy model orientation/scale configuration.
- Clean Korean text encoding in augment labels.
- Add visible warning if `pyrender` is unavailable.
- Tune gate marker validation with real camera samples.

Medium priority:

- Add simple hit/attack/idle transform animations.
- Add persistent best-wave storage.
- Add a final demo marker drawing guide.
- Add model asset checklist.

Low priority:

- HTTPS/WSS support.
- Real GLB animation playback.
- Camera calibration workflow.
- Revisit CNN gesture classifier path.
