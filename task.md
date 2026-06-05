# VisionQuest Task Notes

Last updated: 2026-06-05

## Current Focus

Improve demo completeness while keeping the CV pipeline stable:

- reliable A4 board registration/tracking
- clean gameplay view
- textured enemy/ground AR rendering
- understandable gesture-driven battle flow

## Completed Recently

- Replaced plain A4-only tracking with hand-drawn dark corner mark tracking.
- Added homography confidence, reprojection error checks, and EMA smoothing.
- Added partial occlusion handling.
- Changed missing marker display to use homography-projected world marker positions.
- Added hand occlusion masks to reject marker candidates under the detected hand.
- Added debug overlay toggle with `D`.
- Hidden marker/homography/FPS debug text during gameplay by default.
- Removed decorative board corner pillars.
- Removed player model rendering.
- Added enemy-specific ground model path.
- Added GLB/GLTF/OBJ rendering through `trimesh` + `pyrender`.
- Added OpenCV alpha blending of pyrender RGBA output.
- Fixed GLB scene graph loading for multi-mesh assets.
- Fixed pyrender camera clipping by setting a larger `zfar`.
- Converted GLB/GLTF Y-up assets to board Z-up.
- Added enemy floating/bobbing motion.
- Switched current enemy/ground paths in `game/wave_manager.py` to `.glb`.

## Validation Checklist

Run:

```bash
python main.py
```

Check:

- phone page connects and sends frames
- A4 corner marks are detected
- `SPACE` registers the board without returning to QR setup
- `OK_Sign` or second `SPACE` starts game
- debug overlays are hidden after game start
- `D` toggles debug overlays
- ground GLB appears on board
- enemy GLB appears above ground
- enemy floats slowly up/down
- `R` resets the game

Model-only check:

```bash
python models/test_gesture_model.py
```

## Known Issues / Risks

- HTTPS/WSS is not implemented.
- `pyrender` depends on local OpenGL/offscreen rendering support.
- GLB animation clips are not played.
- Camera calibration is approximate.
- Board pose quality still depends on marker visibility and homography stability.
- Gesture accuracy depends on landmark dataset quality.

## Next Useful Work

- Add per-enemy render parameters:
  - `model_scale`
  - `ground_scale`
  - `yaw`
  - `height_offset`
  - `bob_amplitude`
  - `bob_speed`
- Add a clear warning overlay if `pyrender` is unavailable.
- Add simple attack/idle transform animations.
- Add optional real GLB animation playback later.
- Tune marker ROI and confidence thresholds after live testing.
- Improve gesture dataset balance for `Open_Palm` and `OK_Sign`.
