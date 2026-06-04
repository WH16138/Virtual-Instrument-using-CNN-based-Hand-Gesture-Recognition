# VisionQuest Task Tracker

Last updated: 2026-06-05

Status legend:

- `TODO`: not started
- `DOING`: actively being worked on
- `VERIFY`: implemented but needs real runtime validation
- `DONE`: implemented and validated enough for current scope
- `BLOCKED`: cannot proceed without a decision, dependency, or environment change

Priority legend:

- `P0`: required for the demo to work
- `P1`: important for MVP quality
- `P2`: useful but not required for the immediate demo

## Current Focus

Stabilize the vision pipeline before expanding gameplay.

The immediate project risk is not game logic. It is whether streamed camera frames can produce reliable hand landmarks, gesture classifications, plane registration, and AR overlay in the full loop.

## P0 Tasks

| Status | Task | Evidence / Done Criteria |
| --- | --- | --- |
| VERIFY | Confirm phone camera stream works end-to-end in `main.py` | Phone page shows `WebSocket: connected`, `Frames sent` increases, PC OpenCV window shows live phone stream for several minutes. |
| DONE | Show QR and mobile URL in `main.py` waiting screen | When no camera frame is connected, the OpenCV window displays the generated QR code, exact URL, and connection instructions. |
| VERIFY | Confirm hand detection works from streamed phone frames | `main.py` HUD shows `Hands: 1` or `Hands: 2` while a hand is visible; hand landmarks are drawn on the PC frame. |
| VERIFY | Confirm standalone model tester works | `python models/test_gesture_model.py` opens camera, draws landmarks, and updates class/confidence values. |
| DONE | Unify gesture model input format | `models/train.py`, `vision/dataset_capture.py`, `vision/dataset_collector.py`, and `vision/gesture_detector.py` now use normalized 63-value landmark feature vectors. |
| TODO | Recollect landmark-vector dataset | Collect fresh `.npy` samples for `Fist`, `Open_Palm`, and `V_Sign` into `dataset_landmarks/` using `python vision/dataset_capture.py`. Keep legacy PNG files in `dataset/` for future CNN experiments. |
| TODO | Retrain landmark-vector gesture model | Run `python models/train.py --mode landmarks`; it should save `models/gesture_model.pkl` and show acceptable per-class recall, especially for `Open_Palm`. |
| DONE | Separate landmark and CNN dataset/training programs | Landmark path uses `dataset_landmarks` with `dataset_capture_landmarks.py`/`train_landmarks.py`; CNN path uses `dataset` with `dataset_capture_cnn.py`/`train_cnn.py`. |
| DONE | Add combined dataset capture | `python vision/dataset_capture.py` defaults to `--mode both` and saves PNG crops plus landmark `.npy` features from the same capture event. |
| DONE | Keep CNN PNG samples clean | CNN capture saves crops from a clean camera frame before drawing UI or landmarks. |
| VERIFY | Remove visual flicker in the OpenCV window | One `cv2.imshow()` and one `cv2.waitKey()` path per loop in normal streaming mode; FPS/HUD text should not flicker. |
| TODO | Clean critical broken HUD strings | User-facing text in `main.py` and `ui/hud.py` is readable during demo. |
| VERIFY | Confirm manual plane registration | Pressing `SPACE` on a textured surface prints/registers success and sets `plane_registered`. |
| TODO | Confirm AR overlay appears after registration and game start | Battlefield grid/player/enemy overlay appears when homography is valid and `game_started` is true. |

## P1 Tasks

| Status | Task | Evidence / Done Criteria |
| --- | --- | --- |
| TODO | Add explicit debug modes to `main.py` | Modes for camera-only, hand-tracking-only, gesture-only, and full game can be selected by flag or key. |
| VERIFY | Improve MediaPipe performance | Vision processing now runs every `VISION_INTERVAL_FRAMES` frames and display updates remain live; validate FPS on phone stream. |
| TODO | Add clear network diagnostics | PC or browser clearly reports HTTP URL, WebSocket URL, client connected, last frame age, and frame size. |
| TODO | Add HTTPS/WSS option | Phone camera works without insecure-origin browser flags. |
| TODO | Improve AR registration guidance | HUD tells user to aim at a textured flat surface and press `SPACE`; failure reason is visible. |
| TODO | Add basic integration checklist script or manual test doc | Demo operator can verify camera, WebSocket, hand detection, gesture, plane, AR in order. |
| TODO | Standardize encoding | Markdown and Python source strings are UTF-8 or ASCII; no corrupted demo-visible text remains. |

## P2 Tasks

| Status | Task | Evidence / Done Criteria |
| --- | --- | --- |
| TODO | Add model evaluation metrics | Training script outputs validation accuracy/confusion matrix for the three gestures. |
| TODO | Add optional CNN runtime detector | `models/gesture_model_cnn.keras` can be tested in gameplay only after a CNN detector or detector mode switch is implemented. |
| DONE | Remove TensorFlow from default runtime path | `main.py` uses `GestureDetector()` which prefers `models/gesture_model.pkl`; TensorFlow is only imported for optional `.keras` loading/CNN paths. |
| TODO | Improve dataset management | Dataset counts, capture method, and preprocessing format are documented and reproducible. |
| TODO | Add sound or visual combat effects | Only after P0 vision pipeline is stable. |
| TODO | Add multiple enemies or boss behavior | Only after MVP demo is stable. |
| TODO | Add hand-based target selection | Stretch goal after reliable tracking and gesture classification. |

## Completed Work Log

| Date | Work |
| --- | --- |
| 2026-06-05 | Added `models/test_gesture_model.py` for standalone model/camera testing. |
| 2026-06-05 | Updated README and handover to reflect smartphone streaming, current risks, and validation flow. |
| 2026-06-05 | Added `architecture.md` and `task.md` for project structure and task tracking. |
| 2026-06-05 | Added stale frame detection in `network.FrameReceiver`. |
| 2026-06-05 | Added WebSocket port query parameter and browser reconnect logic. |
| 2026-06-05 | Reduced mobile upload frame width for performance. |
| 2026-06-05 | Removed mid-loop debug display that caused flickering. |
| 2026-06-05 | Restored missing `cv2` import in `vision/hand_tracker.py`. |
| 2026-06-05 | Unified gesture training/inference around normalized landmark-vector `.npy` samples. |
| 2026-06-05 | Added lightweight frame/drop control in `main.py` to preserve smoother display updates. |
| 2026-06-05 | Split gesture data/training into landmark MLP and legacy PNG/CNN options. |
| 2026-06-05 | Added combined capture mode that saves both CNN PNG samples and landmark MLP samples. |
| 2026-06-05 | Added QR code and mobile URL display to the `main.py` camera waiting screen. |

## Validation Commands

Python compile check:

```bash
python -m py_compile main.py network/websocket_server.py network/frame_receiver.py network/qr_generator.py
python -m py_compile models/test_gesture_model.py vision/hand_tracker.py vision/gesture_detector.py
```

Standalone model check:

```bash
python models/test_gesture_model.py
```

Full app check:

```bash
python main.py
```

Manual phone check:

1. Open `http://<pc-ip>:8000/?ws_port=8765` on the phone.
2. Confirm camera preview.
3. Confirm `WebSocket: connected`.
4. Confirm `Frames sent` increases.
5. Confirm desktop OpenCV window shows the stream.

## Maintenance Rules

- Update this file when a task changes status.
- Add a done criterion before marking a task `DONE`.
- Prefer marking uncertain work as `VERIFY` rather than `DONE`.
- Keep `README.md` focused on usage.
- Keep `Project_handover.md` focused on current state, risks, and next engineer context.
- Keep `architecture.md` focused on structure and contracts.
- Keep `task.md` focused on actionable work and validation.
