# VisionQuest

**VisionQuest**는 스마트폰 카메라, 손 제스처 인식, 커스텀 마커 기반 AR 보드 추적을 결합한 실시간 AR 던전 전투 게임입니다.  
대학교 컴퓨터 비전 텀프로젝트 제출용으로, 카메라 입력부터 제스처 분류, 평면 추적, 3D 렌더링, 게임 상호작용까지 하나의 실행 프로그램으로 동작하도록 구성했습니다.

Last updated: 2026-06-07

## 1. Project Overview

VisionQuest의 전체 파이프라인은 다음과 같습니다.

```text
Phone camera
  -> WebSocket JPEG streaming
  -> OpenCV frame loop
  -> MediaPipe hand landmarks
  -> landmark gesture classifier
  -> custom 150 mm gate marker tracking
  -> homography / solvePnP pose estimation
  -> pyrender GLB enemy and ground rendering
  -> AR card battle UI
  -> optional rendered preview back to phone
```

핵심 목표는 **별도 AR 장비 없이 A4 용지 위에 직접 그린 마커를 전장으로 사용하고, 손 제스처만으로 게임을 진행하는 것**입니다.

## 2. Demo / Screenshots

제출 전 실제 실행 화면을 아래 위치에 추가하는 것을 권장합니다.

| Scene | Recommended Path | Description |
|---|---|---|
| QR setup | `assets/docs/01_qr_setup.png` | PC 화면에 QR과 연결 주소가 표시되는 초기 화면 |
| Board registration | `assets/docs/02_board_registration.png` | gate marker를 감지하고 `OK_Sign` 홀드로 등록하는 화면 |
| Combat | `assets/docs/03_combat.png` | AR 보드, 적 모델, 플레이어 카드 UI가 함께 보이는 전투 화면 |
| Reward select | `assets/docs/04_reward_select.png` | 웨이브 클리어 후 보상 카드 3장이 표시되는 화면 |
| Defeat restart | `assets/docs/05_defeat_restart.png` | 패배 후 `OK_Sign` 홀드로 게임을 재시작하는 화면 |

가능하면 30~60초 분량의 짧은 데모 영상 링크를 함께 첨부하면 평가자가 실행 전 흐름을 빠르게 파악할 수 있습니다.

## 3. Requirements

Python 3.10 기준으로 테스트했습니다.

```bash
pip install -r requirements.txt
```

주요 라이브러리:

- `opencv-python`: frame processing, marker detection, homography, solvePnP
- `mediapipe`: hand landmark detection
- `scikit-learn`: landmark-vector MLP gesture classifier runtime
- `websockets`: phone-to-PC camera streaming
- `qrcode`: QR connection helper
- `trimesh`, `pyrender`, `PyOpenGL`: GLB/GLTF model rendering
- `pygame`: PC-side BGM/SFX playback

필수 런타임 파일:

```text
models/hand_landmarker.task
models/gesture_model.pkl
assets/
```

`main.py`는 TensorFlow DLL 문제를 피하기 위해 기본적으로 `.pkl` gesture model을 사용합니다. CNN/학습용 `.keras` 파일과 raw dataset은 재학습용이며 최종 실행에는 필수는 아닙니다.

## 4. How To Run

```bash
python main.py
```

실행하면 `main.py`가 HTTP 서버와 WebSocket 서버를 함께 시작합니다.

```text
HTTP camera page: http://<PC_IP>:8000/?ws_port=8765
WebSocket frames: ws://<PC_IP>:8765
```

실제 사용 순서:

1. PC와 스마트폰을 같은 네트워크에 연결합니다.
2. PC에서 `python main.py`를 실행합니다.
3. PC 화면에 표시되는 QR 코드를 스마트폰으로 스캔합니다.
4. 스마트폰 브라우저에서 카메라 권한을 허용합니다.
5. A4 용지 위에 그린 150 mm gate marker를 카메라에 비춥니다.
6. `OK_Sign`을 2초간 유지하면 보드가 등록되고 게임이 시작됩니다.
7. 전투 중에는 `Fist`, `Open_Palm`, `V_Sign` 또는 `Gun_Sign`을 2초간 유지해 카드를 선택합니다.
8. 패배 후에는 `OK_Sign`을 2초간 유지해 보드 등록은 유지한 채 게임만 재시작합니다.

키보드 입력은 디버깅/종료용입니다.

```text
Q      quit program
R      hard reset, including board registration
D      toggle debug overlays
```

일반 플레이 흐름은 스마트폰 카메라와 손 제스처만으로 진행되도록 설계했습니다.

## 5. Marker Design

기본 마커는 A4 용지 중앙에 직접 그리는 **single gate marker**입니다. 마커 자체가 게임 보드입니다.

권장 치수:

- 검은색 속 빈 정사각형: 약 `15 cm x 15 cm`
- 중앙의 속 빈 원 또는 링
- 중앙 원에서 아래 방향으로 내려오는 짧은 직선
- 직선은 아래 테두리에 닿지 않고 중간에서 끝나야 합니다.

이 구조를 선택한 이유:

- 큰 사각형 외곽은 네 모서리와 원근 변환을 안정적으로 제공합니다.
- 중앙 링은 책상 모서리, 일반 사각형, 그림자 같은 오검출을 줄이는 보조 검증 요소입니다.
- 아래 방향 stem은 보드의 회전 방향을 구분합니다.
- 체스보드보다 게임 분위기에 자연스럽고, 네 개의 작은 L 마커보다 검출이 안정적입니다.

## 6. Computer Vision Techniques

### Phone Camera Streaming

스마트폰은 브라우저에서 카메라 프레임을 JPEG로 압축해 WebSocket으로 PC에 전송합니다. PC는 최신 프레임만 처리해 stale frame 누적을 줄이고, 렌더링된 결과 프레임을 다시 휴대폰으로 전송할 수 있습니다.

### Gesture Recognition

MediaPipe hand landmark를 사용해 손의 3D landmark vector를 추출하고, 학습된 MLP classifier로 제스처를 분류합니다.

```text
Fist       -> Strike
Open_Palm  -> Guard
V_Sign     -> Shot
Gun_Sign   -> Shot
OK_Sign    -> setup / restart
```

`V_Sign`과 `Gun_Sign`은 학습 시 별도 클래스로 두어 정확도를 높이고, 게임에서는 같은 `Shot` 카드로 매핑합니다. 오인식을 줄이기 위해 top probability와 class margin을 함께 검사하며, 특히 `Shot` 계열은 더 엄격한 threshold를 사용합니다.

독립 테스트:

```bash
python models/test_gesture_model.py
```

### Gate Marker Tracking

마커 추적은 STag 및 planar fiducial marker 연구에서 사용되는 안정적인 코너 검출, 보조 심볼 검증, confidence 기반 추적 아이디어를 참고했습니다. 단, 실제 구현은 손으로 그릴 수 있는 단일 gate marker에 맞게 직접 구성했습니다.

```text
dark contour / edge candidate
  -> quadrilateral validation
  -> canonical perspective patch
  -> border continuity validation
  -> central ring validation
  -> stem orientation validation
  -> homography and confidence score
  -> solvePnP pose
  -> optical-flow tracking and periodic re-detection
```

주요 안정화 기법:

- downscaled frame detection으로 초기 검출 비용 감소
- normalized marker patch에서 외곽, 링, stem을 검증
- confidence score와 EMA smoothing으로 흔들림 완화
- LK optical flow와 RANSAC homography로 등록 후 추적 유지
- 스마트폰 회전 등 해상도 변경 시 tracking cache reset
- debug mode에서만 후보/진단 overlay 표시

### AR Rendering

2D 바닥은 homography로 보드 평면에 맞추고, 3D 적 모델은 `solvePnP` pose를 기반으로 `pyrender`에서 RGBA frame으로 렌더링한 뒤 OpenCV frame에 alpha blending합니다.

```text
board corners
  -> homography for board-space UI
  -> solvePnP rvec/tvec
  -> pyrender offscreen rendering
  -> OpenCV alpha blend
```

모델 포맷:

- 권장: `.glb` in `assets/models/`
- `.obj`는 이전 fallback 용도로만 유지 가능

좌표 변환:

```text
game_x = asset_x
game_y = asset_z
game_z = asset_y
```

일반적인 Y-up GLB asset을 보드 기준 Z-up 좌표계로 맞추기 위한 변환입니다. 개별 모델의 정면 방향은 asset마다 다를 수 있어 필요 시 타입별 보정값을 둘 수 있습니다.

## 7. Game System

게임은 무한 웨이브 방식의 던전 전투입니다.

```text
camera setup
  -> OK hold board registration/start
  -> wave intro
  -> player card hold
  -> simultaneous reveal
  -> round resolution
  -> wave clear
  -> reward select
  -> next wave
  -> defeat
  -> OK hold restart
```

전투는 플레이어와 적이 카드를 동시에 공개하는 구조입니다.

| Player Card | Gesture | Meaning |
|---|---|---|
| Strike | `Fist` | 기본 공격 |
| Guard | `Open_Palm` | 회복/방어 계열 선택 |
| Shot | `V_Sign` or `Gun_Sign` | 고위험 고화력 스킬 |

플레이어 기본 스탯:

```text
Max HP: 100
Attack power: 15
Strike damage: attack_power + strike_bonus
Shot damage: (attack_power + shot_bonus) * 2
Guard heal: max(5, missing_hp * (10% + guard ratio bonus)) + flat bonus
```

난이도:

```text
Wave N multiplier = 1.15 ** (N - 1)
```

적 HP와 공격력은 웨이브 배율을 적용해 증가합니다.

## 8. Rewards and Augments

웨이브 클리어 후 3장의 보상 카드가 등장합니다. 보상 선택은 기존 제스처 카드 UX를 재사용하며, 잘못 선택되지 않도록 hold 기반으로 확정합니다.

보상 카테고리:

- `stat`
- `heal`
- `card_upgrade`
- `augment`

기본 보상 예시:

- 최대 체력 증가
- 공격력 증가
- 전체 피해 배율 증가
- Strike 피해 증가
- Guard 회복량 증가
- Shot 피해 증가

구현된 augment:

- Double Attack
- Cull the Weak
- Deep Rest
- Counter Guard
- Chicken Game
- Vampire
- Prepared
- Insurance
- First Strike

이미 획득한 augment는 이후 보상 풀에서 제외됩니다.

## 9. PC Audio

PC-side audio는 `pygame.mixer` 기반의 `audio/AudioManager`가 담당합니다. 오디오 파일이 없거나 `pygame` 초기화가 실패해도 게임은 계속 실행됩니다.

지원 구조:

```text
assets/audio/bgm/dungeon_*.mp3
assets/audio/bgm/setup_*.mp3
assets/audio/sfx/*.wav
```

BGM은 같은 카테고리의 여러 파일 중 하나를 랜덤으로 재생할 수 있습니다. SFX는 카드 포커스, 카드 확정, 공격, 방어, 피격, 보상, 증강 발동, 패배 등 게임 이벤트에 연결됩니다.

자세한 트리거 목록은 `assets/audio/README.md`를 참고하십시오.

## 10. Project Structure

```text
ar/                 marker tracking, homography, AR rendering
audio/              PC BGM/SFX manager
assets/             cards, GLB models, audio assets
game/               battle, wave, reward, augment, player/enemy state
models/             runtime gesture model and training scripts
network/            HTTP/WebSocket camera server and frame receiver
ui/                 AR-space cards, HUD, floating text
vision/             hand tracking, gesture feature extraction, dataset tools
web/                phone camera web page
main.py             main runtime loop
```

## 11. Training and Dataset Notes

최종 실행에는 raw dataset이 필요하지 않습니다.

데이터 수집/학습 관련 도구:

```text
vision/dataset_capture_both.py
vision/dataset_capture_cnn.py
models/train_landmarks.py
models/train_cnn.py
models/train.py
```

제출 zip에서는 `dataset/`, `dataset_landmarks/`를 제외합니다. 모델을 재학습해야 하는 경우 별도 저장소 또는 Google Drive 링크로 데이터셋을 제공할 수 있습니다.

## 12. References and Credits

라이브러리:

- OpenCV: image processing, contour analysis, homography, solvePnP, optical flow
- MediaPipe: hand landmark detection
- scikit-learn: MLP gesture classifier runtime
- trimesh / pyrender / PyOpenGL: GLB/GLTF model loading and offscreen rendering
- pygame: PC-side audio playback

참고 아이디어:

- STag: A Stable Fiducial Marker System
- Planar Fiducial Markers: A Comparative Study
- Designing Highly Reliable Fiducial Markers
- Fiducial Markers for Pose Estimation: Overview and Comparison

에셋:

- 현재 포함된 PNG, GLB, 오디오 에셋은 CC0 또는 자체 제작 에셋만 사용하는 제출 정책을 따릅니다.
- 자세한 기록은 `assets/ASSET_LICENSES.md`를 참고하십시오.

## 13. Known Limitations

- HTTPS/WSS는 구현하지 않았습니다. 같은 LAN에서 HTTP camera page를 사용하는 구조입니다.
- 카메라 intrinsic calibration은 근사값을 사용합니다.
- `pyrender`는 로컬 OpenGL/offscreen rendering 환경에 의존합니다.
- GLB animation clip 재생은 아직 지원하지 않습니다.
- raw dataset은 최종 제출 zip에서 제외됩니다.
- 실제 인식 성능은 조명, 마커 선명도, 손 제스처 데이터 균형에 영향을 받습니다.

## 14. Verification and Submission

정적 검증:

```bash
python -m py_compile main.py ar/*.py game/*.py ui/*.py vision/*.py network/*.py models/*.py audio/*.py
```

수동 실행 검증:

1. `python main.py` 실행
2. QR 접속 및 스마트폰 카메라 프레임 수신 확인
3. gate marker 등록과 `OK_Sign` 2초 시작 확인
4. 전투 카드 선택, 보상 선택, defeat restart 확인
5. BGM/SFX 재생 확인

제출 패키지 정책:

- 포함: source code, `models/hand_landmarker.task`, `models/gesture_model.pkl`, `assets/`, `requirements.txt`, `README.md`, `LICENSE`
- 제외: `.git/`, `dataset/`, `dataset_landmarks/`, `__pycache__/`, `*.pyc`, generated QR images, local zip files
- 생성된 제출용 zip 예시: `VisionQuest_submission_20260607.zip`

최종 제출은 e-Class에 Repository URL과 source zip을 함께 업로드합니다.
