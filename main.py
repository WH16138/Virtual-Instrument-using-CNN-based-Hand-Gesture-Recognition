import cv2
import numpy as np
from vision.hand_tracker import HandTracker
from vision.gesture_detector import GestureDetector
from ar.plane_tracker import PlaneTracker
from ar.ar_renderer import ARRenderer
from game.game_manager import GameManager
from game.battle_system import BattleState
from network.frame_receiver import FrameReceiver
from network.websocket_server import WebSocketFrameServer
from ui.hud import HUD


def main():
    # 초기화
    print("🎮 VisionQuest 초기화 중...")
    
    # 네트워크 기반 스마트폰 카메라 입력
    frame_receiver = FrameReceiver()
    websocket_server = WebSocketFrameServer(frame_receiver)
    websocket_server.start()
    
    # 모듈 초기화
    hand_tracker = HandTracker(max_num_hands=2)
    gesture_detector = GestureDetector('models/gesture_model.keras')
    plane_tracker = PlaneTracker()
    ar_renderer = ARRenderer(plane_size=(400, 300))
    game_manager = GameManager()
    
    # 게임 상태
    game_started = False
    plane_registered = False
    # 양손 자동 시작 관련
    both_hands_counter = 0
    both_hands_threshold = 15  # 연속 프레임 수 (조정 가능)
    require_open_palms = True  # True면 양손이 Open_Palm 제스처여야 시작
    frame_count = 0
    fps_clock = cv2.getTickCount()
    
    print("✓ 초기화 완료")
    print("\n💡 사용법:")
    print("  1. 평평한 배경을 카메라에 비추세요")
    print("  2. SPACE: 게임 시작 (평면 등록)")
    print("  3. 손을 보이고 제스처 인식 대기")
    print("  4. R: 재시작, Q: 종료\n")
    
    # 메인 루프
    while True:
        frame = frame_receiver.get_latest_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'Waiting for mobile camera...', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, 'Scan QR code or connect your phone', (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow('VisionQuest', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        frame_count += 1
        frame = cv2.flip(frame, 1)  # 좌우 반전
        height, width = frame.shape[:2]
        
        # FPS 계산
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_clock)
        fps_clock = cv2.getTickCount()
        
        # 1. 평면 추적
        if plane_registered:
            tracking_result = plane_tracker.track_plane(frame)
            H = tracking_result['H']
        else:
            H = None
        
        # 2. 손 감지
        hand_detection = hand_tracker.detect_hands(frame)
        
        # 3. 제스처 인식
        gesture_info_left = gesture_detector.detect_gesture(hand_detection['left_hand'])
        gesture_info_right = gesture_detector.detect_gesture(hand_detection['right_hand'])
        
        # 우측 손 제스처 우선 사용
        gesture_info = gesture_info_right if gesture_info_right['confidence'] > gesture_info_left['confidence'] else gesture_info_left
        
        # 4. 게임 로직 업데이트
        # 자동 시작: 평면이 등록되어 있고 게임이 아직 시작되지 않은 경우
        if plane_registered and not game_started:
            left_present = hand_detection.get('left_hand') is not None
            right_present = hand_detection.get('right_hand') is not None

            palms_ok = True
            if require_open_palms:
                palms_ok = (
                    gesture_info_left.get('gesture') == 'Open_Palm' and
                    gesture_info_right.get('gesture') == 'Open_Palm'
                )

            if left_present and right_present and palms_ok:
                both_hands_counter += 1
            else:
                both_hands_counter = 0

            # 화면에 대기 카운터 표시
            if both_hands_counter > 0:
                cv2.putText(frame, f'Both hands detected: {both_hands_counter}/{both_hands_threshold}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            if both_hands_counter >= both_hands_threshold:
                game_manager.start_game()
                game_started = True
                print("🎮 양손 감지 완료 — 게임 시작!")

        if game_started:
            game_manager.process_gesture(gesture_info)
            game_manager.update()
        
        # 5. 렌더링
        # AR 전장 렌더링
        if H is not None and game_started:
            frame = ar_renderer.render_battlefield(
                frame, H,
                game_manager.player_pos,
                game_manager.enemy_pos
            )
        
        # 손 랜드마크 그리기 (디버깅)
        frame = hand_tracker.draw_hands(frame, hand_detection)
        
        # HUD 그리기
        if game_started:
            game_state = game_manager.get_game_state()
            frame = HUD.draw_hp_bar(frame, game_manager.player, game_manager.enemy)
            frame = HUD.draw_game_state(frame, game_state)
            frame = HUD.draw_gesture_recognition(frame, gesture_info)
        
        frame = HUD.draw_instructions(frame, game_manager.get_game_state() if game_started else {'battle_state': BattleState.WAITING})
        
        # FPS 표시
        cv2.putText(frame, f'FPS: {fps:.1f}', (width - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 평면 등록 상태 표시
        status_text = "✓ 평면 등록됨" if plane_registered else "✗ 평면 미등록"
        cv2.putText(frame, status_text, (10, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if plane_registered else (0, 0, 255), 1)
        
        # 화면 표시
        cv2.imshow('VisionQuest', frame)
        
        # 키입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # 종료
            print("👋 프로그램 종료")
            break
        
        elif key == ord(' '):  # SPACE: 게임 시작/평면 등록
            if not game_started:
                if not plane_registered:
                    # 평면 등록
                    if plane_tracker.register_plane(frame):
                        plane_registered = True
                        print("✓ 평면 등록 성공. 이제 게임을 시작할 수 있습니다.")
                    else:
                        print("❌ 평면 등록 실패")
                else:
                    # 게임 시작
                    game_manager.start_game()
                    game_started = True
                    print("🎮 게임 시작!")

        elif key == ord('r'):  # R: 재시작
            if game_started:
                game_manager.reset_game()
                game_started = False
                plane_registered = False
                print("🔄 게임 재시작")
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
