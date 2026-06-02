import cv2
import numpy as np
from ar.homography import HomographyEstimator

class ARRenderer:
    """AR 오버레이 렌더링"""
    
    def __init__(self, plane_size=(400, 300)):
        self.plane_width, self.plane_height = plane_size
    
    def render_battlefield(self, frame, H, player_pos, enemy_pos, game_state=None):
        """
        AR 전장 렌더링
        
        Args:
            frame: BGR 입력 프레임
            H: 호모그래피 행렬
            player_pos: (x, y) 플레이어 위치 (평면 좌표)
            enemy_pos: (x, y) 적 위치 (평면 좌표)
            game_state: 게임 상태 dict
            
        Returns:
            frame: 렌더링된 프레임
        """
        if H is None:
            return frame
        
        # 1. 격자 그리기 (배경)
        frame = HomographyEstimator.draw_grid_on_plane(frame, H, grid_size=50, color=(50, 50, 50))
        
        # 2. 플레이어 렌더링
        player_screen = HomographyEstimator.transform_point(player_pos, H)
        if 0 <= player_screen[0] < frame.shape[1] and 0 <= player_screen[1] < frame.shape[0]:
            cv2.circle(frame, player_screen, 15, (0, 0, 255), -1)  # 빨강 (플레이어)
            cv2.putText(frame, 'P', (player_screen[0]-5, player_screen[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 3. 적 렌더링
        enemy_screen = HomographyEstimator.transform_point(enemy_pos, H)
        if 0 <= enemy_screen[0] < frame.shape[1] and 0 <= enemy_screen[1] < frame.shape[0]:
            cv2.circle(frame, enemy_screen, 15, (0, 255, 0), -1)  # 초록 (적)
            cv2.putText(frame, 'E', (enemy_screen[0]-5, enemy_screen[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 4. 평면 영역 표시 (선택사항)
        corners = np.array([[0, 0], [self.plane_width, 0], 
                           [self.plane_width, self.plane_height], 
                           [0, self.plane_height]], dtype=np.float32)
        corners_screen = np.array([HomographyEstimator.transform_point(tuple(c), H) for c in corners])
        cv2.polylines(frame, [corners_screen.astype(np.int32)], True, (255, 255, 0), 2)
        
        return frame
