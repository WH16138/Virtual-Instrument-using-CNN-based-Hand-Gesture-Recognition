import numpy as np
import cv2

class HomographyEstimator:
    """호모그래피 기반 좌표 변환"""
    
    @staticmethod
    def transform_point(point, H):
        """
        단일 점 변환
        
        Args:
            point: (x, y)
            H: 호모그래피 행렬
            
        Returns:
            tuple: (x', y')
        """
        if H is None:
            return point
        
        p = np.array([point[0], point[1], 1])
        p_transformed = H @ p
        x = int(p_transformed[0] / p_transformed[2])
        y = int(p_transformed[1] / p_transformed[2])
        return (x, y)
    
    @staticmethod
    def transform_points(points, H):
        """
        여러 점 변환
        
        Args:
            points: [(x1, y1), (x2, y2), ...]
            H: 호모그래피 행렬
            
        Returns:
            list: [(x1', y1'), (x2', y2'), ...]
        """
        if H is None:
            return points
        
        transformed = []
        for point in points:
            transformed.append(HomographyEstimator.transform_point(point, H))
        return transformed
    
    @staticmethod
    def draw_grid_on_plane(frame, H, grid_size=50, color=(0, 255, 0)):
        """
        호모그래피를 이용해 평면에 그리드 그리기
        
        Args:
            frame: BGR 입력 프레임
            H: 호모그래피 행렬
            grid_size: 그리드 간격 (픽셀)
            
        Returns:
            frame: 그리드가 그려진 프레임
        """
        if H is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # 수평선
        for y in range(0, height, grid_size):
            points = [(0, y), (width, y)]
            transformed = HomographyEstimator.transform_points(points, np.linalg.inv(H))
            if all(0 <= p[0] < width and 0 <= p[1] < height for p in transformed):
                cv2.line(frame, tuple(map(int, transformed[0])), tuple(map(int, transformed[1])), color, 1)
        
        # 수직선
        for x in range(0, width, grid_size):
            points = [(x, 0), (x, height)]
            transformed = HomographyEstimator.transform_points(points, np.linalg.inv(H))
            if all(0 <= p[0] < width and 0 <= p[1] < height for p in transformed):
                cv2.line(frame, tuple(map(int, transformed[0])), tuple(map(int, transformed[1])), color, 1)
        
        return frame
