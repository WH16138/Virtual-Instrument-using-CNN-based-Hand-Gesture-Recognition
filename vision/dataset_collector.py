import cv2
import os
import numpy as np
from datetime import datetime

class DatasetCollector:
    """제스처 데이터셋 수집"""
    
    def __init__(self, dataset_dir='dataset'):
        self.dataset_dir = dataset_dir
        self.gesture_classes = ['Fist', 'Open_Palm', 'V_Sign']
        
        # 디렉토리 생성
        for gesture_class in self.gesture_classes:
            os.makedirs(f'{dataset_dir}/{gesture_class}', exist_ok=True)
    
    def save_gesture_image(self, gesture_label, hand_landmarks):
        """
        손 랜드마크를 이미지로 저장
        
        Args:
            gesture_label: 'Fist', 'Open_Palm', 'V_Sign'
            hand_landmarks: (21, 3) 배열
        """
        if gesture_label not in self.gesture_classes or hand_landmarks is None:
            return False
        
        try:
            # 이미지 생성 (gesture_detector와 동일한 방식)
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            
            center = hand_landmarks.mean(axis=0)
            normalized = hand_landmarks - center
            
            max_dist = np.max(np.linalg.norm(normalized, axis=1))
            if max_dist > 0:
                normalized = normalized / max_dist
            
            for i, (x, y, z) in enumerate(normalized):
                img_x = int((x + 1) * 32)
                img_y = int((y + 1) * 32)
                img_x = np.clip(img_x, 0, 63)
                img_y = np.clip(img_y, 0, 63)
                cv2.circle(img, (img_x, img_y), 2, (0, 255, 0), -1)
            
            # 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'{self.dataset_dir}/{gesture_label}/{gesture_label}_{timestamp}.png'
            cv2.imwrite(filename, img)
            return True
        
        except Exception as e:
            print(f"데이터셋 저장 오류: {e}")
            return False
    
    def get_dataset_stats(self):
        """데이터셋 통계"""
        stats = {}
        for gesture_class in self.gesture_classes:
            path = f'{self.dataset_dir}/{gesture_class}'
            if os.path.exists(path):
                count = len(os.listdir(path))
                stats[gesture_class] = count
        return stats
