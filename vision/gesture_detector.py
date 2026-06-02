import numpy as np
import tensorflow as tf
import cv2
from collections import deque

class GestureDetector:
    """CNN 모델을 이용한 제스처 분류"""
    
    def __init__(self, model_path='models/gesture_model.keras'):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True
        except:
            print(f"⚠️ 모델을 로드할 수 없습니다: {model_path}")
            self.model = None
            self.model_loaded = False
        
        self.gesture_classes = ['Fist', 'Open_Palm', 'V_Sign']
        self.smoothing_window = deque(maxlen=3)  # 최근 3프레임 평균
    
    def preprocess_hand_landmarks(self, landmarks):
        """
        손 랜드마크를 CNN 입력으로 변환
        
        Args:
            landmarks: (21, 3) 배열
            
        Returns:
            (64, 64, 3) 또는 None
        """
        if landmarks is None:
            return None
        
        try:
            # 1. 정규화: 손의 중심을 원점으로
            center = landmarks.mean(axis=0)
            normalized = landmarks - center
            
            # 2. 스케일 정규화
            max_dist = np.max(np.linalg.norm(normalized, axis=1))
            if max_dist > 0:
                normalized = normalized / max_dist
            
            # 3. 64x64 이미지로 변환
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # 손 포인트를 이미지에 그리기
            for i, (x, y, z) in enumerate(normalized):
                # 정규화된 좌표(-1~1)를 이미지 좌표(0~64)로 변환
                img_x = int((x + 1) * 32)
                img_y = int((y + 1) * 32)
                
                img_x = np.clip(img_x, 0, 63)
                img_y = np.clip(img_y, 0, 63)
                
                # 포인트 그리기 (초록색)
                cv2.circle(img, (img_x, img_y), 2, (0, 255, 0), -1)
            
            # 4. 연결선 그리기 (손의 구조)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # 엄지
                (0, 5), (5, 6), (6, 7), (7, 8),  # 검지
                (0, 9), (9, 10), (10, 11), (11, 12),  # 중지
                (0, 13), (13, 14), (14, 15), (15, 16),  # 약지
                (0, 17), (17, 18), (18, 19), (19, 20)  # 소지
            ]
            
            for start, end in connections:
                x1, y1, z1 = normalized[start]
                x2, y2, z2 = normalized[end]
                
                img_x1 = int((x1 + 1) * 32)
                img_y1 = int((y1 + 1) * 32)
                img_x2 = int((x2 + 1) * 32)
                img_y2 = int((y2 + 1) * 32)
                
                img_x1 = np.clip(img_x1, 0, 63)
                img_y1 = np.clip(img_y1, 0, 63)
                img_x2 = np.clip(img_x2, 0, 63)
                img_y2 = np.clip(img_y2, 0, 63)
                
                cv2.line(img, (img_x1, img_y1), (img_x2, img_y2), (255, 0, 0), 1)
            
            return img / 255.0  # 정규화
        
        except Exception as e:
            print(f"전처리 오류: {e}")
            return None
    
    def detect_gesture(self, hand_landmarks):
        """
        제스처 분류
        
        Args:
            hand_landmarks: (21, 3) 배열 또는 None
            
        Returns:
            dict: {
                'gesture': 제스처명,
                'confidence': 신뢰도 (0~1),
                'smoothed_gesture': 스무딩된 제스처
            }
        """
        if hand_landmarks is None or not self.model_loaded:
            return {
                'gesture': 'Unknown',
                'confidence': 0.0,
                'smoothed_gesture': 'Unknown'
            }
        
        try:
            # 전처리
            img = self.preprocess_hand_landmarks(hand_landmarks)
            if img is None:
                return {
                    'gesture': 'Unknown',
                    'confidence': 0.0,
                    'smoothed_gesture': 'Unknown'
                }
            
            # 추론
            img_batch = np.expand_dims(img, axis=0)
            predictions = self.model.predict(img_batch, verbose=0)
            confidence = float(np.max(predictions))
            gesture_idx = np.argmax(predictions[0])
            gesture = self.gesture_classes[gesture_idx]
            
            # 스무딩 (최근 3프레임 평균)
            self.smoothing_window.append(gesture)
            smoothed_gesture = max(set(self.smoothing_window), key=list(self.smoothing_window).count)
            
            return {
                'gesture': gesture,
                'confidence': confidence,
                'smoothed_gesture': smoothed_gesture
            }
        
        except Exception as e:
            print(f"제스처 인식 오류: {e}")
            return {
                'gesture': 'Unknown',
                'confidence': 0.0,
                'smoothed_gesture': 'Unknown'
            }
