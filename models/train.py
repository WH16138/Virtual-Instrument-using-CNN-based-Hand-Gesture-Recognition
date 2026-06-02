import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

class GestureModelTrainer:
    """제스처 인식 CNN 모델 학습"""
    
    def __init__(self, dataset_dir='dataset', model_output_path='models/gesture_model.keras'):
        self.dataset_dir = dataset_dir
        self.model_output_path = model_output_path
        self.gesture_classes = ['Fist', 'Open_Palm', 'V_Sign']
        self.class_to_idx = {gesture: idx for idx, gesture in enumerate(self.gesture_classes)}
    
    def load_dataset(self):
        """
        데이터셋 로드
        
        Returns:
            tuple: (X, y) 입력 이미지 배열, 라벨
        """
        X = []
        y = []
        
        for gesture_class in self.gesture_classes:
            class_dir = os.path.join(self.dataset_dir, gesture_class)
            if not os.path.exists(class_dir):
                print(f"⚠️ {class_dir}이 존재하지 않습니다")
                continue
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.png'):
                    filepath = os.path.join(class_dir, filename)
                    img = cv2.imread(filepath)
                    if img is not None:
                        # 64x64로 정규화
                        img = cv2.resize(img, (64, 64))
                        img = img / 255.0  # 정규화
                        X.append(img)
                        y.append(self.class_to_idx[gesture_class])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """CNN 모델 구축"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=50, batch_size=16, validation_split=0.2):
        """
        모델 학습
        
        Args:
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
        """
        # 데이터 로드
        print("📊 데이터셋 로드 중...")
        X, y = self.load_dataset()
        
        if len(X) == 0:
            print("❌ 학습 데이터가 없습니다. 먼저 dataset_collector로 데이터를 수집하세요.")
            return
        
        print(f"✓ 로드된 샘플 수: {len(X)}")
        
        # 데이터 증강
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ])
        
        # 모델 구축
        print("🤖 모델 구축 중...")
        model = self.build_model()
        model.summary()
        
        # 학습
        print("🎓 모델 학습 중...")
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # 모델 저장
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        model.save(self.model_output_path)
        print(f"✓ 모델 저장 완료: {self.model_output_path}")
        
        return model


# 사용 예시
if __name__ == '__main__':
    trainer = GestureModelTrainer()
    trainer.train(epochs=50, batch_size=16)
