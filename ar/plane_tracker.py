import cv2
import numpy as np

class PlaneTracker:
    """ORB 특징 감지 기반 평면 추적"""
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.is_registered = False
    
    def register_plane(self, frame):
        """
        평면 등록 (첫 프레임에서 특징점 추출)
        
        Args:
            frame: BGR 입력 프레임
            
        Returns:
            bool: 등록 성공 여부
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 20:
            print("⚠️ 특징점이 부족합니다. 다른 평면을 시도하세요.")
            return False
        
        self.reference_frame = gray.copy()
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors
        self.is_registered = True
        
        print(f"✓ 평면 등록 완료: {len(keypoints)}개 특징점 감지")
        return True
    
    def track_plane(self, frame):
        """
        평면 추적 및 호모그래피 계산
        
        Args:
            frame: BGR 입력 프레임
            
        Returns:
            dict: {
                'success': bool,
                'H': 호모그래피 행렬 또는 None,
                'matched_points': 매칭된 특징점 개수
            }
        """
        if not self.is_registered:
            return {'success': False, 'H': None, 'matched_points': 0}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 20:
            return {'success': False, 'H': None, 'matched_points': 0}
        
        # 특징점 매칭 (Lowe's ratio test)
        matches = self.bf_matcher.knnMatch(self.reference_descriptors, descriptors, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return {'success': False, 'H': None, 'matched_points': len(good_matches)}
        
        # 호모그래피 계산
        src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return {'success': False, 'H': None, 'matched_points': len(good_matches)}
        
        return {
            'success': True,
            'H': H,
            'matched_points': len(good_matches)
        }
