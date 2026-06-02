#!/usr/bin/env python3
"""
MediaPipe HandLandmarker 모델 다운로드 스크립트
공식 Google 저장소 또는 Hugging Face에서 hand_landmarker.task 파일을 다운로드합니다.
"""

import urllib.request
from pathlib import Path
import sys

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# 여러 소스에서 시도할 URL들 (순서대로 시도)
MODEL_URLS = [
    ("Google Official Repository", "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
    ("Hugging Face (mirror)", "https://huggingface.co/google/mediapipe-models/resolve/main/hand_landmarker.task"),
    ("Google CDN 1", "https://storage.googleapis.com/mediapipe-models/tasks/hand_landmarker/hand_landmarker.task"),
]


def download_model():
    """모델 파일 다운로드 (여러 소스에서 시도)"""
    print("📥 MediaPipe HandLandmarker 모델 다운로드 중...\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in MODEL_URLS:
        print(f"시도 중: {name}")
        print(f"  URL: {url}")

        try:
            urllib.request.urlretrieve(
                url,
                MODEL_PATH,
                reporthook=_report_progress
            )
            print(f"\n✓ 다운로드 완료: {MODEL_PATH}")
            print(f"  파일 크기: {MODEL_PATH.stat().st_size / (1024 * 1024):.1f} MB\n")
            return True
        except Exception as exc:
            print(f"  ❌ 실패: {exc}\n")
            continue

    return False


def _report_progress(block_num, block_size, total_size):
    """다운로드 진행 상황 표시"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded * 100) // total_size)
        sys.stdout.write(f"\r  진행 중: {percent}% ({downloaded / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB)")
        sys.stdout.flush()


def print_manual_download_guide():
    """수동 다운로드 방법 안내"""
    print("\n" + "=" * 70)
    print("📖 수동 다운로드 방법")
    print("=" * 70)
    print("\n자동 다운로드에 실패했습니다. 다음 방법으로 수동 다운로드할 수 있습니다:\n")

    print("1️⃣  Hugging Face (권장)")
    print("   🔗 https://huggingface.co/google/mediapipe-models/blob/main/hand_landmarker.task")
    print("   - 위 링크에서 다운로드 버튼 클릭\n")

    print("2️⃣  Google Official (문서)")
    print("   🔗 https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python")
    print("   - 'Download model' 섹션 참조\n")

    print("3️⃣  다운로드 완료 후")
    print(f"   - 파일을 다음 폴더에 저장:\n     → {Path.cwd() / MODELS_DIR}\n")


if __name__ == "__main__":
    if MODEL_PATH.exists():
        print(f"✓ 모델 파일이 이미 존재합니다: {MODEL_PATH}\n")
        sys.exit(0)

    success = download_model()

    if not success:
        print_manual_download_guide()
        print("\n" + "=" * 70)
        sys.exit(1)

    sys.exit(0)
