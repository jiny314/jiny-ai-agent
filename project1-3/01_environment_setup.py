import os
import sys
import platform
import matplotlib.pyplot as plt
import seaborn as sns

# 1. OS 독립적 경로 설정 (pathlib 활용)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. OS별 한글 폰트 설정 및 마이너스 기호 깨짐 방지
def setup_korean_font():
    os_name = platform.system()
    if os_name == "Darwin":  # macOS
        plt.rc("font", family="AppleGothic")
    elif os_name == "Windows":  # Windows
        plt.rc("font", family="Malgun Gothic")
    else:  # Linux / Colab 등
        plt.rc("font", family="NanumGothic")

    # 마이너스 폰트 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[SUCCESS] 한글 폰트 설정 완료 (OS: {os_name})")


if __name__ == "__main__":
    setup_korean_font()
    print(f"[SUCCESS] 프로젝트 디렉토리 준비 완료: {DATA_DIR}")