from pathlib import Path
import sys
import pandas as pd

# 1. 동적 경로 설정 및 데이터 파일 검증
BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "data" / "소비자.csv"

if not csv_path.exists():
    print(f"[오류] 데이터 파일이 존재하지 않습니다: {csv_path}")
    print("data/ 폴더에 '소비자.csv' 파일을 위치시킨 후 다시 실행해 주세요.")
    sys.exit(1)

# 2. 원본 데이터 불러오기 (인코딩 예외 처리)
try:
    df = pd.read_csv(csv_path, encoding="utf-8")
except (UnicodeDecodeError, Exception):
    try:
        df = pd.read_csv(csv_path, encoding="cp949")
    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        sys.exit(1)

# 3. 이상치/노이즈 데이터 파악
print("=" * 70)
print("1. 연령대(age) 범주 값 이상치 확인")
print("=" * 70)
print(df["age"].value_counts(dropna=False))

print("\n" + "=" * 70)
print("2. 투자 성향(investment_propensity) 미응답/표기 노이즈 확인")
print("=" * 70)
print(df["investment_propensity"].value_counts(dropna=False))

print("\n" + "=" * 70)
print("3. 소득 구간(income_bracket) 표기 오류 이상치 확인")
print("=" * 70)
print(df["income_bracket"].value_counts(dropna=False))