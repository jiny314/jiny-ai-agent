from pathlib import Path
import subprocess
import sys

# run_all.py 위치 기준 경로
BASE_DIR = Path(__file__).resolve().parent

# 로컬에 실제 존재하는 파이프라인 스크립트 리스트
scripts = [
    "1_preprocess.py",
    "2_eda_categorical_unique.py",
    "3_eda_missing_values.py",
    "10_clustering_evaluation.py",
    "11_persona_clustering.py",
    "12_association_rules.py",
    "13_networkx_visualization.py",
]

print("=" * 70)
print("=== [Mainproj] 데이터 분석 파이프라인 자동 실행을 시작합니다 ===")
print("=" * 70)

executed_count = 0

for script in scripts:
    target_path = BASE_DIR / script

    # 루트에 없으면 하위 src/ 또는 project1/ 폴더 검색
    if not target_path.exists():
        target_path = BASE_DIR / "src" / script
    if not target_path.exists():
        target_path = BASE_DIR / "project1" / script

    # 파일이 없는 경우 안내 후 통과
    if not target_path.exists():
        print(f"\n[건너뜀] 파일이 존재하지 않아 다음 단계로 넘어갑니다: {script}")
        continue

    executed_count += 1
    print(f"\n▶ [{executed_count}/{len(scripts)}] 실행 중: {target_path.name} ...")
    result = subprocess.run([sys.executable, str(target_path)])

    if result.returncode != 0:
        print(
            f"\n[오류 발생] {target_path.name} 실행 중 에러가 발생하여 파이프라인을 중단합니다."
        )
        sys.exit(1)

print("\n" + "=" * 70)
print(
    f"=== [완료] 총 {executed_count}개의 스크립트 실행이 성공적으로 실행되었습니다! ==="
)
print("=" * 70)