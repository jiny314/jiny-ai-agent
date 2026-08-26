# 🤖 AI 에이전트 프로젝트 (AI Agent Repository)

본 레포지토리는 AI 에이전트 기반 분석 및 머신러닝 모듈을 프로젝트별로 체계적으로 관리하기 위해 구축되었습니다.

---

## 📌 프로젝트 목록 (Project List)

* **1차 프로젝트**: [project1] 초개인화 페르소나 기반 금융 상품 패키징 및 교차 판매(Cross-selling) 전략 도출
* **2차 프로젝트**: *(추후 `project2/` 폴더 생성 및 확장 예정)*

---

### 📑 Data Source & Citation
* **출처**: [한국지능정보사회진흥원(NIA) AI Hub — 「금융상품·서비스 및 소비자 특성 데이터」](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71938)
* **정책 준수**: 본 레포지토리는 AI Hub 이용 정책을 엄격히 준수하며, 원본 데이터 파일(`소비자.csv` 등)은 라이선스 정책상 포함되어 있지 않습니다.

---

## 📋 [1차 프로젝트] 개요 및 목적

* **프로젝트명**: 초개인화 페르소나 기반 금융 상품 패키징 및 교차 판매(Cross-selling) 전략 도출
* **배경 및 목적**: 
  • 대규모 금융 거래 데이터(38.5만 건)를 활용하여 비지도학습($K=3$ K-Means) 기반 **초개인화 페르소나**를 도출합니다.
  • 페르소나별 **FP-Growth 연관 규칙** 및 **NetworkX 연결 중심성 그래프** 분석을 적용하여 고객 유지율과 LTV를 극대화하는 **맞춤형 교차 판매(Cross-selling) 패키징 시나리오**를 수립합니다.

---

## 📁 프로젝트 폴더 구조 (Directory Structure)

1차 분석 스크립트 및 데이터 산출물은 모두 **`project1/`** 폴더 내부에서 독립적으로 관리됩니다.

```text
jiny-ai-agent/
│
├── project1/                         # 1차 프로젝트 전용 폴더
│   ├── data/                         # 원본 데이터 폴더 (.gitignore 제외)
│   │   └── 소비자.csv
│   │
│   ├── output/                       # 전처리 및 분석 결과 산출물 (.pkl, .csv)
│   │   ├── customer_processed.pkl
│   │   ├── customer_clustered_k3.pkl
│   │   └── association_rules_k3.csv
│   │
│   ├── 1_preprocess.py               # 01. 고객 단위 전처리 및 요약 집계
│   ├── 2_eda_categorical_unique.py   # 02. EDA 주요 변수 분포 및 TOP10 상품 시각화
│   ├── 3_eda_missing_values.py       # 03. 명시적/암묵적 결측치 정밀 집계
│   ├── 10_clustering_evaluation.py      # 04. K-Means 최적 K값 산출 (Elbow & Silhouette)
│   ├── 11_persona_clustering.py         # 05. 3대 페르소나 군집화 및 프로파일링
│   ├── 12_association_rules.py          # 06. FP-Growth 기반 페르소나별 연관 규칙 도출
│   └── 13_network_visualization.py      # 07. NetworkX 네트워크 그래프 및 중심성 분석
│
├── project2/                         # (추후 2차 프로젝트 진행 시 생성 예정)
│
├── run_all.py                        # 파이프라인 자동 실행 오케스트레이터
├── requirements.txt                  # 실행 환경 라이브러리 의존성 목록
├── .gitignore                        # 대용량 데이터 및 설정 파일 제외
└── README.md                         # 프로젝트 종합 통합 안내서