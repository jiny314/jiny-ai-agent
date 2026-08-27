# 🤖 AI 에이전트 프로젝트 (AI Agent Repository)

본 레포지토리는 AI 에이전트 기반 분석 및 머신러닝 모듈을 프로젝트별로 체계적으로 관리하기 위해 구축되었습니다.

## 📌 프로젝트 목록 (Project List)

- **1차 프로젝트** : 
  - `[project1]` 초개인화 페르소나 기반 금융 상품 패키징 및 교차 판매(Cross-selling) 전략 도출
  - `[project1-2]` SEOUL-GHOST-STORE: 상권-배후지 유동인구 갭(Gap) 분석 기반의 '잠재 단절 상권' 감지 및 입지 가치 재평가 파이프라인
- **2차 프로젝트** : *(추후 project2/ 폴더 생성 및 확장 예정)*

## 📑 Data Source & Citation

- **출처** : 
  - [한국지능정보사회진흥원(NIA) AI Hub — 「금융상품·서비스 및 소비자 특성 데이터」](https://aihub.or.kr)
  - [서울열린데이터광장 — 「서울시 상권분석서비스」 (서울특별시)](https://data.seoul.go.kr)
- **정책 준수** : 본 레포지토리는 AI Hub 및 서울열린데이터광장 이용 정책을 엄격히 준수하며, 원본 데이터 파일(`소비자.csv`, 상권 분석 데이터 등)은 라이선스 정책상 포함되어 있지 않습니다.

## 📋 [1차 프로젝트] 개요 및 목적

- **Project1** : 초개인화 페르소나 기반 금융 상품 패키징 및 교차 판매(Cross-selling) 전략 도출
  - **배경 및 목적** : 
    - 대규모 금융 거래 데이터(38.5만 건)를 활용하여 비지도학습($K = 3$ K-Means) 기반 *초개인화 페르소나*를 도출합니다.
    - 페르소나별 *FP-Growth 연관 규칙* 및 *NetworkX 연결 중심성 그래프* 분석을 적용하여 고객 유지율과 LTV를 극대화하는 *맞춤형 교차 판매(Cross-selling) 패키징 시나리오*를 수립합니다.

- **Project1-2** : SEOUL-GHOST-STORE: 상권-배후지 유동인구 갭(Gap) 분석 기반의 '잠재 단절 상권' 감지 및 입지 가치 재평가 파이프라인
  - **배경 및 목적** : 
    - 서울시 5개년(2021 Q1 ~ 2026 Q1) 길단위 유동인구 데이터를 활용하여 배후지 유동인구 대비 상권 유입률이 저평가된 '유령 상권(Shadow Market)'을 발굴합니다.
    - 단순 상권 매출이나 총인구수에 의존하던 입지 분석 방식에서 벗어나, '상권배후지-상권 간 유동인구 전이율'을 정의하고 비즈니스 활성화 기회를 도출합니다.
    - 배후지 유동인구 대비 상권 유동인구 비율을 지표화하여, 입지적 한계를 극복할 수 있는 잠재력 높은 '단절 상권'을 발굴하고 상권 활성화 솔루션을 제안합니다.

## 📁 프로젝트 폴더 구조 (Directory Structure)

1차 분석 스크립트 및 데이터 산출물은 `project1/` 및 `project1-2/` 폴더 내부에서 독립적으로 관리됩니다.

```text
jiny-ai-agent/
│
├── project1/                         # 1차 프로젝트 (금융 상품 교차판매) 전용 폴더
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
│   ├── 10_clustering_evaluation.py   # 04. K-Means 최적 K값 산출 (Elbow & Silhouette)
│   ├── 11_persona_clustering.py      # 05. 3대 페르소나 군집화 및 프로파일링
│   ├── 12_association_rules.py       # 06. FP-Growth 기반 페르소나별 연관 규칙 도출
│   └── 13_network_visualization.py   # 07. NetworkX 네트워크 그래프 및 중심성 분석
│
├── project1-2/                       # 1차 추가 프로젝트 (서울시 유령상권 분석) 전용 폴더
│   ├── data/                         # 프로젝트 데이터 폴더 (.gitignore 제외)
│   │   ├── raw/                      # 원본 데이터 (상권 및 상권배후지 유동인구 CSV)
│   │   ├── processed/                # 전처리, 지표 계산 및 군집화 완료 데이터
│   │   └── shape/                    # 상권 영역 공간 데이터 (.shp, .dbf, .prj 등)
│   │
│   ├── 01_eda_inspection.py          # 01. 기초 데이터 탐색 및 구조 검증
│   ├── 02_merge_pipeline.py          # 02. 5개년 데이터 병합 및 시계열 전처리 파이프라인
│   ├── 03_gap_metric_calculation.py  # 03. 상권-배후지 유동인구 전이율 및 Gap 지표 산출
│   ├── 04_ghost_market_clustering.py # 04. 잠재 단절(유령) 상권 탐지 및 군집화
│   ├── 05_visualization_report.py    # 05. 상권 입지 가치 재평가 및 공간 시각화 리포트
│   ├── run_all.py                    # project1-2 전체 파이프라인 일괄 실행기
│   └── requirements.txt              # project1-2 의존성 목록
│
├── project2/                         # (추후 2차 프로젝트 진행 시 생성 예정)
│
├── run_all.py                        # 전체 레포지토리 오케스트레이터
├── requirements.txt                  # 통합 환경 실행 라이브러리 의존성 목록
├── .gitignore                        # 대용량 데이터 및 설정 파일 제외
└── README.md                         # 프로젝트 종합 통합 안내서