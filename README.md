# 🤖 AI 에이전트 프로젝트 (AI Agent Repository)

본 레포지토리는 AI 에이전트 기반 분석 및 머신러닝 모듈을 프로젝트별로 체계적으로 관리하기 위해 구축되었습니다.

---

## 🛠️ Tech Stack & Tools

| 구분 | 사용 기술 및 라이브러리 |
| --- | --- |
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy, GeoPandas |
| **Database & SQL** | SQLite3, SQL Window Functions (LAG, LEAD, Moving Average) |
| **Machine Learning & Clustering** | scikit-learn (K-Means, DBSCAN, Gaussian Mixture Model, StandardScaler, RobustScaler, Random Forest, TruncatedSVD) |
| **Association & Sequence Mining** | mlxtend (Apriori, FP-Growth), PrefixSpan (시퀀셜 패턴 마이닝) |
| **Network Analysis** | NetworkX (연관 네트워크, Community Detection/Louvain) |
| **Optimization** | SciPy (Hungarian Algorithm, Sparse Matrix) |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Git, GitHub, VS Code |
---

## 📌 프로젝트 목록 (Project List)

* **[project1]** 초개인화 페르소나 기반 금융 상품 패키징 및 교차 판매(Cross-selling) 전략 도출
* **[project1-2]** SEOUL-GHOST-STORE: 상권-배후지 유동인구 갭(Gap) 분석 기반의 '잠재 단절 상권' 감지 및 입지 가치 재평가 파이프라인
* **[project1-3]** AI 기반 OTT 콘텐츠 흥행 트래킹 및 글로벌 수급 의사결정 지원 시스템 기획
* **2차 프로젝트** *(추후 `project2/` 폴더 생성 및 확장 예정)*

---

## 📑 Data Source & Citation

* **한국지능정보사회진흥원(NIA) AI Hub**: [금융상품·서비스 및 소비자 특성 데이터](https://www.aihub.or.kr/)
* **서울열린데이터광장**: [서울시 상권분석서비스](https://data.seoul.go.kr/)
* **Kaggle**: [Netflix Top 10 Weekly Dataset](https://www.kaggle.com/) *(원자료: [Netflix Top 10](https://top10.netflix.com/))*

> **정책 준수 Notice**  
> 본 레포지토리는 AI Hub, 서울열린데이터광장 및 데이터 제공처의 이용 정책을 엄격히 준수합니다. 원본 데이터 파일(`소비자.csv`, 상권 분석 데이터, Netflix 주차별 CSV 등)은 데이터 보호 및 용량 관리 정책상 Git 트래킹에서 제외되어 있습니다.

---

## 📋 [1차 프로젝트] 개요 및 목적

### 1. Project1 : 초개인화 페르소나 기반 금융 상품 패키징 및 교차 판매(Cross-selling) 전략 도출
* **배경 및 목적**  
  대규모 금융 거래 데이터(38.5만 건)를 활용하여 비지도학습(K-Means, K=3) 기반 초개인화 페르소나를 도출합니다.  
  페르소나별 FP-Growth 연관 규칙 및 NetworkX 연결 중심성 그래프 분석을 적용하여 고객 유지율과 LTV를 극대화하는 맞춤형 교차 판매 패키징 시나리오를 수립합니다.

### 2. Project1-2 : SEOUL-GHOST-STORE (잠재 단절 상권 감지 및 입지 가치 재평가 파이프라인)
* **배경 및 목적**  
  서울시 5개년(2021 Q1 ~ 2026 Q1) 길단위 유동인구 데이터를 활용하여 배후지 유동인구 대비 상권 유입률이 저평가된 '유령 상권(Shadow Market)'을 발굴합니다.  
  단순 상권 매출이나 총인구수에 의존하던 입지 분석 방식에서 벗어나 '상권배후지-상권 간 유동인구 전이율'을 정의하고, K-Means & 2x2 매트릭스를 통해 전체 1,090개 분석 상권 중 304개(27.9%)의 '잠재 단절 상권'을 핀셋 추출하여 활성화 솔루션을 제안합니다.

### 3. Project1-3 : AI 기반 OTT 콘텐츠 흥행 트래킹 및 글로벌 수급 의사결정 지원 시스템 기획
* **배경 및 목적**  
  수십 개 국가의 주차별 순위 변동과 시청 데이터 파편화로 인한 콘텐츠 수급 판단 시간 지연 문제를 해결합니다.  
  글로벌 시청 실적 및 94개국 순위 데이터를 통합 데이터 마트(Star Schema)로 정제하고, SQL 윈도우 함수(LAG, LEAD, 이동평균) 기반의 순위 상승 모멘텀 지표를 산출합니다. Random Forest 변수 중요도 분석 및 시계열 잔류 패턴 분류를 통해 글로벌 Top 10 차트 잔류 가능성을 예측하며, AI 기반 자동 예측 리포트 서비스(AX) 기획의 데이터 기반을 구축합니다.

---

## 📁 프로젝트 폴더 구조 (Directory Structure)

1차 분석 스크립트 및 데이터 산출물은 `project1/`, `project1-2/` 및 `project1-3/` 폴더 내부에서 독립적으로 관리됩니다.

```text
jiny-ai-agent/
│
├── project1/                              # 1차 프로젝트 (금융 상품 교차판매) 전용 폴더
│   ├── data/                              # 원본 데이터 폴더 (.gitignore 제외)
│   │   └── 소비자.csv
│   │
│   ├── output/                            # 전처리 및 분석 결과 산출물 (.pkl, .png)
│   │
│   ├── 1_preprocess.py                    # 01. 고객 단위 전처리 및 요약 집계
│   ├── 2_eda_categorical_unique.py        # 02. 결측치/이상치 확인, 분포 및 변수 간 관계 시각화
│   ├── 3_eda_missing_values.py            # 03. 명시적/암묵적 결측치 정밀 집계
│   ├── 4_customer_data_vis.py             # 04. 주요 범주형 변수 유니크 값 및 분포 확인
│   ├── 5_eda_summary_stats.py             # 05. 수치형 변수 요약 통계량
│   ├── 6_eda_outliers.py                  # 06. 이상치/노이즈 데이터 파악 및 정제
│   ├── 7_eda_product_distribution.py      # 07. 주요 분석 축 분포 및 클래스 불균형 분석
│   ├── 8_eda_relationships.py             # 08. 변수 간 상관관계 및 교차 분석
│   ├── 9_eda_visualizations_advanced.py   # 09. TOP 20 상품 파레토 분석 및 상관관계 히트맵
│   ├── 10_clustering_evaluation.py        # 10. K-Means 최적 K값 산출 (Elbow & Silhouette)
│   ├── 11_persona_clustering.py           # 11. 3대 페르소나 군집화 및 프로파일링
│   ├── 14_gmm_validation.py               # 12. GMM 기반 군집 결과 교차 검증
│   ├── 16_dbscan_outlier_detection.py     # 13. DBSCAN 기반 밀도 이상치 고객 탐지
│   ├── 12_association_rules.py            # 14. FP-Growth 기반 페르소나별 연관 규칙 도출
│   ├── 13_networkx_visualization.py       # 15. NetworkX 네트워크 그래프 및 중심성 분석
│   ├── 15_community_detection.py          # 16. Louvain 기반 상품 커뮤니티 탐지
│   ├── 17_recommendation_engine.py        # 17. 협업 필터링/행렬분해 기반 추천 엔진 및 평가
│   └── 18_sequential_pattern_mining.py    # 18. PrefixSpan 기반 시퀀셜 패턴 마이닝
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
├── project1-3/                       # 1차 추가 프로젝트 (OTT 콘텐츠 흥행 트래킹 및 AX 기획) 전용 폴더
│   ├── data/                         # 원본 Netflix CSV 데이터 저장 폴더 (.gitignore 제외)
│   │   ├── all-weeks-global.csv
│   │   └── all-weeks-countries.csv
│   │
│   ├── output/                       # 정제된 데이터 마트 및 시각화 산출물 (.gitignore 제외)
│   │   ├── ott_data_mart.db          # SQLite3 데이터 마트 (Star Schema)
│   │   └── feature_importance.png    # Random Forest 변수 중요도 차트
│   │
│   ├── 01_environment_setup.py    # 01. OS 독립적 경로 및 한글 폰트 설정
│   ├── 02_data_preprocessing.py   # 02. 데이터 정제 & SQLite 데이터 마트(Star Schema) 구축
│   ├── 03_problem_definition.py   # 03. 데이터 마트 요약 집계 및 비즈니스 문제/가설 정의
│   ├── 04_eda_time_series.py      # 04. SQL 윈도우 함수 시계열 분석 & Random Forest 변수 중요도 산출
│   └── requirements.txt           # project1-3 전용 의존성 목록
│
├── project2/                         # (추후 2차 프로젝트 진행 시 생성 예정)
│
├── run_all.py                        # 전체 레포지토리 오케스트레이터
├── requirements.txt                  # 통합 환경 실행 라이브러리 의존성 목록
├── .gitignore                        # 대용량 데이터 및 설정 파일 제외
└── README.md                         # 프로젝트 종합 통합 안내서