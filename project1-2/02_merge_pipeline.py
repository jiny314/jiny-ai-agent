import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd

RAW_DATA_DIR = os.path.join(".", "data", "raw")
SHAPE_DATA_DIR = os.path.join(".", "data", "shape")
PROCESSED_DATA_DIR = os.path.join(".", "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_data():
    print("[STEP 1] 원천 데이터 로드 시작...")
    
    # 1. 상권 CSV
    store_files = glob.glob(os.path.join(RAW_DATA_DIR, "*상권*.csv"))
    store_files = [f for f in store_files if "배후지" not in f]
    df_store_list = []
    for f in store_files:
        try:
            df_store_list.append(pd.read_csv(f, encoding='utf-8-sig'))
        except UnicodeDecodeError:
            df_store_list.append(pd.read_csv(f, encoding='cp949'))
    df_store = pd.concat(df_store_list, ignore_index=True)

    # 2. 배후지 CSV
    hinterland_files = glob.glob(os.path.join(RAW_DATA_DIR, "*배후지*.csv"))
    df_hinterland_list = []
    for f in hinterland_files:
        try:
            df_hinterland_list.append(pd.read_csv(f, encoding='utf-8-sig'))
        except UnicodeDecodeError:
            df_hinterland_list.append(pd.read_csv(f, encoding='cp949'))
    df_hinterland = pd.concat(df_hinterland_list, ignore_index=True)

    # 3. SHP 공간 데이터
    shp_files = glob.glob(os.path.join(SHAPE_DATA_DIR, "*.shp"))
    shp_path = shp_files[0]
    gdf_shape = None
    for enc in ['utf-8', 'cp949', 'euc-kr']:
        try:
            gdf_shape = gpd.read_file(shp_path, encoding=enc)
            sample_text = str(gdf_shape['TRDAR_CD_N'].iloc[0])
            if '諛' not in sample_text and '떊' not in sample_text:
                break
        except Exception:
            continue

    return df_store, df_hinterland, gdf_shape

def clean_and_preprocess(df_store, df_hinterland, gdf_shape):
    print("\n[STEP 2] 데이터 정제 및 컬럼 표준화 진행 중...")
    
    # 상권 결측치 제거
    df_store_clean = df_store.dropna(subset=['상권_코드']).copy()
    df_store_clean['상권_코드'] = df_store_clean['상권_코드'].astype(int)
    
    # 배후지 컬럼명 맞춤
    df_hinterland_clean = df_hinterland.copy()
    df_hinterland_clean.rename(columns={
        '상권배후지_코드': '상권_코드',
        '상권배후지_코드_명': '상권_코드_명'
    }, inplace=True)
    
    key_cols = ['기준_년분기_코드', '상권_코드']
    
    # 공통 속성 컬럼 제외 후 프리픽스 부여
    drop_meta = ['상권_구분_코드', '상권_구분_코드_명', '상권_코드_명']
    store_val_cols = [c for c in df_store_clean.columns if c not in key_cols and c not in drop_meta]
    hinterland_val_cols = [c for c in df_hinterland_clean.columns if c not in key_cols and c not in drop_meta]
    
    df_store_sub = df_store_clean[key_cols + drop_meta[:1] + store_val_cols].copy()
    df_hinterland_sub = df_hinterland_clean[key_cols + hinterland_val_cols].copy()
    
    # 컬럼 구분 용 접두사 처리
    df_store_sub.rename(columns={c: f"상권_{c}" for c in store_val_cols}, inplace=True)
    df_hinterland_sub.rename(columns={c: f"배후지_{c}" for c in hinterland_val_cols}, inplace=True)
    
    # 상권과 배후지 데이터 병합 (Inner Join)
    df_merged = pd.merge(df_store_sub, df_hinterland_sub, on=key_cols, how='inner')
    
    # SHP 공간 정보 결합 (GIS 데이터)
    if gdf_shape is not None and 'TRDAR_CD' in gdf_shape.columns:
        gdf_shape['TRDAR_CD'] = gdf_shape['TRDAR_CD'].astype(int)
        df_merged = pd.merge(
            df_merged, 
            gdf_shape[['TRDAR_CD', 'SIGNGU_CD_']], 
            left_on='상권_코드', 
            right_on='TRDAR_CD', 
            how='left'
        )
        df_merged.drop(columns=['TRDAR_CD'], inplace=True)
        
    print(f"[완료] 최종 병합 데이터 크기: {df_merged.shape}")
    return df_merged

def save_processed_data(df):
    output_path = os.path.join(PROCESSED_DATA_DIR, "merged_floating_population.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[완료] 전처리 데이터 저장 성공: {output_path}")

if __name__ == "__main__":
    df_store, df_hinterland, gdf_shape = load_data()
    df_merged = clean_and_preprocess(df_store, df_hinterland, gdf_shape)
    save_processed_data(df_merged)