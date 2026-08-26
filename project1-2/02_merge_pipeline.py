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
    
    # 3. SHP 공간 데이터 (인코딩 크로스 체크)
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
    
    df_store_clean = df_store.dropna(subset=['상권_코드']).copy()
    df_store_clean['상권_코드'] = df_store_clean['상권_코드'].astype(int)
    
    df_hinterland_clean = df_hinterland.copy()
    df_hinterland_clean.rename(columns={'상권배후지_코드': '상권_코드', '상권배후지_코드_명': '상권_코드_명'}, inplace=True)
    
    key_cols = ['기준_년분기_코드', '상권_코드']
    
    store_val_cols = [c for c in df_store_clean.columns if c not in key_cols and c not in ['상권_구분_코드', '상권_구분_코드_명', '상권_코드_명']]
    hinterland_val_cols = [c for c in df_hinterland_clean.columns if c not in key_cols and c not in ['상권_구분_코드', '상권_구분_코드_명', '상권_코드_명']]
    
    df_store_sub = df_store_clean[key_cols + store_val_cols].rename(columns={c: f"상권_{c}" for c in store_val_cols})
    df_hinterland_sub = df_hinterland_clean[key_cols + hinterland_val_cols].rename(columns={c: f"배후지_{c}" for c in hinterland_val_cols})
    
    shp_master = gdf_shape[['TRDAR_CD', 'TRDAR_CD_N', 'SIGNGU_CD_']].copy()
    shp_master.columns = ['상권_코드', '상권_명_마스터', '자치구명']
    shp_master['상권_코드'] = shp_master['상권_코드'].astype(int)
    
    return df_store_sub, df_hinterland_sub, shp_master

def merge_datasets(df_store_sub, df_hinterland_sub, shp_master):
    print("\n[STEP 3] 데이터 병합 수행 중...")
    
    merged_df = pd.merge(df_store_sub, df_hinterland_sub, on=['기준_년분기_코드', '상권_코드'], how='inner')
    final_df = pd.merge(merged_df, shp_master, on='상권_코드', how='left')
    
    front_cols = ['기준_년분기_코드', '자치구명', '상권_코드', '상권_명_마스터', '상권_총_유동인구_수', '배후지_총_유동인구_수']
    other_cols = [c for c in final_df.columns if c not in front_cols]
    final_df = final_df[front_cols + other_cols]
    
    return final_df

if __name__ == "__main__":
    df_store, df_hinterland, gdf_shape = load_data()
    df_store_sub, df_hinterland_sub, shp_master = clean_and_preprocess(df_store, df_hinterland, gdf_shape)
    final_merged_df = merge_datasets(df_store_sub, df_hinterland_sub, shp_master)
    
    output_path = os.path.join(PROCESSED_DATA_DIR, "merged_population_5years.csv")
    final_merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 80)
    print(f"[성공] 전처리 및 병합 완료! 저장 경로: {output_path}")
    print(f"- 최종 생성된 데이터 크기 (행, 열): {final_merged_df.shape}")
    print("=" * 80)