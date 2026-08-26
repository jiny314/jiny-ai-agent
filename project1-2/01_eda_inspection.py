import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd

RAW_DATA_DIR = os.path.join(".", "data", "raw")
SHAPE_DATA_DIR = os.path.join(".", "data", "shape")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_and_merge_csv(file_pattern):
    search_path = os.path.join(RAW_DATA_DIR, file_pattern)
    file_list = glob.glob(search_path)
    
    if not file_list:
        print(f"[경고] {search_path} 패턴에 해당하는 CSV 파일이 없습니다.")
        return None
    
    df_list = []
    for f in file_list:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding='cp949')
        df_list.append(df)
        
    return pd.concat(df_list, ignore_index=True)

def inspect_basic_info(df, dataset_name):
    print("=" * 80)
    print(f"[{dataset_name}] 데이터 구조 및 결측치 점검")
    print("=" * 80)
    print(f"- 전체 데이터 크기 (행, 열): {df.shape}")
    
    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Missing_Count': df.isnull().sum(),
        'Missing_Ratio(%)': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique_Values': df.nunique()
    })
    print(info_df.head(15))

def inspect_shapefile():
    shp_files = glob.glob(os.path.join(SHAPE_DATA_DIR, "*.shp"))
    if not shp_files:
        print("\n[안내] SHP 파일이 존재하지 않습니다.")
        return
        
    shp_path = shp_files[0]
    print(f"\n[GIS SHP 공간 데이터 점검]: {os.path.basename(shp_path)}")
    
    encodings = ['utf-8', 'cp949', 'euc-kr']
    gdf = None
    for enc in encodings:
        try:
            gdf = gpd.read_file(shp_path, encoding=enc)
            sample_text = str(gdf['TRDAR_CD_N'].iloc[0])
            if '諛' not in sample_text and '떊' not in sample_text:
                break
        except Exception:
            continue

    if gdf is None:
        gdf = gpd.read_file(shp_path)

    print(f"- SHP 데이터 크기: {gdf.shape}")
    print("- SHP 컬럼 목록:", gdf.columns.tolist())
    
    target_cols = ['TRDAR_CD', 'TRDAR_CD_N', 'SIGNGU_CD_']
    available_cols = [c for c in target_cols if c in gdf.columns]
    
    print("\n[한글 데이터 정상 출력 확인]")
    print(gdf[available_cols].head())

if __name__ == "__main__":
    df_store = load_and_merge_csv("*상권*.csv")
    if df_store is not None:
        inspect_basic_info(df_store, "상권 유동인구")

    df_hinterland = load_and_merge_csv("*배후지*.csv")
    if df_hinterland is not None:
        inspect_basic_info(df_hinterland, "상권배후지 유동인구")

    inspect_shapefile()