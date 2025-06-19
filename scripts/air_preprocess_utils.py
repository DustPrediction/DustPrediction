import glob
import numpy as np
import copy
from scripts.utils import os, pd, sort_by_date, strip_column_names, to_datetime_column

def save_to_csv(df, region_name, output_dir='processed', prefix=''):
    """
    전처리된 데이터프레임을 CSV 파일로 저장하는 함수
    
    Parameters:
        df (pd.DataFrame): 저장할 데이터프레임
        region_name (str): 지역명 (파일명에 사용)
        output_dir (str): 저장할 디렉토리 경로 (기본값: 'processed')
        prefix (str): 파일명 접두사 (기본값: '')
    """
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 파일명 생성
    file_name = f'{prefix}{region_name}.csv'
    file_path = os.path.join(output_dir, file_name)
    
    # CSV 파일로 저장
    df.to_csv(file_path, index=False, encoding='utf-8')
      
def normalize_missing_values(df, cols):
    """
    지정한 컬럼들에서 비정상 결측값(빈 문자열, 공백, None, -999, 'nan', 'NaN', 'null' 등)을 pandas의 NA로 통일하는 함수

    Parameters:
        df (pd.DataFrame): 결측값을 정규화할 데이터프레임
        cols (list): 결측값을 정규화할 컬럼명 리스트

    Returns:
        pd.DataFrame: 결측값이 정규화된 데이터프레임
    """
    # 각 지정 컬럼에 대해 비정상 결측값을 pd.NA로 변환
    for col in cols:
        df[col] = df[col].replace(
            ['', ' ', None, -999, 'nan', 'NaN', 'NAN', 'null', 'None', 'NA', np.nan], 
            pd.NA
        )
    return df

def preprocess_air_quality_data(df, date_col='date', start_date='2018-01-01', end_date='2024-12-31', columns_to_keep=None):
    """
    대기질 데이터 전처리를 위한 통합 함수
    
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        date_col (str): 날짜 컬럼명 (기본값: 'date')
        start_date (str): 시작 날짜 (기본값: '2018-01-01')
        end_date (str): 종료 날짜 (기본값: '2024-12-31')
        columns_to_keep (list): 유지할 컬럼 리스트 (기본값: None, 모든 컬럼 유지)
        
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    df = df.copy()
    
    # 1. 열 이름의 공백 제거
    df = strip_column_names(df)
    
    # 2. 날짜 컬럼을 datetime으로 변환
    df = to_datetime_column(df, date_col)
    
    # 3. 필요한 컬럼만 선택
    if columns_to_keep:
        df = df[columns_to_keep]
    
    # 4. 날짜 범위 필터링
    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    
    # 5. 날짜 기준 정렬
    df = sort_by_date(df, date_col=date_col, ascending=True)
    
    return df

def check_missing_data(df, date_col='date', pm25_col='pm25', pm10_col='pm10'):
    """
    데이터프레임에서 결측 데이터를 확인하고 기록하는 함수
    
    Parameters:
        df (pd.DataFrame): 확인할 데이터프레임
        date_col (str): 날짜 컬럼명 (기본값: 'date')
        pm25_col (str): PM2.5 컬럼명 (기본값: 'pm25')
        pm10_col (str): PM10 컬럼명 (기본값: 'pm10')
        
    Returns:
        dict: 결측 데이터 정보를 담은 딕셔너리
            - key: 결측 유형 ('날짜 없음', 'PM25 없음', 'PM10 없음', '날짜만 있음')
            - value: 결측 날짜 리스트
    """
    
    df = df.copy()
    # 날짜를 datetime으로 통일
    df = to_datetime_column(df, date_col)  # date_col만 datetime으로 변환
    # 결측치 정규화
    df = normalize_missing_values(df, [pm25_col, pm10_col])

    # 날짜 범위 생성
    date_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq='D')
    
    # 결측 데이터 정보를 저장할 딕셔너리
    missing_data = {
        '날짜 없음': [],  # 날짜가 없는 경우
        'PM25 없음': [],  # PM25 값이 없는 경우
        'PM10 없음': [],  # PM10 값이 없는 경우
        '날짜만 있음': []  # 날짜는 있지만 PM25, PM10 모두 없는 경우
    }
    
    # 날짜가 없는 경우 확인
    missing_dates = set(date_range) - set(pd.to_datetime(df[date_col]))
    if missing_dates:
        # 각 날짜를 'YYYY-MM-DD' 문자열로 변환해서 저장
        missing_data['날짜 없음'] = sorted([d.strftime('%Y-%m-%d') for d in missing_dates])

    # 결측치 유형별로 분류
    for date in pd.to_datetime(df[date_col].unique()):
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')  # 문자열로 변환
        date_data = df[df[date_col] == date]

        pm25_na = date_data[pm25_col].isna().all()
        pm10_na = date_data[pm10_col].isna().all()
        # 둘 다 결측
        if pm25_na and pm10_na:
            missing_data['날짜만 있음'].append(date_str)
        # pm25만 결측
        elif pm25_na:
            missing_data['PM25 없음'].append(date_str)
        # pm10만 결측
        elif pm10_na:
            missing_data['PM10 없음'].append(date_str)
    
    # 결과 반환
    return missing_data

def process_subdata_year_folder(year_folder_path):
    """
    연도별 서브데이터 폴더(예: '2018')를 받아, 하위 1~12월 엑셀 파일을 모두 읽어
    region/date/pm10/pm25 4개 컬럼만 추출하고, date는 일 단위(YYYYMMDD)로 평균을 냅니다.
    region 이름순, 날짜 오름차순으로 정렬된 DataFrame을 반환합니다.

    Parameters:
        year_folder_path : str
            연도별 서브데이터가 들어있는 폴더 경로 (예: '../data/raw/air_quality/sub/2018')

    Returns:
        pd.DataFrame
            region, date, pm10, pm25 컬럼으로 구성된 정제된 데이터프레임
    """
    all_dfs = []

    files = sorted(glob.glob(os.path.join(year_folder_path, '*.xlsx')))
    for file in files:
        df = pd.read_excel(file)
        
        # 열 공백 제거
        df = strip_column_names(df)
        
        # 열 이름 표준화
        rename_dict = {
            '측정일시': 'date',
            'PM10': 'pm10',
            'PM25': 'pm25',
            '지역': 'region',
        }
        df = df.rename(columns=rename_dict)

        # '서울 '로 시작하는 행만 남김
        df = df[df['region'].str.startswith('서울 ')].copy()
        
        # 필요한 컬럼만 남기기
        df = df[['region', 'date', 'pm10', 'pm25']]

        # '서울 ' 제거해서 구 이름만 남김
        # '서울 성북구' -> '성북구'
        df['region'] = df['region'].str.replace('서울 ', '', regex=False)
       
        # date 컬럼을 문자열로 변환한 뒤, 앞 8자리(YYYYMMDD)만 추출하여 날짜 포맷을 통일
        df['date'] = df['date'].astype(str).str[:8]
        # 추출한 8자리 문자열을 pandas datetime64[ns] 타입(Timestamp)으로 변환
        df = to_datetime_column(df)
        
        # 일자별, region별 평균
        df = df.groupby(['region', 'date'], as_index=False)[['pm10', 'pm25']].mean()
        df['pm10'] = df['pm10'].round().astype('Int64')
        df['pm25'] = df['pm25'].round().astype('Int64')
        all_dfs.append(df)
    
    # 모든 월 데이터 합치기
    result = pd.concat(all_dfs, ignore_index=True)
    
    # region 이름순, date 오름차순 정렬
    result = result.sort_values(['region', 'date']).reset_index(drop=True)

    return result

def handle_missing_type_from_sub(missing_type, missing_list, main_df, sub_dir, region_name):
    """
    결측 유형별로(main_df에서) 서브데이터(sub_dir)에서 값을 찾아 main_df에 채워넣는 함수

    Parameters:
        missing_type (str): 결측 유형 ('날짜만 있음', 'PM10 없음', 'PM25 없음')
        missing_list (list): 해당 결측 유형에 해당하는 날짜 리스트 (YYYY-MM-DD 문자열)
        main_df (pd.DataFrame): 결측값을 채울 메인 데이터프레임
        sub_dir (str): 서브데이터가 저장된 폴더 경로
        region_name (str): 처리할 구 이름
    Returns:
        None (main_df는 참조로 수정됨)
    """
    # 결측 날짜 리스트를 복사본으로 순회
    for date_str in missing_list[:]:
        # 날짜 문자열을 datetime 객체로 변환
        date_obj = pd.to_datetime(str(date_str)[:10], errors='coerce')
        year = str(date_obj.year)
        # 해당 연도 서브데이터 파일 경로 생성
        sub_path = os.path.join(sub_dir, f"{year}.csv")
        if not os.path.exists(sub_path):
            continue

        # 서브데이터 읽기
        sub_df = pd.read_csv(sub_path)
        # date 컬럼을 문자열(YYYY-MM-DD)로 통일
        sub_df['date'] = sub_df['date'].astype(str).str[:10]
        # date 컬럼을 datetime으로 변환
        sub_df = to_datetime_column(sub_df)

        # region, 날짜가 일치하는 row 찾기
        row = sub_df[(sub_df['region'] == region_name) & (sub_df['date'].dt.date == date_obj.date())]
        if not row.empty:
            pm10_val = row.iloc[0]['pm10']
            pm25_val = row.iloc[0]['pm25']

            # 결측 유형에 따라 main_df에 값 채우기
            if missing_type == '날짜만 있음':
                if (pm10_val is not None and not pd.isna(pm10_val)) and (pm25_val is not None and not pd.isna(pm25_val)):
                    main_df.loc[main_df['date'] == date_obj, 'pm10'] = pm10_val
                    main_df.loc[main_df['date'] == date_obj, 'pm25'] = pm25_val
            elif missing_type == 'PM10 없음':
                if pm10_val is not None and not pd.isna(pm10_val):
                    main_df.loc[main_df['date'] == date_obj, 'pm10'] = pm10_val
            elif missing_type == 'PM25 없음':
                if pm25_val is not None and not pd.isna(pm25_val):
                    main_df.loc[main_df['date'] == date_obj, 'pm25'] = pm25_val
        else:
            # 서브데이터에 해당 region, 날짜 row가 없을 때
            print(f"[서브데이터 없음] {region_name} {date_obj} → 값 없음, main_df 변경 없음")

        # 처리한 날짜는 리스트에서 제거
        date_str_fmt = date_obj.strftime('%Y-%m-%d')
        if date_str_fmt in missing_list:
            missing_list.remove(date_str_fmt)

def fill_missing_from_sub(main_df, region_name, missing_info, sub_dir):
    """
    main_df의 결측값을 서브데이터(sub_dir)에서 찾아 채우는 통합 함수
    (날짜 없음, 날짜만 있음, PM10 없음, PM25 없음 순서로 처리)

    Parameters:
        main_df (pd.DataFrame): 결측값을 채울 메인 데이터프레임
        region_name (str): 처리할 구 이름
        missing_info (dict): 결측 유형별 날짜 리스트 딕셔너리
        sub_dir (str): 서브데이터가 저장된 폴더 경로
    Returns:
        pd.DataFrame: 결측값이 보정된 main_df (날짜 기준 정렬)
    """
    # main_df, missing_info 복사본 사용 (원본 보호)
    main_df = main_df.copy()
    missing_info = copy.deepcopy(missing_info)

    # 날짜 컬럼을 datetime으로 통일
    main_df = to_datetime_column(main_df)
    added_values = []

    # 날짜 없음 처리: main_df에 해당 날짜 row 추가, '날짜만 있음'으로 이동
    for date_str in missing_info['날짜 없음'][:]:
        # 문자열(YYYY-MM-DD)을 datetime으로 변환
        date_obj = pd.to_datetime(str(date_str)[:10], errors='coerce')
        if pd.isna(date_obj):
            continue

        # 결측 날짜 row를 main_df에 추가 (pm10, pm25는 NaN)
        new_row = {'date': date_obj, 'pm25': np.nan, 'pm10': np.nan}
        main_df = pd.concat([main_df, pd.DataFrame([new_row])], ignore_index=True)

        # string(YYYY-MM-DD)로 변환해서 '날짜만 있음'에 추가
        date_str_fmt = date_obj.strftime('%Y-%m-%d')
        missing_info['날짜만 있음'].append(date_str_fmt)

        # '날짜 없음'에서 해당 날짜(string) 제거
        if date_str_fmt in missing_info['날짜 없음']:
            missing_info['날짜 없음'].remove(date_str_fmt)
    
    # 날짜만 있음, PM10 없음, PM25 없음 처리 → 공통 함수로 대체
    handle_missing_type_from_sub('날짜만 있음', missing_info['날짜만 있음'], main_df, sub_dir, region_name)
    handle_missing_type_from_sub('PM10 없음', missing_info['PM10 없음'], main_df, sub_dir, region_name)
    handle_missing_type_from_sub('PM25 없음', missing_info['PM25 없음'], main_df, sub_dir, region_name)
    
    main_df = to_datetime_column(main_df)
    # 최종 정렬 후 반환
    return sort_by_date(main_df)

def merge_air_quality_files(input_dir, output_file):
    """
    input_dir 폴더 내 모든 csv 파일을 하나로 합쳐 output_file로 저장합니다.
    각 파일명(구 이름)을 '측정소명' 컬럼으로 추가합니다.

    Parameters:
        input_dir (str): 통합할 CSV 파일들이 들어있는 폴더 경로
        output_file (str): 저장할 통합 CSV 파일명
    """
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))  # 폴더 내 모든 csv 파일 경로 리스트업
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file) 
        # 파일명에서 구 이름 추출
        region_name = os.path.splitext(os.path.basename(file))[0]
        # 'region' 대신 '측정소명' 컬럼에 구 이름 저장
        df['측정소명'] = region_name
        df_list.append(df)
    
    # 데이터프레임 통합
    merged_df = pd.concat(df_list, ignore_index=True)
    
    if 'date' in merged_df.columns:
        # 날짜 컬럼 datetime 변환
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        # 측정소명, 날짜 기준 정렬
        merged_df = merged_df.sort_values(['측정소명', 'date']).reset_index(drop=True)
    
    merged_df.to_csv(output_file, index=False, encoding='utf-8')  # 통합 파일로 저장
    
    print(f'통합 완료! → {output_file}')