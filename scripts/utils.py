import os
import matplotlib.pyplot as plt
import platform
import pandas as pd

def setup_font():
    """한글 폰트 설정 및 마이너스 기호 깨짐 방지"""
    print(f"Current OS: {platform.system()}")  # 현재 OS 출력
    
    if platform.system() == 'Windows':
        print("Setting Windows font: Malgun Gothic")
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':  # macOS
        print("Setting macOS font: AppleGothic")
        plt.rc('font', family='AppleGothic')
    else:  # Linux
        print("Setting Linux font: NanumGothic")
        plt.rc('font', family='NanumGothic')

    plt.rcParams['axes.unicode_minus'] = False
    
    # 현재 설정된 폰트 확인
    print(f"Current font settings: {plt.rcParams['font.family']}")

def sort_by_date(df, date_col="date", ascending=True):
    """날짜 기준으로 정렬"""
    df = df.sort_values(by=date_col, ascending=ascending)
    df = df.reset_index(drop=True)
    return df 

def add_month_column(df):
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["month"] = df["날짜"].dt.month
    return df

def pivot_monthly_avg_by_station(df, value_col="pm10"):
    """
    측정소명과 월(month)을 기준으로 주어진 값(value_col)의 평균을 피벗 테이블로 반환하는 함수

    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        value_col (str): 평균을 계산할 컬럼명 (기본값: 'pm10')

    Returns:
        pd.DataFrame: 측정소명-월 기준 평균값 피벗 테이블
    """
    pivot_df = (
        df
        .dropna(subset=["측정소명", "month", value_col])
        .groupby(["측정소명", "month"])[value_col]
        .mean()
        .reset_index()
        .pivot(index="측정소명", columns="month", values=value_col)
    )
    return pivot_df

def create_future_input(station_names, month, temp_list, rain_list, wind_list, pm25_list=None):
    """
    모델 예측을 위한 입력 데이터를 생성하는 함수

    Parameters:
        station_names (array-like): 측정소명 리스트
        month (int): 예측하고자 하는 월 (모든 지역 동일)
        pm25_list (array-like, optional): 각 지역별 pm2.5 값
        temp_list (array-like): 각 지역별 평균기온
        rain_list (array-like): 각 지역별 일강수량
        wind_list (array-like): 각 지역별 평균 풍속

    Returns:
        pd.DataFrame: 예측용 입력 데이터프레임
    """

    if pm25_list is None:
        return pd.DataFrame({
            "측정소명": station_names,
            "month": [month] * len(station_names),
            "평균기온(°C)": temp_list,
            "일강수량(mm)": rain_list,
            "평균 풍속(m/s)": wind_list
        })
    else:
        return pd.DataFrame({
        "측정소명": station_names,
        "month": [month] * len(station_names),
        "pm25": pm25_list,
        "평균기온(°C)": temp_list,
        "일강수량(mm)": rain_list,
        "평균 풍속(m/s)": wind_list
    })

def strip_column_names(df):
    """
    데이터프레임의 열 이름에서 공백을 제거하는 함수
    
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 열 이름의 공백이 제거된 데이터프레임
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def save_to_csv(df, output_dir='processed', file_name='result'):
    """
    전처리된 데이터프레임을 CSV 파일로 저장하는 함수
    
    Parameters:
        df (pd.DataFrame): 저장할 데이터프레임
        output_dir (str): 저장할 디렉토리 경로 (기본값: 'processed')
        file_name (str): 파일명 (기본값: 'result')
    """
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 파일명 생성
    file_name = f'{file_name}.csv'
    file_path = os.path.join(output_dir, file_name)
    
    # CSV 파일로 저장
    df.to_csv(file_path, index=False, encoding='utf-8')
    
    #print(f'파일이 저장되었습니다: {file_path}')

def to_datetime_column(df, date_col='date'):
    """
    지정한 컬럼을 pandas datetime 타입으로 변환하는 함수

    Parameters:
        df (pd.DataFrame): 변환할 데이터프레임
        date_col (str): datetime으로 변환할 컬럼명 (기본값: 'date')

    Returns:
        pd.DataFrame: 지정 컬럼이 datetime 타입으로 변환된 데이터프레임
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df