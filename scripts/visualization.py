"""
결과 시각화를 위한 함수 모음.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

def plot_feature_importance(model):
    """특성 중요도 시각화"""
    importances = model.feature_importances_
    names = model.feature_names_in_
    plt.figure(figsize=(8, 6))
    sns.barplot(x = names, y = importances)
    plt.title("Feature Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_decision_tree(model, feature_names, max_depth=3):
    """의사결정트리 시각화"""
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=feature_names, filled=True, fontsize=10, max_depth=max_depth)
    plt.tight_layout()
    plt.show()

def plot_nearby_daycares_outside_district(daycare_df, monitoring_station_df, station_name, radius_km=3):
    """
    특정 측정소를 중심으로 반경 내에 존재하지만, 다른 자치구에 속한 어린이집들을 시각화하는 함수입니다.

    Parameters:
        daycare_df (pd.DataFrame): 어린이집 정보와 측정소 거리 정보가 포함된 데이터프레임
        monitoring_station_df (pd.DataFrame): 대기오염 측정소 위치 정보
        station_name (str): 분석하고자 하는 기준 측정소 이름
        radius_km (float): 반경 거리 (기본값 3km)
    """

    # 기준 측정소 위치 및 자치구 정보 추출
    target_station = monitoring_station_df[monitoring_station_df["측정소명"] == station_name].iloc[0]
    target_lat = target_station["위도"]
    target_lon = target_station["경도"]
    target_district = target_station["측정소명"]  # 측정소명이 곧 자치구명이라는 전제

    # 기준 측정소 반경 내 어린이집 중, 자치구가 다른 데이터만 필터링
    filtered = daycare_df[
        (daycare_df["측정소까지거리(km)"] <= radius_km) &
        (daycare_df["측정소명"] == station_name) &
        (daycare_df["어린이집 위치"] != target_district)
    ]

    # 시각화 시작
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # 자치구별로 다른 색상으로 어린이집 위치 시각화
    for district in filtered["어린이집 위치"].unique():
        district_df = filtered[filtered["어린이집 위치"] == district]
        ax.scatter(
            district_df["경도"], district_df["위도"],
            label=district, s=80, edgecolors='black'
        )

        # 어린이집 옆에 자치구 이름 표시
        for _, row in district_df.iterrows():
            ax.text(
                row["경도"] + 0.0005, row["위도"] + 0.0005,
                row["어린이집 위치"], fontsize=7, color="gray"
            )

    # 기준 측정소 위치 마커(X)로 표시
    ax.scatter(target_lon, target_lat, color="black", marker="X", s=100, label=f"{station_name} 측정소")

    # 측정소 중심으로 반경 원 그리기 (위도 기준: 1도 ≈ 111km)
    circle = plt.Circle(
        (target_lon, target_lat), radius_km / 111,
        color='gray', fill=False, linestyle='--', alpha=0.3
    )
    ax.add_patch(circle)

    # 그래프 제목 및 축 설정
    ax.set_title(f"{station_name} 측정소 반경 {radius_km}km 이내, 타 자치구 어린이집 분포")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.axis("equal")  # 가로세로 비율 고정
    ax.grid(True)
    ax.legend(title="어린이집 자치구", loc="upper right")
    plt.tight_layout()
    plt.show()

def draw_monthly_pm10_subplot(data_df, monitoring_station_df, start_month, end_month, radius_km=3):
    """
    여러 달(month)의 PM10 농도를 시각화하는 함수입니다.
    각 월에 대해, 서울 측정소 반경 radius_km 이내에 위치한 어린이집의 PM10 값을 지도 위에 산점도로 표시합니다.

    Parameters:
        data_df (pd.DataFrame): 어린이집 및 PM10 정보가 포함된 데이터프레임
        monitoring_station_df (pd.DataFrame): 서울시 대기오염 측정소 위치 정보
        start_month (int): 시각화할 시작 월
        end_month (int): 시각화할 마지막 월
        radius_km (float): 측정소 반경 (기본값 3km)
    """
    
    # 서울 측정소만 필터링
    seoul_stations = monitoring_station_df[monitoring_station_df["지역명"] == "서울"]

    month_range = range(start_month, end_month + 1)
    n_months = len(month_range)

    # 서브플롯 행과 열 설정
    n_cols = 3
    n_rows = (n_months + n_cols - 1) // n_cols  # 필요한 행 개수 계산

    # subplot 생성 및 크기 지정
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 5 * n_rows)
    )
    axes = axes.flatten()  # 다차원 배열을 1차원으로 평탄화

    for i, month in enumerate(month_range):
        ax = axes[i]

        # 해당 월에 해당하는 데이터만 필터링
        subset = data_df[
            (data_df["측정소까지거리(km)"] <= radius_km) &
            (data_df["pm10"].notna()) &
            (data_df["month"] == month)
        ]

        # 어린이집 위치 산점도로 표시 (PM10 값을 색상으로 표현)
        scatter = ax.scatter(
            subset["경도"], subset["위도"],
            c=subset["pm10"], cmap="YlOrRd", s=20, alpha=0.7,
            vmin=0, vmax=60
        )

        # 각 측정소에 대해 위치와 이름, 반경 원 그리기
        for _, station in seoul_stations.iterrows():
            name = station["측정소명"]
            lat = station["위도"]
            lon = station["경도"]

            # 측정소 위치 마커(X) 표시
            ax.scatter(lon, lat, marker="X", color="black", s=60)
            ax.text(lon + 0.002, lat + 0.002, name, fontsize=8, color="black")

            # 반경 km 내 원 그리기 (위도 1도 ≈ 111km)
            circle = plt.Circle(
                (lon, lat), radius_km / 111,
                color='gray', fill=False, linestyle='--', alpha=0.3
            )
            ax.add_patch(circle)

        # 서브플롯 제목 및 설정
        ax.set_title(f"{month}월", fontsize=12)
        ax.set_xlim(126.8, 127.2)
        ax.set_ylim(37.35, 37.75)
        ax.axis("equal")
        ax.grid(True)

    # 전체 subplot보다 축이 더 많을 경우, 빈 subplot 제거
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 공통 colorbar 설정 (우측에 별도 축 생성)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("PM10 농도 (ug/m³)", fontsize=13)

    # 전체 제목 설정
    plt.suptitle(
        f"{start_month}월 ~ {end_month}월 PM10 농도 변화 시각화\n(측정소 반경 {radius_km}km 내 어린이집)",
        fontsize=16
    )

    # 최종 시각화 출력
    plt.show()

def plot_bad_pm10_heatmap(data_df):
    """
    PM10 '나쁨' 등급의 측정소-월별 분포를 히트맵으로 시각화합니다.

    '매우 나쁨' 등급은 제외된 이유:
    - 발생 빈도가 낮아 분석 전체의 흐름을 왜곡할 수 있음
    - 색상 스케일의 비효율적 사용을 방지하고, '나쁨' 기준 중심으로 시각적 명확성 확보
    - 정책적으로도 '나쁨' 등급부터 외부 활동 자제, 주의 권고 등의 의미가 부여됨

    Parameters:
    - data_df: pd.DataFrame, PM10 정보가 포함된 어린이집 데이터
    - radius_km: float, 측정소 반경 거리 기준 (기본값 3km)
    """

    def pm10_grade(val):
        """
        PM10 농도에 따라 등급(좋음, 보통, 나쁨, 매우 나쁨)을 부여하는 함수입니다.

        [환경부 기준 (24시간 평균)]
        - 좋음: 0 ~ 30 µg/m³
        - 보통: 31 ~ 80 µg/m³
        - 나쁨: 81 ~ 150 µg/m³
        - 매우 나쁨: 151 µg/m³ 초과

        출처:
        - 「대기오염 예측·발표의 대상지역 및 기준과 내용 등에 관한 고시」 제2조제1호  
        - (환경부 고시, 법제처 국가법령정보센터)  
        - https://www.law.go.kr/행정규칙/대기오염예측·발표의대상지역및기준과내용등에관한고시
        """
        if val <= 30:
            return "좋음"
        elif val <= 80:
            return "보통"
        elif val <= 150:
            return "나쁨"
        else:
            return "매우 나쁨"

    # 등급 컬럼 생성
    data_df = data_df.copy()
    data_df["pm10등급"] = data_df["pm10"].apply(pm10_grade)

    # 측정소기준_구역, 월별 '나쁨' 등급 빈도수 계산
    grade_dist = data_df[data_df["측정소까지거리(km)"] <= 3].groupby(["측정소명", "month"])["pm10등급"].value_counts().unstack(fill_value=0)

    # 나쁨 등급 히트맵
    bad_df = grade_dist["나쁨"].unstack().fillna(0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(bad_df, annot=True, fmt=".0f", cmap="Reds")
    plt.title("측정소별 월별 PM10 '나쁨' 일수 분포")
    plt.xlabel("월")
    plt.ylabel("측정소기준 구역")
    plt.show()

def plot_pm10_prediction_timeseries(model, X_test, y_test, y_pred, title):
    """
    회귀 모델의 PM10 예측 결과를 시계열 형태로 시각화하는 함수입니다.

    - 입력된 테스트 데이터(X_test)에 대해 모델 예측값을 구하고,
    - 실제값(y_test)과 예측값을 시계열 순서(인덱스 기준)로 정렬하여 비교합니다.
    - 데이터가 많을 경우, 앞에서부터 500개 샘플만 추출하여 그래프에 표시합니다.
    - 실제값과 예측값의 추이를 한눈에 비교할 수 있도록 선 그래프로 시각화합니다.

    Parameters:
        model: 학습된 회귀 모델 객체
        X_test: 테스트용 특성 데이터프레임
        y_test: 테스트용 실제 PM10 값 (시리즈)
        title: 그래프 제목 (str)
    """

    # y_test의 인덱스를 기준으로 정렬하여 시계열 순서 유지
    sorted_index = y_test.sort_index().index
    y_test_sorted = y_test.loc[sorted_index]
    # 예측값도 동일한 인덱스 순서로 정렬
    y_pred_sorted = pd.Series(y_pred, index=X_test.index).loc[sorted_index]

    # 샘플이 많을 경우, 앞에서부터 500개만 추출하여 시각화
    sample_y = y_test_sorted[:500].reset_index(drop=True)
    sample_pred = y_pred_sorted[:500].reset_index(drop=True)

    # 실제값과 예측값을 선 그래프로 비교
    plt.figure(figsize=(14, 6))
    plt.plot(sample_y, label="실제값", color="blue")
    plt.plot(sample_pred, label="예측값", color="orange", alpha=0.7)
    plt.title(f"{title} (샘플 500개)")
    plt.xlabel("시간 순 샘플")
    plt.ylabel("PM10")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_scatter_prediction(y_true, y_pred, title="예측 결과"):
    """
    실제값과 예측값의 산점도를 그려 예측 성능을 직관적으로 확인하는 함수입니다.
    대각선(이상적 예측선)도 함께 표시하여 예측이 얼마나 실제값과 일치하는지 시각적으로 보여줍니다.
    """
    plt.figure(figsize=(7, 7))
    # 실제값과 예측값의 산점도
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    # 대각선(이상적 예측선) 추가
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("실제 PM10")
    plt.ylabel("예측 PM10")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_regression_metrics_bar(scores: dict, model_name: str):
    """
    회귀 모델의 성능 지표(MSE, RMSE, MAE, R²)를 막대그래프로 시각화하는 함수입니다.

    Parameters:
        scores (dict): {"MSE": ..., "RMSE": ..., "MAE": ..., "R²": ...} 형식의 성능 지표 딕셔너리
        model_name (str): 모델 이름 (그래프 제목에 사용)
    """
    print(f"\n{model_name} 성능 지표")
    for metric, value in scores.items():
        print(f" - {metric}: {value:.4f}")  # 소수점 4자리로 출력

    # 성능 지표를 막대그래프로 시각화
    plt.figure(figsize=(8, 5))
    sns.barplot(x = scores.keys(), y = scores.values())
    plt.title(f"{model_name} 성능 지표")
    plt.ylabel("값")
    plt.ylim(0, max(scores.values()) * 1.2)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()