import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_decision_tree(X_train, y_train, max_depth=4, random_state=42):
    """의사결정트리 모델 학습"""
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """랜덤포레스트 모델 학습
    
    Parameters:
    - X_train: 학습 데이터의 특성
    - y_train: 학습 데이터의 타겟
    - n_estimators: 생성할 트리의 개수 (기본값: 100)
    - random_state: 랜덤 시드 (기본값: 42)
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def print_tree_rules(model, feature_names):
    """트리 규칙 텍스트 출력"""
    return export_text(model, feature_names=feature_names)

def split_features_and_target(df, target_column, use_pm25=True, test_size=0.2, random_state=42):
    """
    PM10 예측을 위한 특성과 타깃을 분리하고, 학습/테스트 세트로 분할합니다.

    Parameters:
    - df (pd.DataFrame): 입력 데이터프레임
    - target_column (str): 예측 대상 컬럼명 (예: "pm10")
    - use_pm25 (bool): True이면 pm25 포함, False이면 제외
    - test_size (float): 테스트 세트 비율 (default: 0.2)
    - random_state (int): 난수 시드 (default: 42)

    Returns:
    - X (pd.DataFrame): 전체 특성
    - y (pd.Series): 전체 타깃
    - X_train, X_test, y_train, y_test: 학습/테스트 분할된 데이터
    """
    # 공통 피처
    feature_columns = ["평균기온(°C)", "일강수량(mm)", "평균 풍속(m/s)", "month"]
    
    # pm25 포함 여부에 따라 컬럼 추가
    if use_pm25:
        feature_columns.insert(0, "pm25")

    # 결측치 제거
    df_cleaned = df.dropna(subset=feature_columns + [target_column])

    # 특성과 타깃 분리
    X = df_cleaned[feature_columns]
    y = df_cleaned[target_column]

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X, y, X_train, X_test, y_train, y_test

def predict_pm10(model, input_df, use_pm25=True):
    """
    학습된 회귀 모델을 기반으로 주어진 입력 조건에서 PM10을 예측하는 함수

    Parameters:
    - model: 학습된 회귀 모델 (예: DecisionTreeRegressor, RandomForestRegressor 등)
    - input_df: 예측에 사용할 입력 데이터프레임
                (필수 컬럼: pm25, 평균기온(°C), 일강수량(mm), 평균 풍속(m/s), month)

    Returns:
    - '예측_PM10' 컬럼이 추가된 데이터프레임 반환
    """
    # 학습에 사용된 feature들 (순서 및 이름 일치 필수)
    feature_columns = ["평균기온(°C)", "일강수량(mm)", "평균 풍속(m/s)", "month"]

    if use_pm25:
        feature_columns.insert(0, "pm25")

    # 입력 데이터에서 필요한 피처 추출
    X = input_df[feature_columns]

    # 예측 수행
    predictions = model.predict(X)

    # 결과 추가
    result_df = input_df.copy()
    result_df["예측_PM10"] = predictions

    return result_df

def get_evaluate_regression_scores(y_true, y_pred):
    """
    회귀 모델의 성능을 평가하는 함수입니다.
    MSE, RSME, MAE, R² Score를 출력합니다.

    Parameters:
        y_true (array-like): 실제 값
        y_pred (array-like): 예측 값
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}