
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    return df


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_data(df):

    # SORT TEMPORALLY
    df = df.sort_values(
        ["district_name", "year", "month"]
    ).reset_index(drop=True)

    # REMOVE DUPLICATES
    df = df.drop_duplicates(
        subset=["district_name", "year", "month"]
    )

    # FILL LULC
    df["lulc_majority_class"] = (

        df.groupby("district_name")["lulc_majority_class"]
        .transform(lambda x: x.ffill().bfill())
    )

    # INTERPOLATE POPULATION
    df["population_mean"] = (

        df.groupby("district_name")["population_mean"]
        .transform(
            lambda x: x.interpolate(
                method="linear",
                limit_direction="both"
            )
        )
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    temporal_cols = [

        "ndvi_mean",
        "ndwi_mean",

        "precipitation_mean",

        "lst_mean",

        "drought_signal"
    ]

    # LAG FEATURES
    for col in temporal_cols:

        df[f"{col}_lag1"] = (

            df.groupby("district_name")[col]
            .shift(1)
        )

        df[f"{col}_lag3"] = (

            df.groupby("district_name")[col]
            .shift(3)
        )

    # ROLLING FEATURES
    for col in temporal_cols:

        df[f"{col}_roll3"] = (

            df.groupby("district_name")[col]
            .transform(
                lambda x: x.rolling(3).mean()
            )
        )

    # CHANGE FEATURES
    for col in temporal_cols:

        df[f"{col}_change"] = (

            df.groupby("district_name")[col]
            .diff()
        )

    engineered_cols = []
    
    for col in temporal_cols:
    
        engineered_cols.extend([
            f"{col}_lag1",
            f"{col}_lag3",
            f"{col}_roll3",
            f"{col}_change"
        ])
    
    df = df.dropna(
        subset=engineered_cols
    )

    return df


# ============================================================
# RISK SCORING
# ============================================================

def risk_scoring(df):

    risk_cols = [

        "drought_signal",

        "lst_anomaly",

        "ndvi_anomaly",

        "precipitation_anomaly"
    ]

    scaler = StandardScaler()

    df[risk_cols] = scaler.fit_transform(
        df[risk_cols]
    )

    # RISK SCORE
    df["risk_score"] = (

          0.35 * df["drought_signal"]

        + 0.25 * df["lst_anomaly"]

        - 0.25 * df["ndvi_anomaly"]

        - 0.15 * df["precipitation_anomaly"]
    )

    # RISK STATES
    df["risk_state"] = pd.qcut(

        df["risk_score"],

        q=4,

        labels=[
            "Stable",
            "Alert",
            "Stress",
            "Crisis"
        ]
    )

    return df


# ============================================================
# TARGET GENERATION
# ============================================================

def create_targets(df):

    df["target_state"] = (

        df.groupby("district_name")["risk_state"]
        .shift(-1)
    )

    df = df.dropna().reset_index(drop=True)

    risk_mapping = {

        "Stable": 0,
        "Alert": 1,
        "Stress": 2,
        "Crisis": 3
    }

    df["target_state"] = (
        df["target_state"]
        .map(risk_mapping)
    )

    return df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):

    exclude_cols = [

        "uid",

        "district_name",

        "risk_state",

        "target_state"
    ]

    feature_cols = [

        col for col in df.columns
        if col not in exclude_cols
    ]

    X = df[feature_cols]

    y = df["target_state"]

    return X, y, feature_cols


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(X_train, y_train):

    model = XGBClassifier(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="multi:softprob",

        num_class=4,

        eval_metric="mlogloss",

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    report = classification_report(
        y_test,
        preds
    )

    return report


# ============================================================
# MASTER ENGINE PIPELINE
# ============================================================

def run_baseline_engine(df):

    df = df.copy()

    # PREPROCESS
    df = preprocess_data(df)

    # FEATURES
    df = feature_engineering(df)

    # RISK
    df = risk_scoring(df)

    # TARGETS
    df = create_targets(df)

    # FINAL FEATURES
    X, y, feature_cols = prepare_features(df)

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    # TRAIN MODEL
    model = train_model(X_train, y_train)

    # EVALUATE
    report = evaluate_model(
        model,
        X_test,
        y_test
    )

    return {

        "model": model,

        "report": report,

        "features": feature_cols,

        "data": df
    }
