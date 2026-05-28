
# ============================================================
# DYNAMICS ENGINE
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(df):

    df = df.sort_values(

        ["district_name", "year", "month"]

    ).reset_index(drop=True)

    return df


# ============================================================
# COMPUTE COMPONENT INDICES
# ============================================================

def compute_component_indices(df):

    # ------------------------------------------------
    # HAZARD FEATURES
    # ------------------------------------------------

    hazard_cols = [

        "precipitation_anomaly",

        "lst_anomaly",

        "drought_signal",

        "ndvi_anomaly"
    ]

    # ------------------------------------------------
    # VULNERABILITY FEATURES
    # ------------------------------------------------

    vulnerability_cols = [

        "rwi_mean",

        "ndvi_std",

        "lst_std"
    ]

    # ------------------------------------------------
    # EXPOSURE FEATURES
    # ------------------------------------------------

    exposure_cols = [

        "population_mean",

        "area"
    ]

    scaler = MinMaxScaler()

    # ------------------------------------------------
    # NORMALIZE
    # ------------------------------------------------

    H = scaler.fit_transform(
        df[hazard_cols]
    )

    V = scaler.fit_transform(
        df[vulnerability_cols]
    )

    E = scaler.fit_transform(
        df[exposure_cols]
    )

    # ------------------------------------------------
    # AGGREGATE INDICES
    # ------------------------------------------------

    df["hazard_index"] = H.mean(axis=1)

    df["vulnerability_index"] = V.mean(axis=1)

    df["exposure_index"] = E.mean(axis=1)

    return df


# ============================================================
# APPLY LATENT RISK
# ============================================================

def apply_latent_risk(df):

    state_risk_weights = {

        0: 0.65,

        1: 0.40,

        2: 0.20,

        3: 0.95,

        4: 0.35
    }

    df["latent_risk"] = (

        df["latent_state"]

        .map(state_risk_weights)
    )

    return df


# ============================================================
# COMPUTE BASE RISK
# ============================================================

def compute_base_risk(df):

    df["base_risk"] = (

        0.35 * df["hazard_index"]

        +

        0.25 * df["vulnerability_index"]

        +

        0.20 * df["exposure_index"]

        +

        0.20 * df["latent_risk"]
    )

    return df


# ============================================================
# COMPUTE TEMPORAL DYNAMIC RISK
# ============================================================

def compute_dynamic_risk(

    df,

    alpha=0.7
):

    df["dynamic_risk"] = 0.0

    # ------------------------------------------------
    # DISTRICT-WISE TEMPORAL RECURSION
    # ------------------------------------------------

    for district in df["district_name"].unique():

        idx = df[

            df["district_name"] == district

        ].index

        risks = []

        previous_risk = 0

        for i in idx:

            current_base = df.loc[
                i,
                "base_risk"
            ]

            current_risk = (

                alpha * previous_risk

                +

                (1 - alpha) * current_base
            )

            risks.append(
                current_risk
            )

            previous_risk = current_risk

        df.loc[idx, "dynamic_risk"] = risks

    return df


# ============================================================
# NORMALIZE DYNAMIC RISK
# ============================================================

def normalize_risk(df):

    scaler = MinMaxScaler()

    df["dynamic_risk_score"] = (

        scaler.fit_transform(

            df[["dynamic_risk"]]

        )
    )

    return df


# ============================================================
# CREATE RISK STATES
# ============================================================

def create_risk_states(df):

    q1 = df[
        "dynamic_risk_score"
    ].quantile(0.25)

    q2 = df[
        "dynamic_risk_score"
    ].quantile(0.50)

    q3 = df[
        "dynamic_risk_score"
    ].quantile(0.75)

    conditions = [

        df["dynamic_risk_score"] < q1,

        (
            df["dynamic_risk_score"] >= q1
        )

        &

        (
            df["dynamic_risk_score"] < q2
        ),

        (
            df["dynamic_risk_score"] >= q2
        )

        &

        (
            df["dynamic_risk_score"] < q3
        ),

        df["dynamic_risk_score"] >= q3
    ]

    labels = [

        "Low",

        "Moderate",

        "High",

        "Extreme"
    ]

    df["risk_state"] = np.select(

        conditions,

        labels,

        default="Unknown"
    )

    return df


# ============================================================
# RUN DYNAMICS ENGINE
# ============================================================

def run_dynamics_engine(df):

    df = df.copy()

    # ------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------

    df = preprocess_data(df)

    # ------------------------------------------------
    # COMPONENT INDICES
    # ------------------------------------------------

    df = compute_component_indices(df)

    # ------------------------------------------------
    # LATENT RISK
    # ------------------------------------------------

    df = apply_latent_risk(df)

    # ------------------------------------------------
    # BASE RISK
    # ------------------------------------------------

    df = compute_base_risk(df)

    # ------------------------------------------------
    # TEMPORAL DYNAMICS
    # ------------------------------------------------

    df = compute_dynamic_risk(df)

    # ------------------------------------------------
    # NORMALIZE
    # ------------------------------------------------

    df = normalize_risk(df)

    # ------------------------------------------------
    # CREATE STATES
    # ------------------------------------------------

    df = create_risk_states(df)

    # ------------------------------------------------
    # RETURN
    # ------------------------------------------------

    return {

        "data": df,

        "risk_distribution":
        df["risk_state"].value_counts()
    }
