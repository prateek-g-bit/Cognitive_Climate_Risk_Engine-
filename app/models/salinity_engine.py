
# ============================================================
# SALINITY ENGINE
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_data(df):

    df = df.sort_values(

        ["district_name", "year", "month"]

    ).reset_index(drop=True)

    return df


# ============================================================
# ADD SALINITY FEATURES
# ============================================================

def add_salinity_features(df):

    # ------------------------------------------------
    # PROTOTYPE SYNTHETIC STREAMS
    # ------------------------------------------------

    np.random.seed(42)

    df["ndsi_mean"] = np.random.uniform(

        0.1,
        0.8,
        len(df)
    )

    df["smap_soil_moisture"] = np.random.uniform(

        0.2,
        0.9,
        len(df)
    )

    df["river_discharge"] = np.random.uniform(

        10,
        100,
        len(df)
    )

    df["groundwater_ec"] = np.random.uniform(

        100,
        3000,
        len(df)
    )

    return df


# ============================================================
# COMPUTE SALINITY PRESSURE
# ============================================================

def compute_salinity_pressure(df):

    scaler = MinMaxScaler()

    cols = [

        "ndsi_mean",

        "groundwater_ec",

        "river_discharge"
    ]

    # ------------------------------------------------
    # NORMALIZE
    # ------------------------------------------------

    df[cols] = scaler.fit_transform(
        df[cols]
    )

    # ------------------------------------------------
    # SALINITY PRESSURE
    # ------------------------------------------------

    df["salinity_pressure"] = (

        0.4 * df["ndsi_mean"]

        +

        0.4 * df["groundwater_ec"]

        +

        0.2 * (1 - df["river_discharge"])
    )

    return df


# ============================================================
# DROUGHT TYPE CLASSIFICATION
# ============================================================

def classify_drought_type(row):

    # ------------------------------------------------
    # VEGETATION DECLINE
    # ------------------------------------------------

    if row["ndvi_mean_change"] < -0.05:

        # --------------------------------------------
        # LOW MOISTURE + LOW SALINITY
        # --------------------------------------------

        if row["smap_soil_moisture"] < 0.3:

            if row["ndsi_mean"] < 0.4:

                return "standard_drought"

        # --------------------------------------------
        # ADEQUATE MOISTURE + HIGH SALINITY
        # --------------------------------------------

        else:

            if row["ndsi_mean"] > 0.6:

                return "osmotic_drought"

    return "normal"


# ============================================================
# APPLY DROUGHT LOGIC
# ============================================================

def compute_drought_logic(df):

    df["drought_type"] = df.apply(

        classify_drought_type,

        axis=1
    )

    return df


# ============================================================
# COMPUTE SALINITY RISK
# ============================================================

def compute_salinity_risk(df):

    df["salinity_risk"] = (

        0.35 * df["salinity_pressure"]

        +

        0.25 * df["latent_risk"]

        +

        0.20 * df["drought_signal"]

        +

        0.20 * df["lst_anomaly"]
    )

    return df


# ============================================================
# COMPUTE FINAL INTEGRATED RISK
# ============================================================

def compute_final_risk(df):

    # ------------------------------------------------
    # FUSE:
    # - SPATIO TEMPORAL RISK
    # - SALINITY RISK
    # ------------------------------------------------

    df["final_risk_score"] = (

        0.7 * df["spatio_temporal_risk"]

        +

        0.3 * df["salinity_risk"]
    )

    return df


# ============================================================
# CREATE FINAL STATES
# ============================================================

def create_final_states(df):

    df["final_risk_state"] = pd.qcut(

        df["final_risk_score"],

        q=4,

        labels=[

            "Low",

            "Moderate",

            "High",

            "Extreme"
        ]
    )

    return df


# ============================================================
# RUN SALINITY ENGINE
# ============================================================

def run_salinity_engine(df):

    df = df.copy()

    # ------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------

    df = preprocess_data(df)

    # ------------------------------------------------
    # ADD FEATURES
    # ------------------------------------------------

    df = add_salinity_features(df)

    # ------------------------------------------------
    # SALINITY PRESSURE
    # ------------------------------------------------

    df = compute_salinity_pressure(df)

    # ------------------------------------------------
    # DROUGHT LOGIC
    # ------------------------------------------------

    df = compute_drought_logic(df)

    # ------------------------------------------------
    # SALINITY RISK
    # ------------------------------------------------

    df = compute_salinity_risk(df)

    # ------------------------------------------------
    # FINAL FUSION
    # ------------------------------------------------

    df = compute_final_risk(df)

    # ------------------------------------------------
    # FINAL STATES
    # ------------------------------------------------

    df = create_final_states(df)

    # ------------------------------------------------
    # RETURN
    # ------------------------------------------------

    return {

        "data": df,

        "risk_distribution":
        df["final_risk_state"].value_counts(),

        "drought_distribution":
        df["drought_type"].value_counts()
    }
