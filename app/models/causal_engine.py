
# ============================================================
# CAUSAL ENGINE
# ============================================================

import pandas as pd
import numpy as np

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(df):

    # ------------------------------------------------
    # USE FINAL INTEGRATED RISK
    # ------------------------------------------------

    causal_columns = [

        "ndvi_mean",

        "lst_mean",

        "precipitation_mean",

        "final_risk_score"
    ]

    # ------------------------------------------------
    # KEEP ONLY AVAILABLE COLUMNS
    # ------------------------------------------------

    causal_columns = [

        col for col in causal_columns

        if col in df.columns
    ]

    causal_df = df[causal_columns].copy()

    # ------------------------------------------------
    # DISCRETIZATION
    # ------------------------------------------------

    for col in causal_columns:

        causal_df[col] = pd.qcut(

            causal_df[col],

            q=3,

            labels=False,

            duplicates="drop"
        )

    causal_df = (

        causal_df
        .dropna()
        .reset_index(drop=True)
    )

    return causal_df


# ============================================================
# BUILD CAUSAL MODEL
# ============================================================

def build_causal_model():

    edges = [

        ("lst_mean", "ndvi_mean"),

        ("ndvi_mean", "final_risk_score"),

        ("lst_mean", "final_risk_score"),

        ("precipitation_mean", "final_risk_score")
    ]

    model = DiscreteBayesianNetwork(edges)

    return model


# ============================================================
# FIT MODEL
# ============================================================

def fit_causal_model(

    model,

    causal_df
):

    model.fit(causal_df)

    return model


# ============================================================
# RUN INFERENCE
# ============================================================

def run_causal_inference(infer):

    query = infer.query(

        variables=["final_risk_score"]
    )

    risk_distribution = query.values

    return risk_distribution


# ============================================================
# COMPUTE EXPECTED SCORE
# ============================================================

def compute_risk_score(

    risk_distribution
):

    expected_risk_score = 0

    for i, prob in enumerate(risk_distribution):

        expected_risk_score += i * prob

    return float(expected_risk_score)


# ============================================================
# GENERATE POLICY ACTIONS
# ============================================================

def generate_policy_actions(score):

    # ------------------------------------------------
    # LOW RISK
    # ------------------------------------------------

    if score < 0.75:

        actions = [

            "Maintain regular monitoring",

            "Continue existing mitigation policies"
        ]

    # ------------------------------------------------
    # MODERATE RISK
    # ------------------------------------------------

    elif score < 1.5:

        actions = [

            "Increase climate surveillance",

            "Strengthen irrigation preparedness",

            "Deploy district monitoring teams"
        ]

    # ------------------------------------------------
    # HIGH RISK
    # ------------------------------------------------

    else:

        actions = [

            "Issue drought warning",

            "Activate emergency response systems",

            "Deploy water resource intervention",

            "Initiate agricultural protection measures"
        ]

    return actions


# ============================================================
# GENERATE EXPLANATIONS
# ============================================================

def generate_explanations(

    risk_distribution,

    score
):

    dominant_state = int(
        np.argmax(risk_distribution)
    )

    explanations = {

        "dominant_risk_state":

        dominant_state,

        "dominant_probability":

        float(np.max(risk_distribution)),

        "expected_risk_score":

        float(score)
    }

    return explanations


# ============================================================
# RUN CAUSAL ENGINE
# ============================================================

def run_causal_engine(df):

    df = df.copy()

    # ------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------

    causal_df = preprocess_data(df)

    # ------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------

    model = build_causal_model()

    # ------------------------------------------------
    # FIT MODEL
    # ------------------------------------------------

    model = fit_causal_model(

        model,

        causal_df
    )

    # ------------------------------------------------
    # INFERENCE
    # ------------------------------------------------

    infer = VariableElimination(model)

    risk_distribution = run_causal_inference(
        infer
    )

    # ------------------------------------------------
    # EXPECTED SCORE
    # ------------------------------------------------

    expected_risk_score = compute_risk_score(
        risk_distribution
    )

    # ------------------------------------------------
    # POLICY ACTIONS
    # ------------------------------------------------

    recommended_actions = (

        generate_policy_actions(

            expected_risk_score
        )
    )

    # ------------------------------------------------
    # EXPLANATIONS
    # ------------------------------------------------

    explanations = generate_explanations(

        risk_distribution,

        expected_risk_score
    )

    # ------------------------------------------------
    # RETURN
    # ------------------------------------------------

    return {

        "data": df,

        "model": model,

        "risk_distribution":
        risk_distribution,

        "expected_risk_score":
        expected_risk_score,

        "recommended_actions":
        recommended_actions,

        "explanations":
        explanations
    }
