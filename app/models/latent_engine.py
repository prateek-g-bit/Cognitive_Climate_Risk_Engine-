
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from hmmlearn.hmm import GaussianHMM


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

    df = df.sort_values(

        ["district_name", "year", "month"]

    ).reset_index(drop=True)

    return df


# ============================================================
# OBSERVATION FEATURES
# ============================================================

def prepare_observations(df):

    obs_cols = [

        "ndvi_anomaly",
        "ndwi_anomaly",

        "lst_anomaly",

        "precipitation_anomaly",

        "drought_signal",

        "ndvi_mean_change",

        "ndwi_mean_change",

        "precipitation_mean_change",

        "lst_mean_change"
    ]

    # REMOVE MISSING
    df = df.dropna(

        subset=obs_cols

    ).reset_index(drop=True)

    return df, obs_cols


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(df, obs_cols):

    scaler = StandardScaler()

    X = scaler.fit_transform(

        df[obs_cols]
    )

    return X, scaler


# ============================================================
# TRAIN HMM
# ============================================================

def train_hmm(X, df):

    hmm = GaussianHMM(

        n_components=5,

        covariance_type="diag",

        n_iter=1000,

        tol=1e-4,

        random_state=42
    )

    # TEMPORAL SEQUENCES
    sequence_lengths = (

        df.groupby("district_name")
          .size()
          .tolist()
    )

    # TRAIN
    hmm.fit(

        X,

        lengths=sequence_lengths
    )

    return hmm


# ============================================================
# LATENT STATE INFERENCE
# ============================================================

def infer_latent_states(df, hmm, X):

    df["latent_state"] = hmm.predict(X)
    
    # ============================================================
    # INTERPRETABLE STATE LABELS
    # ============================================================
    
    state_labels = {
    
        0: "Dry Stress",
    
        1: "Recovery",
    
        2: "Stable",
    
        3: "Extreme Crisis",
    
        4: "Transition"
    }
    
    df["latent_label"] = (
    
        df["latent_state"]
    
        .map(state_labels)
    )
    
    return df


# ============================================================
# TRANSITION MATRIX
# ============================================================

def get_transition_matrix(hmm):

    transition_matrix = pd.DataFrame(

        hmm.transmat_
    )

    return transition_matrix


# ============================================================
# MASTER LATENT ENGINE
# ============================================================

def run_latent_engine(df):

    df = df.copy()

    # PREPROCESS
    df = preprocess_data(df)

    # OBSERVATIONS
    df, obs_cols = prepare_observations(df)

    # NORMALIZE
    X, scaler = normalize_features(
        df,
        obs_cols
    )

    # TRAIN HMM
    hmm = train_hmm(
        X,
        df
    )

    # INFER STATES
    df = infer_latent_states(
        df,
        hmm,
        X
    )

    # TRANSITION MATRIX
    transition_matrix = get_transition_matrix(hmm)

    return {

        "model": hmm,

        "data": df,

        "transition_matrix": transition_matrix,

        "features": obs_cols
    }
