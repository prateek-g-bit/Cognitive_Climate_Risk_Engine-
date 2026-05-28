
# ============================================================
# MASTER PIPELINE
# ============================================================

import pandas as pd

from app.models.baseline_engine import run_baseline_engine

from app.models.latent_engine import run_latent_engine

from app.models.dynamics_engine import run_dynamics_engine

from app.models.spatial_engine import run_spatial_engine

from app.models.salinity_engine import run_salinity_engine

from app.models.causal_engine import run_causal_engine


def run_master_pipeline(input_csv):

    # ==========================================
    # LOAD INPUT
    # ==========================================

    df = pd.read_csv(input_csv)

    # ==========================================
    # BASELINE ENGINE
    # ==========================================

    baseline_results = run_baseline_engine(df)

    baseline_df = baseline_results["data"]

    print("BASELINE DONE")

    # ==========================================
    # LATENT ENGINE
    # ==========================================

    latent_results = run_latent_engine(

        baseline_df
    )

    latent_df = latent_results["data"]

    print("LATENT DONE")

    # ==========================================
    # DYNAMICS ENGINE
    # ==========================================

    dynamics_results = run_dynamics_engine(

        latent_df
    )

    dynamics_df = dynamics_results["data"]

    print("DYNAMICS DONE")

    # ==========================================
    # SPATIAL ENGINE
    # ==========================================

    spatial_results = run_spatial_engine(

        dynamics_df
    )

    spatial_df = spatial_results["data"]

    print("SPATIAL DONE")

    # ==========================================
    # SALINITY ENGINE
    # ==========================================

    salinity_results = run_salinity_engine(

        spatial_df
    )

    salinity_df = salinity_results["data"]

    print("SALINITY DONE")

    # ==========================================
    # CAUSAL ENGINE
    # ==========================================

    causal_results = run_causal_engine(

        salinity_df
    )

    print("CAUSAL DONE")

    # ==========================================
    # RETURN
    # ==========================================

    return {

        "baseline": baseline_results,

        "latent": latent_results,

        "dynamics": dynamics_results,

        "spatial": spatial_results,

        "salinity": salinity_results,

        "causal": causal_results,

        "final_dataframe": salinity_df
    }
