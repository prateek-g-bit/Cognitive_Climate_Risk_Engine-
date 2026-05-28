
# ============================================================
# FASTAPI DROUGHT INTELLIGENCE API
# ============================================================

from fastapi import FastAPI

from app.pipeline.master_pipeline import run_master_pipeline


# ============================================================
# INITIALIZE API
# ============================================================

app = FastAPI(

    title="Drought Intelligence System",

    description="Spatio-Temporal Climate Risk Intelligence API",

    version="1.0.0"
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")

def home():

    return {

        "message":

        "Drought Intelligence System API Running"
    }


# ============================================================
# PIPELINE ENDPOINT
# ============================================================

@app.get("/run-pipeline")

def execute_pipeline():

    path = (

        "/kaggle/input/datasets/"
        "pg7729600/"
        "nabard-dicra/"
        "FINAL_CLEAN_DATASET.csv"
    )

    results = run_master_pipeline(path)

    final_df = results["final_dataframe"]

    return {

        "status": "success",

        "rows_processed":

        int(final_df.shape[0]),

        "columns_generated":

        int(final_df.shape[1]),

        "final_risk_states":

        final_df[
            "final_risk_state"
        ].value_counts().to_dict(),

        "drought_types":

        final_df[
            "drought_type"
        ].value_counts().to_dict(),

        "recommended_actions":

        results["causal"][
            "recommended_actions"
        ]
    }
