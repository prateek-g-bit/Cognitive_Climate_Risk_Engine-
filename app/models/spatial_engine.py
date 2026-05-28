
# ============================================================
# SPATIAL ENGINE
# ============================================================

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

import networkx as nx


# ============================================================
# CREATE DISTRICT NODES
# ============================================================

def create_district_nodes(df):

    district_nodes = (

        df[[
            "district_name",
            "centroid_x",
            "centroid_y"
        ]]

        .drop_duplicates()

        .reset_index(drop=True)
    )

    return district_nodes


# ============================================================
# BUILD SPATIAL GRAPH
# ============================================================

def build_spatial_graph(

    district_nodes,

    k=4
):

    coords = district_nodes[[
        "centroid_x",
        "centroid_y"
    ]].values

    nbrs = NearestNeighbors(

        n_neighbors=k + 1,

        metric="euclidean"
    )

    nbrs.fit(coords)

    distances, indices = nbrs.kneighbors(coords)

    G = nx.Graph()

    # ------------------------------------------------
    # ADD NODES
    # ------------------------------------------------

    for idx, row in district_nodes.iterrows():

        G.add_node(

            row["district_name"],

            pos=(

                row["centroid_x"],

                row["centroid_y"]
            )
        )

    # ------------------------------------------------
    # ADD EDGES
    # ------------------------------------------------

    for i in range(len(district_nodes)):

        source = district_nodes.iloc[i][
            "district_name"
        ]

        for j in range(1, k + 1):

            target_idx = indices[i][j]

            target = district_nodes.iloc[
                target_idx
            ]["district_name"]

            distance = distances[i][j]

            G.add_edge(

                source,

                target,

                weight=1 / (distance + 1e-6)
            )

    return G


# ============================================================
# COMPUTE NEIGHBOR RISK
# ============================================================

def compute_neighbor_risk(df, G):

    neighbor_risk = []

    for idx, row in df.iterrows():

        district = row["district_name"]

        neighbors = list(
            G.neighbors(district)
        )

        # --------------------------------------------
        # SAME MONTH + SAME YEAR NEIGHBOR RISK
        # --------------------------------------------

        temp_df = df[

            (df["district_name"]
             .isin(neighbors))

            &

            (df["year"] == row["year"])

            &

            (df["month"] == row["month"])
        ]

        # --------------------------------------------
        # NEIGHBOR RISK
        # --------------------------------------------

        if len(temp_df) > 0:

            spatial_risk = (

                temp_df[
                    "dynamic_risk_score"
                ].mean()
            )

        else:

            spatial_risk = row[
                "dynamic_risk_score"
            ]

        neighbor_risk.append(
            spatial_risk
        )

    df["neighbor_risk"] = neighbor_risk

    return df


# ============================================================
# COMPUTE SPATIO-TEMPORAL RISK
# ============================================================

def compute_spatio_temporal_risk(

    df,

    alpha=0.7,

    beta=0.3
):

    df["spatio_temporal_risk"] = (

        alpha * df["dynamic_risk_score"]

        +

        beta * df["neighbor_risk"]
    )

    return df


# ============================================================
# RUN SPATIAL ENGINE
# ============================================================

def run_spatial_engine(df):

    df = df.copy()

    # ------------------------------------------------
    # CREATE NODES
    # ------------------------------------------------

    district_nodes = create_district_nodes(df)

    # ------------------------------------------------
    # BUILD GRAPH
    # ------------------------------------------------

    G = build_spatial_graph(
        district_nodes
    )

    # ------------------------------------------------
    # COMPUTE NEIGHBOR RISK
    # ------------------------------------------------

    df = compute_neighbor_risk(
        df,
        G
    )

    # ------------------------------------------------
    # COMPUTE FINAL SPATIO-TEMPORAL RISK
    # ------------------------------------------------

    df = compute_spatio_temporal_risk(
        df
    )

    # ------------------------------------------------
    # RETURN
    # ------------------------------------------------

    return {

        "data": df,

        "graph": G,

        "nodes": G.number_of_nodes(),

        "edges": G.number_of_edges()
    }
