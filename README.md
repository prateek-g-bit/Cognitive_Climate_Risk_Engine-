
# Drought Climate Risk Intelligence System

## Overview

This project implements a modular spatio-temporal drought intelligence system integrating:

- Baseline climate risk modeling
- Hidden Markov latent state inference
- Temporal recursive dynamics
- Spatial graph propagation
- Salinity-aware drought intelligence
- Bayesian causal reasoning
- Policy recommendation generation

The system is deployable using FastAPI.

---

## Architecture

Raw Climate Features
↓
Baseline ML Risk
↓
Latent HMM States
↓
Temporal Dynamics
↓
Spatial Propagation
↓
Salinity Intelligence
↓
Bayesian Causal Inference
↓
Policy Recommendations

---

## Technologies

- Python
- FastAPI
- Scikit-learn
- hmmlearn
- NetworkX
- pgmpy
- Pandas
- NumPy

---

## API Endpoints

### Root Endpoint

GET /

Returns API status.

### Pipeline Endpoint

GET /run-pipeline

Executes the complete drought intelligence pipeline.

---

## Deployment

Designed for deployment using:

- Render
- Railway
- HuggingFace Spaces

---

## Author

Prateek Gupta
