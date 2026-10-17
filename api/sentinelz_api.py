"""
SentinelZ — Real-Time In-Flight Transaction Fraud Prevention API
Target Architecture: IBM LinuxONE (s390x)
Theme 1: Real-Time AI for Critical Decisions (Detect. Decide. Respond.)
"""

import time
import os
import joblib
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="SentinelZ In-Flight Fraud Prevention Engine",
    description="Sub-millisecond in-flight transaction fraud scoring engine running on IBM LinuxONE (s390x).",
    version="1.0.0"
)

# Enable CORS for front-end dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model references
MODELS_DIR = os.getenv("SENTINELZ_MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

lgb_model = None
xgb_model = None
rf_model = None
scaler = None

# Metrics tracking
stats = {
    "total_transactions": 0,
    "approved_transactions": 0,
    "blocked_transactions": 0,
    "total_latency_ms": 0.0,
    "min_latency_ms": float("inf"),
    "max_latency_ms": 0.0
}

FRAUD_BLOCK_THRESHOLD = 0.55  # Decisions >= 55% risk probability are intercepted

def load_artifacts():
    global lgb_model, xgb_model, rf_model, scaler
    try:
        lgb_path = os.path.join(MODELS_DIR, "lgb_model.pkl")
        xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
        rf_path  = os.path.join(MODELS_DIR, "rf_model.pkl")
        sc_path  = os.path.join(MODELS_DIR, "scaler.pkl")

        lgb_model = joblib.load(lgb_path)
        xgb_model = joblib.load(xgb_path)
        rf_model  = joblib.load(rf_path)
        scaler    = joblib.load(sc_path)
        print("✅ SentinelZ models and scaler loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading models from {MODELS_DIR}: {e}")

@app.on_event("startup")
def startup_event():
    load_artifacts()

class TransactionPayload(BaseModel):
    transaction_id: Optional[str] = None
    features: List[float] = Field(
        ..., 
        description="Feature vector of 30 values (V1..V28 + scaled_Amount + scaled_Time)"
    )

class BatchTransactionPayload(BaseModel):
    transactions: List[TransactionPayload]

class ScoringDecision(BaseModel):
    transaction_id: str
    decision: str  # "APPROVE" or "BLOCK"
    risk_score: float  # 0.0 - 100.0
    fraud_probability: float
    model_probabilities: dict
    latency_ms: float
    platform: str
    recommendation: str

DASHBOARD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html"))

@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    if os.path.exists(DASHBOARD_PATH):
        with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>SentinelZ API Online on IBM LinuxONE (s390x)</h1><p>Visit /docs for API schema.</p>")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "architecture": "s390x",
        "platform": "IBM LinuxONE",
        "models_loaded": {
            "lightgbm": lgb_model is not None,
            "xgboost": xgb_model is not None,
            "random_forest": rf_model is not None,
            "scaler": scaler is not None
        },
        "stats": {
            "total_scored": stats["total_transactions"],
            "approved": stats["approved_transactions"],
            "blocked": stats["blocked_transactions"],
            "avg_latency_ms": round(
                stats["total_latency_ms"] / max(1, stats["total_transactions"]), 4
            ) if stats["total_transactions"] > 0 else 0.0
        }
    }

@app.post("/score", response_model=ScoringDecision)
def score_transaction(payload: TransactionPayload):
    if lgb_model is None or xgb_model is None or rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    t0 = time.perf_counter()
    
    # Validate feature dimensions
    vec = np.array(payload.features).reshape(1, -1)
    if vec.shape[1] != 30:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 30 features (V1..V28, scaled_Amount, scaled_Time), received {vec.shape[1]}"
        )

    # Hybrid multi-model scoring
    p_lgb = float(lgb_model.predict_proba(vec)[0][1])
    p_xgb = float(xgb_model.predict_proba(vec)[0][1])
    p_rf  = float(rf_model.predict_proba(vec)[0][1])

    # Soft-voting ensemble: 45% LGBM + 35% XGB + 20% RF
    ensemble_prob = float((0.45 * p_lgb) + (0.35 * p_xgb) + (0.20 * p_rf))
    risk_score = round(ensemble_prob * 100.0, 2)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    decision = "BLOCK" if ensemble_prob >= FRAUD_BLOCK_THRESHOLD else "APPROVE"
    recommendation = (
        "Pre-settlement freeze triggered. High probability of authorized push payment (APP) scam or account takeover."
        if decision == "BLOCK"
        else "Authorized for low-friction real-time settlement."
    )

    # Update global metrics
    stats["total_transactions"] += 1
    if decision == "BLOCK":
        stats["blocked_transactions"] += 1
    else:
        stats["approved_transactions"] += 1
    stats["total_latency_ms"] += latency_ms
    stats["min_latency_ms"] = min(stats["min_latency_ms"], latency_ms)
    stats["max_latency_ms"] = max(stats["max_latency_ms"], latency_ms)

    txn_id = payload.transaction_id or f"TXN-{int(time.time()*1000)}"

    return ScoringDecision(
        transaction_id=txn_id,
        decision=decision,
        risk_score=risk_score,
        fraud_probability=round(ensemble_prob, 6),
        model_probabilities={
            "lightgbm": round(p_lgb, 6),
            "xgboost": round(p_xgb, 6),
            "random_forest": round(p_rf, 6)
        },
        latency_ms=round(latency_ms, 4),
        platform="IBM LinuxONE (s390x)",
        recommendation=recommendation
    )

@app.get("/metrics")
def get_metrics():
    n = max(1, stats["total_transactions"])
    return {
        "total_scored": stats["total_transactions"],
        "approved": stats["approved_transactions"],
        "blocked": stats["blocked_transactions"],
        "fraud_interception_rate_pct": round((stats["blocked_transactions"] / n) * 100.0, 2),
        "mean_latency_ms": round(stats["total_latency_ms"] / n, 4),
        "min_latency_ms": round(stats["min_latency_ms"], 4) if stats["min_latency_ms"] != float("inf") else 0.0,
        "max_latency_ms": round(stats["max_latency_ms"], 4),
        "hardware_platform": "IBM LinuxONE Community Cloud (s390x)",
        "sla_target_ms": 5.0,
        "sla_met": (stats["total_latency_ms"] / n) < 5.0 if stats["total_transactions"] > 0 else True
    }
