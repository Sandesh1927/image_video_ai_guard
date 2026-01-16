from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Disable docs for customers (VERY IMPORTANT)
app = FastAPI(
    title="AI Guard Backend",
    docs_url=None,
    redoc_url=None
)

# -----------------------------
# Data Model
# -----------------------------
class DetectionResult(BaseModel):
    source: str               # image / video / camera
    threats: List[str]        # detected objects
    risk_score: int           # 1–10
    timestamp: str            # ISO time
    device_id: str            # mobile / desktop id


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "AI Guard backend running",
        "version": "1.0.0"
    }


# -----------------------------
# Receive Detection Result
# -----------------------------
@app.post("/submit-result")
def submit_result(data: DetectionResult):
    # Later you can:
    # - Save to database
    # - Trigger alert
    # - Send notification

    return {
        "message": "Detection received successfully",
        "received_data": data
    }
