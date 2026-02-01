from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import ipaddress

from layers.network_ids import NetworkIDS
from layers.prompt_detector import PromptDetector


app = FastAPI(title="SentinelAI Backend")

# -------------------------------------------------
# Load ML models
# -------------------------------------------------
network_ids = NetworkIDS("../models/network_model.pth")
prompt_detector = PromptDetector("../models/prompt_model.pth")

# -------------------------------------------------
# Schemas
# -------------------------------------------------
class AnalyzeRequest(BaseModel):
    request_type: str
    data: dict = {}


class AnalyzeResponse(BaseModel):
    decision: str
    layer: Optional[str]
    confidence: float


# -------------------------------------------------
# AUTO Access Detection
# -------------------------------------------------
def detect_access(request: Request):
    client_ip = request.client.host

    try:
        ip = ipaddress.ip_address(client_ip)

        # Localhost / private networks → trusted
        if ip.is_loopback or ip.is_private:
            return "India", False

    except ValueError:
        pass

    # Public IPs → simulate foreign / VPN
    return "Other", True


# -------------------------------------------------
# Normalize model output (IMPORTANT)
# -------------------------------------------------
def normalize_score(score: float) -> float:
    """
    Ensures confidence is always between 0.0 and 0.99
    (never exactly 1.0 for demo realism)
    """
    try:
        score = float(score)
    except Exception:
        return 0.5

    if score >= 1.0:
        return 0.95
    if score <= 0.0:
        return 0.05
    return round(score, 2)


# -------------------------------------------------
# Analyze API
# -------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, request: Request):

    # =================================================
    # STEP 1: ACCESS CONTROL (Layer 1)
    # =================================================
    country, vpn = detect_access(request)

    if country != "India":
        return AnalyzeResponse(
            decision="BLOCK",
            layer="Layer 1 (Geo Restriction)",
            confidence=1.0
        )

    if vpn is True:
        return AnalyzeResponse(
            decision="BLOCK",
            layer="Layer 1 (VPN Detection)",
            confidence=1.0
        )

    # =================================================
    # STEP 2: NETWORK IDS (Layer 2)
    # =================================================
    if req.request_type == "network":
        raw_score = network_ids.predict()
        prob = normalize_score(raw_score)

        if prob >= 0.75:
            return AnalyzeResponse(
                decision="WARNING",
                layer="Layer 2 (Network IDS)",
                confidence=prob
            )
        else:
            return AnalyzeResponse(
                decision="ALLOW",
                layer="Layer 2 (Network IDS)",
                confidence=prob
            )

    # =================================================
    # STEP 3: PROMPT INJECTION (Layer 3)
    # =================================================
    if req.request_type == "prompt":
        text = req.data.get("text", "")
        raw_score = prompt_detector.predict(text)
        prob = normalize_score(raw_score)

        if prob >= 0.75:
            return AnalyzeResponse(
                decision="BLOCK",
                layer="Layer 3 (Prompt Injection)",
                confidence=prob
            )
        elif prob >= 0.45:
            return AnalyzeResponse(
                decision="WARNING",
                layer="Layer 3 (Prompt Injection)",
                confidence=prob
            )
        else:
            return AnalyzeResponse(
                decision="ALLOW",
                layer="Layer 3 (Prompt Injection)",
                confidence=prob
            )

    return AnalyzeResponse(
        decision="BLOCK",
        layer="Unknown",
        confidence=1.0
    )
