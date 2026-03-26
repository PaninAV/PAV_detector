from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pav_detector.config import Settings
from pav_detector.core.service import DetectionService
from pav_detector.utils.logging_json import configure_logging

logger = logging.getLogger(__name__)
settings = Settings.from_env()
configure_logging(settings.log_level)
service: DetectionService | None = None
startup_error: str | None = None

app = FastAPI(title=settings.app_name)


class FlowEnvelope(BaseModel):
    flow: Dict[str, Any]
    sensor_name: Optional[str] = None


class BatchFlowEnvelope(BaseModel):
    flows: List[Dict[str, Any]] = Field(default_factory=list)
    sensor_name: Optional[str] = None


@app.on_event("startup")
def startup_initialize() -> None:
    global service, startup_error
    try:
        service = DetectionService(settings)
        startup_error = None
    except Exception as exc:  # pragma: no cover
        service = None
        startup_error = str(exc)
        logger.exception("Failed to initialize detection service")


def _get_service() -> DetectionService:
    if service is None:
        detail = "Detection service unavailable"
        if startup_error:
            detail = f"{detail}: {startup_error}"
        raise HTTPException(status_code=503, detail=detail)
    return service


@app.get("/health")
def health() -> Dict[str, Any]:
    if service is None:
        return {
            "status": "degraded",
            "model_backend": None,
            "error": startup_error or "service_not_initialized",
        }
    return {"status": "ok", "model_backend": service.engine.backend}


@app.post("/v1/classify")
def classify(payload: FlowEnvelope) -> Dict[str, Any]:
    detector = _get_service()
    result = detector.classify_flow(payload.flow, sensor_name=payload.sensor_name)
    return {
        "predicted_class": result.predicted_class,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "should_alert": result.should_alert,
        "reason": result.reason,
        "sensor_name": result.sensor_name,
        "detected_at": result.detected_at.isoformat(),
    }


@app.post("/v1/classify/batch")
def classify_batch(payload: BatchFlowEnvelope) -> Dict[str, Any]:
    detector = _get_service()
    items = []
    for flow in payload.flows:
        result = detector.classify_flow(flow, sensor_name=payload.sensor_name)
        items.append(
            {
                "predicted_class": result.predicted_class,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "should_alert": result.should_alert,
                "reason": result.reason,
                "sensor_name": result.sensor_name,
                "detected_at": result.detected_at.isoformat(),
            }
        )
    return {"count": len(items), "results": items}
