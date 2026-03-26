from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from pav_detector.config import Settings
from pav_detector.core.service import DetectionService
from pav_detector.utils.logging_json import configure_logging

settings = Settings.from_env()
configure_logging(settings.log_level)
service = DetectionService(settings)

app = FastAPI(title=settings.app_name)


class FlowEnvelope(BaseModel):
    flow: Dict[str, Any]
    sensor_name: Optional[str] = None


class BatchFlowEnvelope(BaseModel):
    flows: List[Dict[str, Any]] = Field(default_factory=list)
    sensor_name: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_backend": service.engine.backend}


@app.post("/v1/classify")
def classify(payload: FlowEnvelope) -> Dict[str, Any]:
    result = service.classify_flow(payload.flow, sensor_name=payload.sensor_name)
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
    items = []
    for flow in payload.flows:
        result = service.classify_flow(flow, sensor_name=payload.sensor_name)
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
