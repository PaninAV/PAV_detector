from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from pav_detector.config import Settings
from pav_detector.core.decision import make_decision
from pav_detector.core.inference import InferenceEngine
from pav_detector.core.preprocessing import prepare_feature_frame
from pav_detector.core.schemas import DetectionResult
from pav_detector.db.postgres import PostgresStorage

logger = logging.getLogger(__name__)


class DetectionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = InferenceEngine(
            model_onnx_path=settings.model_onnx_path,
            model_torch_path=settings.model_torch_path,
            scaler_path=settings.scaler_path,
            classes=settings.classes,
        )
        self.storage: Optional[PostgresStorage] = None
        if settings.enable_db:
            self.storage = PostgresStorage(settings.postgres_dsn)
            self.storage.init_schema()

    def classify_flow(self, flow: Dict[str, Any], sensor_name: Optional[str] = None) -> DetectionResult:
        df = pd.DataFrame([flow])
        features_df = prepare_feature_frame(df, self.settings.feature_order)
        features = features_df.to_numpy(dtype=np.float32)

        output = self.engine.predict(features)
        predicted_class = self.settings.classes[output.predicted_index]
        probabilities = self.engine.probabilities_as_dict(output.probabilities)
        confidence = float(probabilities.get(predicted_class, 0.0))
        should_alert, reason = make_decision(predicted_class, probabilities, self.settings.threshold)

        result = DetectionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            should_alert=should_alert,
            flow=flow,
            sensor_name=sensor_name or self.settings.sensor_name,
            reason=reason,
        )
        if should_alert and self.storage is not None:
            self.storage.save_event(
                sensor_name=result.sensor_name,
                event_type=predicted_class,
                confidence=confidence,
                flow=flow,
            )
        return result
