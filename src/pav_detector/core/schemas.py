from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class DetectionResult:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    should_alert: bool
    flow: Dict[str, Any]
    sensor_name: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""

    @property
    def event_type(self) -> Optional[str]:
        if self.should_alert:
            return self.predicted_class
        return None
