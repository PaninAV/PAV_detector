from __future__ import annotations

from typing import Dict


def make_decision(
    predicted_class: str,
    probabilities: Dict[str, float],
    threshold: float,
) -> tuple[bool, str]:
    confidence = float(probabilities.get(predicted_class, 0.0))
    if predicted_class == "LEGIT":
        return False, "predicted_legit"
    if confidence < threshold:
        return False, f"below_threshold:{confidence:.4f}<{threshold:.4f}"
    return True, "vpn_or_proxy_above_threshold"
