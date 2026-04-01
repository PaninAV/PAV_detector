from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv


def _parse_csv_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    app_name: str = "PAV Detector"
    app_env: str = "dev"
    threshold: float = 0.8
    classes: List[str] = field(default_factory=lambda: ["LEGIT", "VPN", "PROXY"])
    feature_order: List[str] = field(default_factory=list)

    model_dir: Path = Path("models")
    model_onnx_path: Path = Path("models/model.onnx")
    model_torch_path: Path = Path("models/model.pt")
    scaler_path: Path = Path("models/scaler.pkl")

    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/pav_detector"
    enable_db: bool = True

    log_level: str = "INFO"
    sensor_name: str = "default-sensor"
    cicflowmeter_cmd: str = "cicflowmeter"

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Settings":
        load_dotenv(env_file)
        return cls(
            app_name=os.getenv("APP_NAME", "PAV Detector"),
            app_env=os.getenv("APP_ENV", "dev"),
            threshold=float(os.getenv("THRESHOLD", "0.8")),
            classes=_parse_csv_list(os.getenv("CLASSES", "LEGIT,VPN,PROXY")),
            feature_order=_parse_csv_list(os.getenv("FEATURE_ORDER", "")),
            model_dir=Path(os.getenv("MODEL_DIR", "models")),
            model_onnx_path=Path(os.getenv("MODEL_ONNX_PATH", "models/model.onnx")),
            model_torch_path=Path(os.getenv("MODEL_TORCH_PATH", "models/model.pt")),
            scaler_path=Path(os.getenv("SCALER_PATH", "models/scaler.pkl")),
            postgres_dsn=os.getenv(
                "POSTGRES_DSN",
                "postgresql://postgres:postgres@localhost:5432/pav_detector",
            ),
            enable_db=os.getenv("ENABLE_DB", "true").lower() in {"1", "true", "yes"},
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            sensor_name=os.getenv("SENSOR_NAME", "default-sensor"),
            cicflowmeter_cmd=os.getenv("CICFLOWMETER_CMD", "cicflowmeter"),
        )
