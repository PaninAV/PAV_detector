from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pav_detector.config import Settings
from pav_detector.core.service import DetectionService
from pav_detector.offline.cicflow import run_cicflowmeter_for_pcap
from pav_detector.utils.logging_json import configure_logging

logger = logging.getLogger(__name__)


def _row_to_flow(row: pd.Series) -> Dict[str, Any]:
    flow: Dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if hasattr(value, "item"):
            value = value.item()
        flow[str(key)] = value
    return flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline VPN/Proxy detection for CICFlowMeter CSV or PCAP input."
    )
    parser.add_argument("--csv", type=Path, help="Path to existing CICFlowMeter CSV file")
    parser.add_argument("--pcap", type=Path, help="Path to PCAP/PCAPNG file")
    parser.add_argument(
        "--generated-csv",
        type=Path,
        default=Path("data/generated_flows.csv"),
        help="Where to save CSV generated from --pcap",
    )
    parser.add_argument(
        "--sensor-name",
        type=str,
        default="offline-sensor",
        help="Sensor name for persisted alert events",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/offline_results.json"),
        help="Where to write classification results",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.csv and not args.pcap:
        parser.error("Provide either --csv or --pcap")

    settings = Settings.from_env()
    configure_logging(settings.log_level)

    input_csv: Path
    if args.csv:
        input_csv = args.csv
    else:
        assert args.pcap is not None
        run_cicflowmeter_for_pcap(settings.cicflowmeter_cmd, args.pcap, args.generated_csv)
        input_csv = args.generated_csv

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    service = DetectionService(settings)

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        flow = _row_to_flow(row)
        detection = service.classify_flow(
            flow=flow,
            sensor_name=args.sensor_name,
            source_mode="offline",
        )
        results.append(
            {
                "predicted_class": detection.predicted_class,
                "confidence": detection.confidence,
                "probabilities": detection.probabilities,
                "should_alert": detection.should_alert,
                "reason": detection.reason,
                "sensor_name": detection.sensor_name,
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")
    logger.info("Processed flows: %s", len(results))
    logger.info("Results written to: %s", args.output_json)


if __name__ == "__main__":
    main()
