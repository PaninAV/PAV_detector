from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_cicflowmeter_for_pcap(cicflowmeter_cmd: str, pcap_path: Path, output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        cicflowmeter_cmd,
        "-f",
        str(pcap_path),
        "-c",
        str(output_csv_path),
    ]
    logger.info("Running CICFlowMeter command: %s", " ".join(command))
    subprocess.run(command, check=True)
