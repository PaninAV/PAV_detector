from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detection_events (
    id BIGSERIAL PRIMARY KEY,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sensor_name TEXT NOT NULL,
    event_type TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    src_ip TEXT,
    src_port INTEGER,
    dst_ip TEXT,
    dst_port INTEGER,
    protocol TEXT,
    raw_flow JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detection_events_detected_at
    ON detection_events (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detection_events_event_type
    ON detection_events (event_type);
CREATE INDEX IF NOT EXISTS idx_detection_events_sensor_name
    ON detection_events (sensor_name);
"""


class PostgresStorage:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _connect(self):
        return psycopg2.connect(self.dsn)

    def init_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.commit()
        logger.info("PostgreSQL schema initialized")

    def save_event(
        self,
        *,
        sensor_name: str,
        event_type: str,
        confidence: float,
        flow: Dict[str, Any],
    ) -> None:
        src_ip = flow.get("Src IP") or flow.get("src_ip")
        src_port = _as_int(flow.get("Src Port") or flow.get("src_port"))
        dst_ip = flow.get("Dst IP") or flow.get("dst_ip")
        dst_port = _as_int(flow.get("Dst Port") or flow.get("dst_port"))
        protocol = str(flow.get("Protocol") or flow.get("protocol") or "")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO detection_events (
                        sensor_name, event_type, confidence,
                        src_ip, src_port, dst_ip, dst_port, protocol, raw_flow
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        sensor_name,
                        event_type,
                        confidence,
                        src_ip,
                        src_port,
                        dst_ip,
                        dst_port,
                        protocol,
                        json.dumps(flow, ensure_ascii=True, default=str),
                    ),
                )
            conn.commit()

    def list_events(
        self,
        *,
        event_type: Optional[str] = None,
        sensor_name: Optional[str] = None,
        limit: int = 500,
    ) -> Iterable[Dict[str, Any]]:
        where_parts = []
        params = []

        if event_type:
            where_parts.append("event_type = %s")
            params.append(event_type)
        if sensor_name:
            where_parts.append("sensor_name = %s")
            params.append(sensor_name)

        where_sql = " AND ".join(where_parts)
        if where_sql:
            where_sql = f"WHERE {where_sql}"

        query = f"""
            SELECT id, detected_at, sensor_name, event_type, confidence,
                   src_ip, src_port, dst_ip, dst_port, protocol, raw_flow
            FROM detection_events
            {where_sql}
            ORDER BY detected_at DESC
            LIMIT %s
        """
        params.append(limit)

        with self._connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return rows


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
