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
    source_mode TEXT NOT NULL DEFAULT 'unknown',
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

CREATE TABLE IF NOT EXISTS detection_alerts (
    id BIGSERIAL PRIMARY KEY,
    log_id BIGINT NOT NULL REFERENCES detection_events(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alert_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_detection_alerts_log_id_unique
    ON detection_alerts (log_id);
CREATE INDEX IF NOT EXISTS idx_detection_alerts_created_at
    ON detection_alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detection_alerts_alert_type
    ON detection_alerts (alert_type);

ALTER TABLE detection_events
    ADD COLUMN IF NOT EXISTS source_mode TEXT NOT NULL DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS idx_detection_events_source_mode
    ON detection_events (source_mode);
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
        source_mode: str,
        event_type: str,
        confidence: float,
        flow: Dict[str, Any],
        create_alert: bool = True,
    ) -> int:
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
                        sensor_name, source_mode, event_type, confidence,
                        src_ip, src_port, dst_ip, dst_port, protocol, raw_flow
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        sensor_name,
                        source_mode,
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
                event_id = int(cur.fetchone()[0])
                if create_alert:
                    cur.execute(
                        """
                        INSERT INTO detection_alerts (log_id, alert_type)
                        VALUES (%s, %s)
                        """,
                        (event_id, event_type),
                    )
            conn.commit()
        return event_id

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
            SELECT id, detected_at, sensor_name, source_mode, event_type, confidence,
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

    def list_sensor_names(
        self,
        *,
        source_mode: Optional[str] = None,
        alerts_only: bool = False,
        limit: int = 1000,
    ) -> Iterable[str]:
        if alerts_only:
            query = """
                SELECT e.sensor_name
                FROM detection_alerts a
                JOIN detection_events e ON e.id = a.log_id
                {where_sql}
                GROUP BY e.sensor_name
                ORDER BY e.sensor_name ASC
                LIMIT %s
            """
        else:
            query = """
                SELECT sensor_name
                FROM detection_events
                {where_sql}
                GROUP BY sensor_name
                ORDER BY sensor_name ASC
                LIMIT %s
            """
        params = []
        where_sql = ""
        if source_mode:
            if alerts_only:
                where_sql = "WHERE e.source_mode = %s"
            else:
                where_sql = "WHERE source_mode = %s"
            params.append(source_mode)

        query = query.format(where_sql=where_sql)
        params.append(limit)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [str(row[0]) for row in rows if row and row[0]]

    def list_alerts(
        self,
        *,
        event_type: Optional[str] = None,
        sensor_name: Optional[str] = None,
        source_mode: Optional[str] = None,
        limit: int = 500,
    ) -> Iterable[Dict[str, Any]]:
        where_parts = []
        params = []

        if event_type:
            where_parts.append("a.alert_type = %s")
            params.append(event_type)
        if sensor_name:
            where_parts.append("e.sensor_name = %s")
            params.append(sensor_name)
        if source_mode:
            where_parts.append("e.source_mode = %s")
            params.append(source_mode)

        where_sql = " AND ".join(where_parts)
        if where_sql:
            where_sql = f"WHERE {where_sql}"

        query = f"""
            SELECT
                a.id AS alert_id,
                a.created_at AS alert_created_at,
                a.alert_type,
                a.status,
                e.id,
                e.detected_at,
                e.sensor_name,
                e.source_mode,
                e.event_type,
                e.confidence,
                e.src_ip,
                e.src_port,
                e.dst_ip,
                e.dst_port,
                e.protocol,
                e.raw_flow
            FROM detection_alerts a
            JOIN detection_events e ON e.id = a.log_id
            {where_sql}
            ORDER BY a.created_at DESC
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
