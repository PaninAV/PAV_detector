from __future__ import annotations

import pandas as pd
import streamlit as st

from pav_detector.config import Settings
from pav_detector.db.postgres import PostgresStorage


def load_events(storage: PostgresStorage, event_type: str | None, sensor_name: str | None, limit: int):
    rows = list(storage.list_events(event_type=event_type, sensor_name=sensor_name, limit=limit))
    return pd.DataFrame(rows)


def main() -> None:
    settings = Settings.from_env()
    st.set_page_config(page_title="PAV Detector UI", layout="wide")
    st.title("VPN / Proxy Detector - Operator UI")

    if not settings.enable_db:
        st.error("Database is disabled. Set ENABLE_DB=true in .env")
        return

    storage = PostgresStorage(settings.postgres_dsn)

    with st.sidebar:
        st.header("Filters")
        event_type = st.selectbox("Event type", ["ALL", "VPN", "PROXY"])
        sensor_name = st.text_input("Sensor name (optional)", value="")
        limit = st.slider("Row limit", min_value=10, max_value=2000, value=500, step=10)
        st.button("Refresh")

    event_type_filter = None if event_type == "ALL" else event_type
    sensor_filter = sensor_name.strip() or None
    df = load_events(storage, event_type_filter, sensor_filter, limit)

    st.subheader("Event Summary")
    if df.empty:
        st.info("No events found.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total events", int(len(df)))
    col2.metric("Avg confidence", float(df["confidence"].mean()))
    col3.metric("Max confidence", float(df["confidence"].max()))

    st.subheader("Distribution by event type")
    dist = df["event_type"].value_counts().rename_axis("event_type").reset_index(name="count")
    st.bar_chart(dist.set_index("event_type"))

    st.subheader("Events table")
    shown_cols = [
        "id",
        "detected_at",
        "sensor_name",
        "event_type",
        "confidence",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "protocol",
    ]
    st.dataframe(df[shown_cols], use_container_width=True)


if __name__ == "__main__":
    main()
