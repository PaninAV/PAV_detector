from __future__ import annotations

import pandas as pd
import streamlit as st

from pav_detector.config import Settings
from pav_detector.db.postgres import PostgresStorage


def load_events(storage: PostgresStorage, event_type: str | None, sensor_name: str | None, limit: int):
    rows = list(storage.list_events(event_type=event_type, sensor_name=sensor_name, limit=limit))
    return pd.DataFrame(rows)


def render_results_view(
    settings: Settings,
    *,
    use_sidebar_filters: bool = True,
    key_prefix: str = "results",
) -> None:
    if not settings.enable_db:
        st.error("База данных отключена. Установите ENABLE_DB=true в .env")
        return

    storage = PostgresStorage(settings.postgres_dsn)

    filter_container = st.sidebar if use_sidebar_filters else st.container()
    with filter_container:
        st.header("Фильтры")
        event_type = st.selectbox(
            "Тип события",
            ["Все", "VPN", "PROXY"],
            key=f"{key_prefix}_event_type",
        )
        sensor_name = st.text_input(
            "Выбор датасета",
            value="",
            key=f"{key_prefix}_sensor_name",
        )
        limit = st.slider(
            "Лимит строк",
            min_value=10,
            max_value=2000,
            value=500,
            step=10,
            key=f"{key_prefix}_limit",
        )
        st.button("Обновить", key=f"{key_prefix}_refresh")

    event_type_filter = None if event_type == "Все" else event_type
    sensor_filter = sensor_name.strip() or None
    df = load_events(storage, event_type_filter, sensor_filter, limit)

    st.subheader("Сводка событий")
    if df.empty:
        st.info("События не найдены.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего событий", int(len(df)))
    col2.metric("Средняя уверенность", float(df["confidence"].mean()))
    col3.metric("Максимальная уверенность", float(df["confidence"].max()))

    st.subheader("Распределение по типу события")
    dist = df["event_type"].value_counts().rename_axis("event_type").reset_index(name="count")
    st.bar_chart(dist.set_index("event_type"))

    st.subheader("Таблица событий")
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


def main() -> None:
    settings = Settings.from_env()
    st.set_page_config(page_title="PAV Detector UI", layout="wide")
    st.title("VPN / Proxy Detector - Operator UI")
    render_results_view(settings, use_sidebar_filters=True, key_prefix="operator")


if __name__ == "__main__":
    main()
