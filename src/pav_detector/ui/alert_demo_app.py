from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from pav_detector.config import Settings
from pav_detector.db.postgres import PostgresStorage


def _format_relative_time(ts) -> str:
    if ts is None:
        return "-"
    now = datetime.now(timezone.utc)
    if isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return str(ts)
    else:
        parsed = ts
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = now - parsed
    if delta < timedelta(minutes=1):
        return "только что"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)} мин назад"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() // 3600)} ч назад"
    return f"{delta.days} дн назад"


def _severity(confidence: float) -> tuple[str, str]:
    if confidence >= 0.9:
        return "Критический", "🔴"
    if confidence >= 0.75:
        return "Высокий", "🟠"
    if confidence >= 0.6:
        return "Средний", "🟡"
    return "Низкий", "🟢"


def _load_latest_alerts(storage: PostgresStorage, limit: int) -> pd.DataFrame:
    rows = list(storage.list_events(limit=limit))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].astype(float)
    return df


def main() -> None:
    st.set_page_config(page_title="Демо-страница алертов", layout="wide")
    st.title("Демо: страница алертов VPN/PROXY")
    st.caption("Черновой UI для проектирования логики реакции на инциденты.")

    settings = Settings.from_env()
    if not settings.enable_db:
        st.error("База данных отключена. Установите ENABLE_DB=true в .env")
        return

    storage = PostgresStorage(settings.postgres_dsn)

    with st.sidebar:
        st.header("Параметры")
        row_limit = st.slider("Сколько алертов загрузить", 10, 1000, 200, 10)
        auto_refresh = st.checkbox("Автообновление (каждые 5 сек)", value=False)
        st.button("Обновить")

    alerts_df = _load_latest_alerts(storage, row_limit)
    if alerts_df.empty:
        st.info("В базе пока нет алертов.")
        return

    total = len(alerts_df)
    vpn_count = int((alerts_df["event_type"] == "VPN").sum())
    proxy_count = int((alerts_df["event_type"] == "PROXY").sum())
    avg_conf = float(alerts_df["confidence"].mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Всего алертов", total)
    k2.metric("VPN", vpn_count)
    k3.metric("PROXY", proxy_count)
    k4.metric("Средняя уверенность", f"{avg_conf:.3f}")

    st.markdown("---")
    st.subheader("Лента алертов")

    for _, row in alerts_df.head(50).iterrows():
        conf = float(row.get("confidence", 0.0))
        sev, icon = _severity(conf)
        detected_at = row.get("detected_at")
        rel_time = _format_relative_time(detected_at)

        header_col, action_col = st.columns([5, 2])
        with header_col:
            st.markdown(
                f"### {icon} {row.get('event_type', '-')}"
                f" | уверенность: `{conf:.3f}` | уровень: **{sev}**"
            )
            st.caption(
                f"ID: {row.get('id')} · Сенсор: {row.get('sensor_name', '-')}"
                f" · Время: {detected_at} ({rel_time})"
            )
            st.write(
                f"Источник: `{row.get('src_ip', '-')}`:{row.get('src_port', '-')}"
                f" → Назначение: `{row.get('dst_ip', '-')}`:{row.get('dst_port', '-')}"
                f" · Протокол: `{row.get('protocol', '-')}`"
            )
        with action_col:
            st.button("Открыть инцидент", key=f"open_{row.get('id')}")
            st.button("Пометить как ложный", key=f"fp_{row.get('id')}")
            st.button("Игнорировать сенсор", key=f"mute_{row.get('id')}")

        with st.expander(f"RAW flow (id={row.get('id')})"):
            st.json(row.get("raw_flow", {}))
        st.markdown("---")

    st.subheader("Распределение алертов")
    dist = alerts_df["event_type"].value_counts().rename_axis("event_type").reset_index(name="count")
    st.bar_chart(dist.set_index("event_type"))

    st.caption(
        "Кнопки действий в демо не пишут изменения в БД — это UI-черновик для отработки логики."
    )

    if auto_refresh:
        import time

        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
