from __future__ import annotations

from argparse import Namespace
import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from pav_detector.train.train_model import train
from pav_detector.train.train_model import _save_artifacts as save_train_artifacts
from pav_detector.utils.logging_json import configure_logging
from pav_detector.config import Settings
from pav_detector.offline.run_offline import _row_to_flow
from pav_detector.core.service import DetectionService
from pav_detector.ui.streamlit_app import render_results_view


def _uploaded_csv_to_dataframe(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("CSV file is required.")
    return pd.read_csv(uploaded_file)


def _run_train_section() -> None:
    st.subheader("1) Обучение модели")
    st.caption("Загрузите обучающий CSV и нажмите кнопку запуска обучения.")

    uploaded_train_csv = st.file_uploader(
        "Файл датасета для обучения (CSV)",
        type=["csv"],
        key="train_csv_uploader",
    )

    col1, col2, col3 = st.columns(3)
    epochs = col1.number_input("Количество эпох", min_value=1, max_value=10000, value=40, step=1)
    batch_size = col2.number_input("Размер батча", min_value=1, max_value=100000, value=512, step=1)
    lr = col3.number_input(
        "Скорость обучения",
        min_value=0.000001,
        max_value=1.0,
        value=0.001,
        step=0.0005,
        format="%.6f",
    )

    classes_raw = st.text_input("Классы (через запятую)", value="LEGIT,VPN,PROXY")
    export_onnx = st.checkbox("Экспортировать ONNX", value=True)

    if st.button("Запустить обучение", type="primary"):
        if uploaded_train_csv is None:
            st.error("Сначала выберите CSV для обучения.")
            return
        try:
            df = _uploaded_csv_to_dataframe(uploaded_train_csv)
            tmp_dir = Path(tempfile.mkdtemp(prefix="pav_train_"))
            tmp_csv = tmp_dir / "train.csv"
            df.to_csv(tmp_csv, index=False)

            args = Namespace(
                train_csv=[tmp_csv],
                label_column="",
                subtype_column="",
                feature_columns="",
                classes=classes_raw.strip(),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                hidden_dim=128,
                dropout=0.2,
                val_split=0.2,
                seed=42,
                export_onnx=bool(export_onnx),
                out_dir=Path("models"),
                save_feature_order_json=Path("models/feature_order.json"),
                save_metrics_json=Path("models/train_metrics.json"),
                log_level="INFO",
            )

            artifacts = train(args)
            save_train_artifacts(args, artifacts)
            st.success("Обучение завершено. Артефакты сохранены в папке models/.")
            st.caption("Метрики обучения:")
            st.json(artifacts.metrics)
        except Exception as exc:
            st.exception(exc)


def _run_offline_section() -> None:
    st.subheader("2) Оффлайн проверка CSV")
    st.caption("Запускает классификацию по загруженному CSV и сохраняет подозрительные события в БД.")

    uploaded_offline_csv = st.file_uploader(
        "Файл для оффлайн проверки (CSV)",
        type=["csv"],
        key="offline_csv_uploader",
    )
    sensor_name = st.text_input(
        "Имя датасета/сенсора",
        value="workbench-offline",
        key="offline_sensor_name",
    )

    if st.button("Запустить оффлайн проверку"):
        if uploaded_offline_csv is None:
            st.error("Сначала выберите CSV для оффлайн проверки.")
            return
        try:
            settings = Settings.from_env()
            configure_logging(settings.log_level)
            service = DetectionService(settings)

            df = _uploaded_csv_to_dataframe(uploaded_offline_csv)
            results = []
            alerts = 0
            for _, row in df.iterrows():
                flow = _row_to_flow(row)
                detection = service.classify_flow(
                    flow=flow,
                    sensor_name=sensor_name,
                    source_mode="offline",
                )
                if detection.should_alert:
                    alerts += 1
                results.append(
                    {
                        "predicted_class": detection.predicted_class,
                        "confidence": detection.confidence,
                        "should_alert": detection.should_alert,
                        "reason": detection.reason,
                    }
                )

            out_path = Path("data/offline_results_workbench.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")

            st.success(
                f"Оффлайн проверка завершена. Обработано: {len(results)}, алертов: {alerts}. "
                f"Результат: {out_path}"
            )
            st.dataframe(pd.DataFrame(results).head(200), use_container_width=True)
        except Exception as exc:
            st.exception(exc)
def _run_online_section() -> None:
    st.subheader("3) Онлайн проверка")
    st.caption(
        "Проверка подключения к онлайн API. "
        "Сами flow-данные должны поступать во внешний API-эндпоинт из отдельного источника."
    )

    api_url = st.text_input("URL API", value="http://127.0.0.1:8000/v1/classify")
    sensor_name = st.text_input(
        "Имя датасета/сенсора",
        value="workbench-online",
        key="online_sensor_name",
    )

    if st.button("Проверить подключение к API"):
        try:
            health_url = api_url.rsplit("/", 1)[0] + "/health"
            request = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read().decode("utf-8")
            st.success("Подключение к API успешно.")
            st.code(body)
            st.info(
                f"Источник данных для online должен отправлять flow в {api_url} "
                f"с sensor_name={sensor_name}."
            )
        except urllib.error.URLError as exc:
            st.error(f"Не удалось подключиться к API: {exc}")
        except Exception as exc:
            st.exception(exc)


def main() -> None:
    st.set_page_config(page_title="Детектор скрытых туннелей и прокси соединений", layout="wide")
    st.title("Детектор скрытых туннелей и прокси соединений")

    tab_train, tab_offline, tab_online, tab_results = st.tabs(
        ["Обучение модели", "Оффлайн проверка", "Онлайн проверка", "Просмотр результатов"]
    )

    with tab_train:
        _run_train_section()
    with tab_offline:
        _run_offline_section()
    with tab_online:
        _run_online_section()
    with tab_results:
        st.subheader("4) Просмотр результатов")
        settings = Settings.from_env()
        render_results_view(settings, use_sidebar_filters=False, key_prefix="workbench_results")

    st.markdown("---")
    st.subheader("Панель алертов (онлайн)")
    settings = Settings.from_env()
    if settings.enable_db:
        from pav_detector.db.postgres import PostgresStorage

        try:
            storage = PostgresStorage(settings.postgres_dsn)
            online_alerts = list(storage.list_alerts(source_mode="online", limit=50))
            if not online_alerts:
                st.info("Онлайн-алерты пока не найдены.")
            else:
                st.caption("Последние события, зафиксированные во время онлайн проверки.")
                alerts_df = pd.DataFrame(online_alerts)
                shown_cols = [
                    "alert_id",
                    "alert_created_at",
                    "alert_type",
                    "status",
                    "sensor_name",
                    "src_ip",
                    "dst_ip",
                    "confidence",
                ]
                available_cols = [col for col in shown_cols if col in alerts_df.columns]
                st.dataframe(alerts_df[available_cols], use_container_width=True)
        except Exception as exc:
            st.warning(
                "Не удалось загрузить панель онлайн-алертов. "
                "Выполните инициализацию схемы БД (pav-init-db) и повторите."
            )
            st.caption(str(exc))
    else:
        st.info("База данных отключена, панель алертов недоступна.")


if __name__ == "__main__":
    main()
