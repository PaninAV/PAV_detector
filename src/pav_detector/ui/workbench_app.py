from __future__ import annotations

from argparse import Namespace
import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from pav_detector.train.train_model import train
from pav_detector.train.train_model import _save_artifacts as save_train_artifacts
from pav_detector.utils.logging_json import configure_logging
from pav_detector.config import Settings
from pav_detector.offline.run_offline import _row_to_flow
from pav_detector.core.service import DetectionService
from pav_detector.ui.streamlit_app import render_results_view


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


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


def _post_flow(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_flow_lines(raw_lines: str) -> list[Dict[str, Any]]:
    flows: list[Dict[str, Any]] = []
    for idx, line in enumerate(raw_lines.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Строка {idx}: некорректный JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Строка {idx}: ожидается JSON-объект flow.")
        flows.append(item)
    return flows


def _run_online_section() -> None:
    st.subheader("3) Онлайн проверка (эмуляция потока в API)")
    st.caption(
        "Отправляет строки flow в /v1/classify. "
        "Вставьте по одной JSON-строке на каждый flow. API должен быть запущен заранее."
    )

    api_url = st.text_input("URL API", value="http://127.0.0.1:8000/v1/classify")
    sensor_name = st.text_input(
        "Имя датасета/сенсора",
        value="workbench-online",
        key="online_sensor_name",
    )
    raw_flow_lines = st.text_area(
        "Строки (JSON, по одной на строку)",
        value=(
            '{"Src IP":"10.0.0.1","Dst IP":"8.8.8.8","Protocol":"6",'
            '"Flow Duration":1200,"Tot Fwd Pkts":5,"Tot Bwd Pkts":3}\n'
            '{"Src IP":"10.0.0.2","Dst IP":"1.1.1.1","Protocol":"17",'
            '"Flow Duration":3500,"Tot Fwd Pkts":12,"Tot Bwd Pkts":11}'
        ),
        height=180,
    )

    if st.button("Запустить онлайн отправку в API"):
        if not raw_flow_lines.strip():
            st.error("Вставьте хотя бы одну JSON-строку flow.")
            return
        try:
            flows = _parse_flow_lines(raw_flow_lines)
            if not flows:
                st.error("Не удалось извлечь flow-строки. Проверьте формат JSON.")
                return

            sent = 0
            alerts = 0
            preview = []
            progress = st.progress(0)
            total = max(len(flows), 1)

            alert_placeholder = st.empty()
            for idx, flow in enumerate(flows, start=1):
                payload = {"sensor_name": sensor_name, "source_mode": "online", "flow": flow}
                result = _post_flow(api_url, payload)

                sent += 1
                if result.get("should_alert"):
                    alerts += 1
                    src_ip = flow.get("Src IP") or flow.get("src_ip") or "неизвестный IP"
                    message = f"С {src_ip} обнаружена угроза ({result.get('predicted_class')})."
                    alert_placeholder.warning(f"⚠ {message}")
                    if hasattr(st, "toast"):
                        st.toast(message, icon="🚨")
                if len(preview) < 200:
                    preview.append(
                        {
                            "predicted_class": result.get("predicted_class"),
                            "confidence": result.get("confidence"),
                            "should_alert": result.get("should_alert"),
                            "reason": result.get("reason"),
                        }
                    )
                progress.progress(min(idx / total, 1.0))

            alert_placeholder.empty()
            st.success(f"Отправлено flow: {sent}, алертов: {alerts}")
            st.caption("Предпросмотр ответов API:")
            st.dataframe(pd.DataFrame(preview), use_container_width=True)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            st.error(f"Ошибка HTTP от API: {exc.code}")
            st.code(body)
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
