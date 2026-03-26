from __future__ import annotations

from argparse import Namespace
import json
import tempfile
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
        "Train CSV",
        type=["csv"],
        key="train_csv_uploader",
    )

    col1, col2, col3 = st.columns(3)
    epochs = col1.number_input("Epochs", min_value=1, max_value=10000, value=40, step=1)
    batch_size = col2.number_input("Batch size", min_value=1, max_value=100000, value=512, step=1)
    lr = col3.number_input("Learning rate", min_value=0.000001, max_value=1.0, value=0.001, step=0.0005, format="%.6f")

    classes_raw = st.text_input("Classes (comma-separated)", value="LEGIT,VPN,PROXY")
    export_onnx = st.checkbox("Export ONNX", value=True)

    if st.button("Запустить обучение", type="primary"):
        if uploaded_train_csv is None:
            st.error("Сначала выберите train CSV.")
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
            st.json(artifacts.metrics)
        except Exception as exc:
            st.exception(exc)


def _run_offline_section() -> None:
    st.subheader("2) Оффлайн проверка CSV")
    st.caption("Запускает классификацию по загруженному CSV и сохраняет suspicious события в БД.")

    uploaded_offline_csv = st.file_uploader(
        "Offline CSV",
        type=["csv"],
        key="offline_csv_uploader",
    )
    sensor_name = st.text_input("Sensor name", value="workbench-offline", key="offline_sensor_name")

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
                detection = service.classify_flow(flow=flow, sensor_name=sensor_name)
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


def _run_online_section() -> None:
    st.subheader("3) Онлайн проверка (эмуляция потока в API)")
    st.caption("Загружает CSV и отправляет flow-строки в /v1/classify. API должен быть запущен заранее.")

    uploaded_online_csv = st.file_uploader(
        "Online CSV",
        type=["csv"],
        key="online_csv_uploader",
    )
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000/v1/classify")
    sensor_name = st.text_input("Sensor name", value="workbench-online", key="online_sensor_name")
    max_rows = st.number_input("Max rows to send", min_value=1, max_value=1000000, value=500, step=1)

    if st.button("Запустить онлайн отправку в API"):
        if uploaded_online_csv is None:
            st.error("Сначала выберите CSV для онлайн отправки.")
            return
        try:
            df = _uploaded_csv_to_dataframe(uploaded_online_csv)
            df = df.head(int(max_rows))

            sent = 0
            alerts = 0
            preview = []
            progress = st.progress(0)
            total = max(len(df), 1)

            for idx, (_, row) in enumerate(df.iterrows(), start=1):
                flow = {str(key): _normalize_value(value) for key, value in row.items()}
                payload = {"sensor_name": sensor_name, "flow": flow}
                result = _post_flow(api_url, payload)

                sent += 1
                if result.get("should_alert"):
                    alerts += 1
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

            st.success(f"Отправлено flow: {sent}, алертов: {alerts}")
            st.dataframe(pd.DataFrame(preview), use_container_width=True)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            st.error(f"HTTP error from API: {exc.code}")
            st.code(body)
        except Exception as exc:
            st.exception(exc)


def main() -> None:
    st.set_page_config(page_title="PAV Detector Workbench", layout="wide")
    st.title("PAV Detector - Workbench")
    st.caption("Интерфейс запуска обучения, оффлайн/онлайн проверки и просмотра результатов из одного окна.")

    with st.expander("Перед запуском"):
        st.markdown(
            "- Для оффлайн/онлайн режимов модель в `models/` должна существовать.\n"
            "- Для онлайн режима сначала поднимите API: `uvicorn pav_detector.api.app:app --host 0.0.0.0 --port 8000`.\n"
            "- Для записи алертов в БД убедитесь, что `ENABLE_DB=true` в `.env`."
        )

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


if __name__ == "__main__":
    main()
