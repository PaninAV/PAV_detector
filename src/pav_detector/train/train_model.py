from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pav_detector.core.scaler import StandardScalerLite
from pav_detector.utils.logging_json import configure_logging


class MLPClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class TrainingArtifacts:
    feature_order: List[str]
    classes: List[str]
    scaler: StandardScalerLite
    model: nn.Module
    metrics: Dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VPN/PROXY detector model on labeled flow CSV.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more labeled training CSV files",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="",
        help=(
            "Name of class label column. If empty, uses 'class_label' when present, "
            "otherwise uses the penultimate CSV column."
        ),
    )
    parser.add_argument(
        "--subtype-column",
        type=str,
        default="",
        help=(
            "Optional subtype column name (e.g., amnezia/wireguard). "
            "If empty and 'subtype' exists or the last column is non-numeric, it is excluded from features."
        ),
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        default="",
        help="Comma-separated feature columns. If empty, all numeric columns except label are used.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="LEGIT,VPN,PROXY",
        help="Comma-separated class order; must match inference config.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio [0,1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--export-onnx", action="store_true", help="Export model.onnx artifact")
    parser.add_argument("--out-dir", type=Path, default=Path("models"), help="Output artifacts directory")
    parser.add_argument(
        "--save-feature-order-json",
        type=Path,
        default=Path("models/feature_order.json"),
        help="Path to save feature order JSON list",
    )
    parser.add_argument(
        "--save-metrics-json",
        type=Path,
        default=Path("models/train_metrics.json"),
        help="Path to save train/val metrics JSON",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    return parser


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _encode_labels(series: pd.Series, classes: Sequence[str]) -> np.ndarray:
    mapping = {name: idx for idx, name in enumerate(classes)}
    encoded: List[int] = []
    unknown = set()
    for value in series.astype(str):
        value = value.strip()
        if value not in mapping:
            unknown.add(value)
            continue
        encoded.append(mapping[value])
    if unknown:
        raise ValueError(f"Found unknown labels not present in --classes: {sorted(unknown)}")
    return np.asarray(encoded, dtype=np.int64)


def _load_train_dataframe(csv_paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Training CSV not found: {path}")
        frames.append(pd.read_csv(path))
        print(f"[train] Loaded CSV: {path}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"[train] Total rows loaded: {len(combined)}")
    return combined


def _resolve_label_column(df: pd.DataFrame, explicit_label_column: str) -> str:
    if explicit_label_column:
        if explicit_label_column not in df.columns:
            raise ValueError(f"Label column '{explicit_label_column}' not found in CSV")
        return explicit_label_column

    if "class_label" in df.columns:
        return "class_label"

    if len(df.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns to auto-detect class label column.")
    return str(df.columns[-2])


def _resolve_subtype_column(
    df: pd.DataFrame,
    label_column: str,
    explicit_subtype_column: str,
) -> str | None:
    if explicit_subtype_column:
        if explicit_subtype_column not in df.columns:
            raise ValueError(f"Subtype column '{explicit_subtype_column}' not found in CSV")
        if explicit_subtype_column == label_column:
            raise ValueError("Subtype column must be different from label column.")
        return explicit_subtype_column

    if "subtype" in df.columns and "subtype" != label_column:
        return "subtype"

    last_col = str(df.columns[-1])
    if last_col != label_column and not pd.api.types.is_numeric_dtype(df[last_col]):
        return last_col
    return None


def _prepare_features(
    df: pd.DataFrame,
    label_column: str,
    subtype_column: str | None,
    selected_features: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV")

    if selected_features:
        feature_order = list(selected_features)
        missing = [col for col in feature_order if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in CSV: {missing}")
        features_df = df[feature_order].copy()
    else:
        numeric = df.select_dtypes(include=["number"]).copy()
        if label_column in numeric.columns:
            numeric = numeric.drop(columns=[label_column])
        if subtype_column and subtype_column in numeric.columns:
            numeric = numeric.drop(columns=[subtype_column])
        if numeric.shape[1] == 0:
            raise ValueError(
                "No numeric feature columns found. Pass --feature-columns explicitly."
            )
        feature_order = list(numeric.columns)
        features_df = numeric

    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    x = features_df.to_numpy(dtype=np.float32)
    y_labels = df[label_column]
    return x, y_labels.to_numpy(), feature_order


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predicted = torch.argmax(logits, dim=1)
    return float((predicted == targets).float().mean().item())


def _macro_f1_from_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1_values: List[float] = []
    for cls_idx in range(num_classes):
        tp = np.sum((y_true == cls_idx) & (y_pred == cls_idx))
        fp = np.sum((y_true != cls_idx) & (y_pred == cls_idx))
        fn = np.sum((y_true == cls_idx) & (y_pred != cls_idx))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_values))


def train(args: argparse.Namespace) -> TrainingArtifacts:
    configure_logging(args.log_level)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes = _parse_csv_list(args.classes)
    if not classes:
        raise ValueError("At least one class must be provided in --classes")

    df = _load_train_dataframe(args.train_csv)
    label_column = _resolve_label_column(df, args.label_column.strip())
    subtype_column = _resolve_subtype_column(df, label_column, args.subtype_column.strip())
    print(f"[train] Using label column: {label_column}")
    if subtype_column:
        print(f"[train] Detected subtype column (excluded from features): {subtype_column}")
    feature_cols = _parse_csv_list(args.feature_columns) if args.feature_columns else []
    x_raw, y_raw_labels, feature_order = _prepare_features(
        df,
        label_column=label_column,
        subtype_column=subtype_column,
        selected_features=feature_cols,
    )
    y = _encode_labels(pd.Series(y_raw_labels), classes)

    if len(x_raw) != len(y):
        raise RuntimeError("Feature/label length mismatch after preprocessing")

    indices = np.arange(len(x_raw))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    x_raw = x_raw[indices]
    y = y[indices]

    split_idx = int(len(x_raw) * (1.0 - args.val_split))
    if split_idx <= 0 or split_idx >= len(x_raw):
        raise ValueError("--val-split produces empty train or validation split")

    x_train_raw = x_raw[:split_idx]
    y_train = y[:split_idx]
    x_val_raw = x_raw[split_idx:]
    y_val = y[split_idx:]

    scaler = StandardScalerLite()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    train_ds = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = MLPClassifier(
        in_features=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        out_features=len(classes),
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    last_train_loss = 0.0
    last_val_loss = 0.0
    last_train_acc = 0.0
    last_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_acc_sum += _accuracy(logits, yb)
            train_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = criterion(val_logits, y_val_t)
            val_acc = _accuracy(val_logits, y_val_t)

        last_train_loss = train_loss_sum / max(train_batches, 1)
        last_train_acc = train_acc_sum / max(train_batches, 1)
        last_val_loss = float(val_loss.item())
        last_val_acc = val_acc

        if epoch % max(args.epochs // 10, 1) == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[train] epoch={epoch:03d}/{args.epochs} "
                f"train_loss={last_train_loss:.4f} train_acc={last_train_acc:.4f} "
                f"val_loss={last_val_loss:.4f} val_acc={last_val_acc:.4f}"
            )

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val_t)
        val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
    macro_f1 = _macro_f1_from_numpy(y_val, val_pred, num_classes=len(classes))

    metrics = {
        "train_loss": float(last_train_loss),
        "train_acc": float(last_train_acc),
        "val_loss": float(last_val_loss),
        "val_acc": float(last_val_acc),
        "val_macro_f1": float(macro_f1),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_features": int(x_train.shape[1]),
        "n_classes": int(len(classes)),
    }
    return TrainingArtifacts(
        feature_order=feature_order,
        classes=list(classes),
        scaler=scaler,
        model=model,
        metrics=metrics,
    )


def _save_artifacts(args: argparse.Namespace, artifacts: TrainingArtifacts) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = args.out_dir / "scaler.pkl"
    joblib.dump(artifacts.scaler, scaler_path)
    print(f"[train] Saved scaler: {scaler_path}")

    model_state_path = args.out_dir / "model_state_dict.pt"
    torch.save(artifacts.model.state_dict(), model_state_path)
    print(f"[train] Saved state dict: {model_state_path}")

    model_jit_path = args.out_dir / "model.pt"
    scripted = torch.jit.script(artifacts.model)
    scripted.save(str(model_jit_path))
    print(f"[train] Saved TorchScript model: {model_jit_path}")

    if args.export_onnx:
        model_onnx_path = args.out_dir / "model.onnx"
        sample = torch.randn(1, len(artifacts.feature_order), dtype=torch.float32)
        try:
            torch.onnx.export(
                artifacts.model,
                sample,
                str(model_onnx_path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )
            print(f"[train] Saved ONNX model: {model_onnx_path}")
        except ModuleNotFoundError as exc:
            missing = str(exc).split("'")
            missing_pkg = missing[1] if len(missing) >= 2 else "unknown"
            raise RuntimeError(
                "ONNX export failed because an extra dependency is missing: "
                f"{missing_pkg}. Install it with `pip install onnx onnxscript` "
                "or rerun training without --export-onnx."
            ) from exc

    args.save_feature_order_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_feature_order_json.write_text(
        json.dumps(artifacts.feature_order, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[train] Saved feature order: {args.save_feature_order_json}")

    args.save_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = dict(artifacts.metrics)
    metrics_payload["classes"] = artifacts.classes
    metrics_payload["feature_order_path"] = str(args.save_feature_order_json)
    args.save_metrics_json.write_text(
        json.dumps(metrics_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[train] Saved metrics: {args.save_metrics_json}")
    print(f"[train] Metrics summary: {json.dumps(artifacts.metrics, ensure_ascii=True)}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    artifacts = train(args)
    _save_artifacts(args, artifacts)


if __name__ == "__main__":
    main()
