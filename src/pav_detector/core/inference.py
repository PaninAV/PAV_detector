from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from pav_detector.core.scaler import StandardScalerLite

logger = logging.getLogger(__name__)


@dataclass
class InferenceOutput:
    predicted_index: int
    probabilities: np.ndarray


class InferenceEngine:
    def __init__(
        self,
        model_onnx_path: Path,
        model_torch_path: Path,
        scaler_path: Path,
        classes: Sequence[str],
    ) -> None:
        self.model_onnx_path = model_onnx_path
        self.model_torch_path = model_torch_path
        self.scaler_path = scaler_path
        self.classes = list(classes)

        self._scaler = None
        self._onnx_session = None
        self._torch_model = None
        self._torch = None
        self._torch_device = None
        self.backend: Optional[str] = None
        self._load_assets()

    def _load_assets(self) -> None:
        if self.scaler_path.exists():
            self._scaler = _load_scaler_with_legacy_support(self.scaler_path)
            logger.info("Loaded scaler from %s", self.scaler_path)
        else:
            logger.warning("Scaler not found at %s; raw features will be used.", self.scaler_path)

        if self.model_onnx_path.exists():
            import onnxruntime as ort

            self._onnx_session = ort.InferenceSession(str(self.model_onnx_path), providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
            logger.info("Loaded ONNX model from %s", self.model_onnx_path)
            return

        if self.model_torch_path.exists():
            import torch

            self._torch = torch
            self._torch_device = torch.device("cpu")
            self._torch_model = torch.jit.load(str(self.model_torch_path), map_location=self._torch_device)
            self._torch_model.eval()
            self.backend = "pytorch"
            logger.info("Loaded PyTorch model from %s", self.model_torch_path)
            return

        raise FileNotFoundError(
            f"No model found. Expected {self.model_onnx_path} or {self.model_torch_path}"
        )

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        x = features.astype(np.float32)
        if self._scaler is not None:
            x = self._scaler.transform(x).astype(np.float32)
        return x

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> InferenceOutput:
        x = self.preprocess(features)

        if self.backend == "onnxruntime":
            assert self._onnx_session is not None
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: x})
            logits = np.asarray(outputs[0], dtype=np.float32)
            probabilities = self._softmax(logits)
        elif self.backend == "pytorch":
            assert self._torch is not None and self._torch_model is not None
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(x)
                logits_tensor = self._torch_model(tensor)
                logits = logits_tensor.cpu().numpy().astype(np.float32)
            probabilities = self._softmax(logits)
        else:
            raise RuntimeError("Inference backend not initialized")

        predicted_index = int(np.argmax(probabilities, axis=1)[0])
        return InferenceOutput(predicted_index=predicted_index, probabilities=probabilities[0])

    def probabilities_as_dict(self, probabilities: np.ndarray) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for idx, cls_name in enumerate(self.classes):
            value = float(probabilities[idx]) if idx < len(probabilities) else 0.0
            result[cls_name] = value
        return result


def _load_scaler_with_legacy_support(path: Path):
    try:
        return joblib.load(path)
    except AttributeError as exc:
        message = str(exc)
        if "StandardScalerLite" not in message:
            raise
        logger.warning(
            "Default scaler loading failed, trying legacy compatibility mode for %s", path
        )
        main_module = sys.modules.get("__main__")
        if main_module is None:
            raise

        had_attr = hasattr(main_module, "StandardScalerLite")
        previous_value = getattr(main_module, "StandardScalerLite", None)
        try:
            # Legacy fix: old scaler pickles may reference __main__.StandardScalerLite.
            setattr(main_module, "StandardScalerLite", StandardScalerLite)
            return joblib.load(path)
        finally:
            if had_attr:
                setattr(main_module, "StandardScalerLite", previous_value)
            else:
                delattr(main_module, "StandardScalerLite")
