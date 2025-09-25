"""
Clasificador TFLite optimizado para dispositivos edge (ej. Raspberry Pi).
"""

from __future__ import annotations

import os
from typing import Tuple
import numpy as np

try:
    import tflite_runtime.interpreter as tflite  # Preferido en Raspberry Pi
except ImportError:
    # Fallback a TensorFlow Lite de TensorFlow completo
    from tensorflow.lite import Interpreter as TFInterpreter
    from tensorflow.lite.python.interpreter import load_delegate

    class tflite:  # type: ignore
        Interpreter = TFInterpreter
        load_delegate = staticmethod(load_delegate)


class TFLiteAnimalClassifier:
    def __init__(self, model_path: str = "model/animal_classifier.tflite", labels_path: str = "model/labels.txt", use_edgetpu: bool = False):
        self.model_path = model_path
        self.labels_path = labels_path
        self.use_edgetpu = use_edgetpu
        self.labels = self._load_labels()
        self.interpreter = self._load_interpreter()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _load_labels(self):
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        # Fallback genérico
        return [f"class_{i}" for i in range(1000)]

    def _load_interpreter(self):
        if self.use_edgetpu and os.path.exists(self.model_path.replace(".tflite", "_edgetpu.tflite")):
            model = self.model_path.replace(".tflite", "_edgetpu.tflite")
            try:
                return tflite.Interpreter(
                    model_path=model,
                    experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
                )
            except Exception:
                pass
        return tflite.Interpreter(model_path=self.model_path)

    def predict(self, image_batched_rgb01: np.ndarray) -> Tuple[str, float]:
        # Ajustar al tamaño esperado
        input_shape = self.input_details[0]["shape"]
        _, h, w, c = input_shape
        img = image_batched_rgb01
        if img.shape[1] != h or img.shape[2] != w:
            import cv2

            img0 = img[0]
            img0 = cv2.resize((img0 * 255).astype(np.uint8), (w, h))
            img0 = img0.astype(np.float32) / 255.0
            img = np.expand_dims(img0, axis=0)

        # Tipado: float32 típico
        dtype = self.input_details[0]["dtype"]
        if dtype == np.uint8:
            tensor = (img * 255).astype(np.uint8)
        else:
            tensor = img.astype(dtype)

        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # Softmax si fuese necesario
        if output_data.ndim == 1 and np.abs(np.sum(output_data) - 1) > 1e-3:
            exps = np.exp(output_data - np.max(output_data))
            output_data = exps / np.sum(exps)

        idx = int(np.argmax(output_data))
        score = float(output_data[idx])
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        return label, score
