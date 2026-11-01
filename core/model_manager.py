import logging
import threading
import time
import cv2
import numpy as np
import tensorflow as tf
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages TensorFlow/Keras vision models:

    lazy loading, caching, inference, and predictions decoding.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._models = {}
                cls._instance._preprocessors = {}
                cls._instance._target_sizes = {}
                cls._instance._init_model_specs()
        return cls._instance

    def _init_model_specs(self):
        """Define factory constructors and preprocessors for supported

        models.
        """
        self._specs = {
            "mobilenet_v3_large": {
                "loader": lambda: tf.keras.applications.MobileNetV3Large(
                    weights="imagenet"
                ),
                "preprocessor": tf.keras.applications.mobilenet_v3.preprocess_input,
                "target_size": (224, 224),
                "last_conv_name": "conv_1",
            },
            "mobilenet_v3_small": {
                "loader": lambda: tf.keras.applications.MobileNetV3Small(
                    weights="imagenet"
                ),
                "preprocessor": tf.keras.applications.mobilenet_v3.preprocess_input,
                "target_size": (224, 224),
                "last_conv_name": "conv_1",
            },
            "resnet50": {
                "loader": lambda: tf.keras.applications.ResNet50(
                    weights="imagenet"
                ),
                "preprocessor": tf.keras.applications.resnet50.preprocess_input,
                "target_size": (224, 224),
                "last_conv_name": "conv5_block3_out",
            },
            "efficientnet_b0": {
                "loader": lambda: tf.keras.applications.EfficientNetB0(
                    weights="imagenet"
                ),
                "preprocessor": tf.keras.applications.efficientnet.preprocess_input,
                "target_size": (224, 224),
                "last_conv_name": "top_conv",
            },
        }

    def get_model(self, model_name: str = None) -> tuple:
        """Retrieve or lazy-load model, preprocessor, and target size."""
        if not model_name or model_name not in self._specs:
            model_name = Config.DEFAULT_MODEL

        with self._lock:
            if model_name not in self._models:
                logger.info(
                    f"Loading model '{model_name}' weights from ImageNet..."
                )
                spec = self._specs[model_name]
                model = spec["loader"]()
                self._models[model_name] = model
                self._preprocessors[model_name] = spec["preprocessor"]
                self._target_sizes[model_name] = spec["target_size"]
                logger.info(f"Model '{model_name}' successfully loaded.")

            return (
                self._models[model_name],
                self._preprocessors[model_name],
                self._target_sizes[model_name],
            )

    @staticmethod
    def _clean_label(raw_label: str) -> str:
        """Format ImageNet label into human-friendly Title Case."""
        return raw_label.replace("_", " ").title()

    @staticmethod
    def _get_confidence_badge(score: float) -> dict:
        """Classify confidence into UX badge categories."""
        if score >= 0.80:
            return {
                "level": "high",
                "label": "High Confidence",
                "color": "#10B981",  # emerald
            }
        elif score >= 0.45:
            return {
                "level": "moderate",
                "label": "Moderate",
                "color": "#F59E0B",  # amber
            }
        else:
            return {
                "level": "low",
                "label": "Ambiguous / Low",
                "color": "#EF4444",  # rose
            }

    @staticmethod
    def _generate_description(label: str) -> str:
        """Generate a contextual helper description for the classified

        object.
        """
        return f"Identified as a {label} based on ImageNet visual feature embeddings."

    def predict(
        self, img_bgr: np.ndarray, model_name: str = None, top_k: int = None
    ) -> dict:
        """Run classification inference on an OpenCV BGR image."""
        if top_k is None:
            top_k = Config.TOP_K
        if not model_name:
            model_name = Config.DEFAULT_MODEL

        # 1. Measure Preprocessing Time
        t0 = time.perf_counter()
        model, preprocessor, target_size = self.get_model(model_name)

        # Convert OpenCV BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize to target dimension
        resized = cv2.resize(
            img_rgb, target_size, interpolation=cv2.INTER_AREA
        )
        # Expand dims for batch: (1, H, W, 3)
        img_array = np.expand_dims(resized.astype(np.float32), axis=0)
        # Apply model-specific preprocessing
        processed_input = preprocessor(img_array.copy())
        preprocess_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Measure Inference Time
        t1 = time.perf_counter()
        raw_predictions = model(processed_input, training=False).numpy()
        inference_time_ms = round((time.perf_counter() - t1) * 1000, 2)

        # 3. Decode predictions
        decoded = tf.keras.applications.imagenet_utils.decode_predictions(
            raw_predictions, top=top_k
        )[0]

        predictions_list = []
        for rank, (class_id, class_name, score) in enumerate(decoded, start=1):
            score_float = float(score)
            clean_name = self._clean_label(class_name)
            badge = self._get_confidence_badge(score_float)

            predictions_list.append(
                {
                    "rank": rank,
                    "class_id": class_id,
                    "label": clean_name,
                    "raw_label": class_name,
                    "confidence_pct": round(score_float * 100, 2),
                    "confidence_score": round(score_float, 4),
                    "badge": badge,
                    "description": self._generate_description(clean_name),
                }
            )

        top_prediction = (
            predictions_list[0]
            if predictions_list
            else {
                "label": "Unknown",
                "confidence_pct": 0.0,
                "badge": self._get_confidence_badge(0.0),
            }
        )

        return {
            "model_name": model_name,
            "model_info": Config.AVAILABLE_MODELS.get(model_name, {}),
            "top_prediction": top_prediction,
            "predictions": predictions_list,
            "telemetry": {
                "preprocess_time_ms": preprocess_time_ms,
                "inference_time_ms": inference_time_ms,
                "total_time_ms": round(
                    preprocess_time_ms + inference_time_ms, 2
                ),
            },
        }


# Global Singleton instance
model_manager = ModelManager()
