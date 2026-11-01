import logging
import cv2
import numpy as np
import tensorflow as tf
from core.cv_processor import CVProcessor
from core.model_manager import model_manager

logger = logging.getLogger(__name__)


class GradCAM:
    """Explainable AI (XAI) using Gradient-weighted Class Activation Mapping

    (Grad-CAM).

    Generates visual heatmaps highlighting the image regions that most
    influenced the model's classification decision.
    """

    @staticmethod
    def _find_last_conv_layer(model, preferred_layer_name: str = None) -> str:
        """Locate the target spatial 4D convolutional layer."""
        if preferred_layer_name:
            try:
                layer = model.get_layer(preferred_layer_name)
                return layer.name
            except ValueError:
                pass

        # Traverse backwards to find a conv layer with spatial dimensions > 1
        for layer in reversed(model.layers):
            try:
                out_shape = layer.output.shape
                # 4D tensor where spatial dimensions are greater than 1x1
                if len(out_shape) == 4 and (out_shape[1] is None or out_shape[1] > 1):
                    if any(k in layer.name.lower() for k in ["conv", "depthwise", "activation"]):
                        return layer.name
            except Exception:
                continue

        # Fallback to any 4D layer
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    return layer.name
            except Exception:
                continue

        raise ValueError(f"Could not locate suitable conv layer in model {model.name}")

    @classmethod
    def generate_heatmap(
        cls,
        img_bgr: np.ndarray,
        model_name: str = None,
        class_index: int = None,
        alpha: float = 0.45,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Generate Grad-CAM heatmap and blended visualization.

        Returns:
            blended_bgr: OpenCV image with heatmap overlaid.
            heatmap_raw: 2D numpy array (0..1).
            heatmap_b64: base64 data URI of the blended visualization.
        """
        model, preprocessor, target_size = model_manager.get_model(model_name)
        orig_h, orig_w = img_bgr.shape[:2]

        # Prepare input tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
        img_array = np.expand_dims(resized.astype(np.float32), axis=0)
        processed_input = preprocessor(img_array.copy())

        # Determine target conv layer
        pref_name = model_manager._specs.get(model_name, {}).get("last_conv_name")
        conv_layer_name = cls._find_last_conv_layer(model, pref_name)
        conv_layer = model.get_layer(conv_layer_name)

        # Construct gradient model using model.input
        model_input = model.input if hasattr(model, "input") else model.inputs[0]
        grad_model = tf.keras.models.Model(
            inputs=model_input,
            outputs=[conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed_input)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            
            # Score for the target class
            loss = predictions[:, class_index]

        # Gradient of class score with respect to feature maps
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            grads = tf.ones_like(conv_outputs)

        # Mean intensity of the gradient over each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels of the feature map by the pooled gradients
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to retain only positive influence
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

        # Handle 1D or degenerate shapes gracefully
        if len(heatmap_np.shape) < 2:
            heatmap_np = np.ones((7, 7), dtype=np.float32) * float(heatmap_np)

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap_np.astype(np.float32), (orig_w, orig_h))
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

        # Apply JET colormap (blue = low activation, red = high activation)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Superimpose heatmap onto original image
        blended = cv2.addWeighted(img_bgr, 1 - alpha, color_heatmap, alpha, 0)

        # Encode to base64
        b64 = CVProcessor.encode_image_base64(blended)

        return blended, heatmap_resized, b64
