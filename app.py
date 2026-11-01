import base64
import os
import time
import urllib.request
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from config import Config
from core.cv_processor import CVProcessor
from core.gradcam import GradCAM
from core.model_manager import model_manager

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["SAMPLE_FOLDER"], exist_ok=True)

START_TIME = time.time()


# ============================================================================
# HELPER UTILITIES
# ============================================================================
def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a permitted image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def extract_image_from_request() -> tuple[np.ndarray, str]:
    """Extract and decode image from either multipart form-data, JSON base64,

    image URL, or sample name.
    """
    img_bgr = None
    source_name = "image"

    # 1. Direct file upload
    if "image" in request.files:
        file = request.files["image"]
        if file and file.filename != "":
            if not allowed_file(file.filename):
                raise ValueError("File type not allowed. Supported: JPG, PNG, WEBP, BMP.")
            source_name = secure_filename(file.filename)
            file_bytes = file.read()
            img_bgr = CVProcessor.decode_image_bytes(file_bytes)

    # 2. JSON Payload (base64, URL, or sample preset)
    elif request.is_json:
        data = request.get_json() or {}

        # 2a. Base64 string
        if "image" in data and data["image"]:
            b64_data = data["image"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            img_bgr = CVProcessor.decode_image_bytes(img_bytes)
            source_name = data.get("filename", "captured_frame.jpg")

        # 2b. Remote Image URL
        elif "imageUrl" in data and data["imageUrl"]:
            url = data["imageUrl"].strip()
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                img_bytes = response.read()
            img_bgr = CVProcessor.decode_image_bytes(img_bytes)
            source_name = url.split("/")[-1].split("?")[0] or "remote_image.jpg"

        # 2c. Built-in Sample Image
        elif "sampleName" in data and data["sampleName"]:
            sample_filename = secure_filename(data["sampleName"])
            sample_path = Config.SAMPLE_FOLDER / sample_filename
            if not sample_path.exists():
                raise FileNotFoundError(f"Sample image '{sample_filename}' not found.")
            img_bgr = cv2.imread(str(sample_path))
            source_name = sample_filename

    # 3. Form-encoded base64 / URL fallback
    elif "imageBase64" in request.form:
        b64_data = request.form["imageBase64"]
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        img_bgr = CVProcessor.decode_image_bytes(img_bytes)
        source_name = "webcam_snap.jpg"

    if img_bgr is None:
        raise ValueError("No valid image provided via upload, base64, URL, or sample.")

    return img_bgr, source_name


# ============================================================================
# WEB PAGES / VIEW ROUTES
# ============================================================================
@app.route("/")
def index():
    """Main dashboard: Single Image Studio, Webcam, CV Diagnostics, Grad-CAM."""
    return render_template(
        "index.html",
        models=Config.AVAILABLE_MODELS,
        default_model=Config.DEFAULT_MODEL
    )


@app.route("/compare")
def compare():
    """Side-by-side dual image comparison view."""
    return render_template("compare.html", models=Config.AVAILABLE_MODELS)


@app.route("/batch")
def batch():
    """Batch multi-image classification and CSV export view."""
    return render_template("batch.html", models=Config.AVAILABLE_MODELS)


@app.route("/api-docs")
def api_docs():
    """Interactive RESTful API documentation."""
    return render_template("api_docs.html", models=Config.AVAILABLE_MODELS)


@app.route("/about")
def about():
    """Technical architecture, benchmarks, and model overview."""
    return render_template("about.html", models=Config.AVAILABLE_MODELS)


# ============================================================================
# REST API ENDPOINTS
# ============================================================================
@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Primary classification endpoint with OpenCV diagnostics and Grad-CAM.

    Query/Form/JSON Params:
      - model: target model name
      - filter: optional OpenCV filter to apply first
      - gradcam: 'true'/'1' to return Grad-CAM visualization
      - saliency: 'true'/'1' to return contour bounding boxes
    """
    try:
        img_bgr, source_name = extract_image_from_request()

        # Parse options
        model_name = (
            request.args.get("model")
            or (request.is_json and request.json.get("model"))
            or request.form.get("model")
            or Config.DEFAULT_MODEL
        )
        filter_name = (
            request.args.get("filter")
            or (request.is_json and request.json.get("filter"))
            or request.form.get("filter")
            or "none"
        )
        include_gradcam = (
            str(
                request.args.get("gradcam")
                or (request.is_json and request.json.get("gradcam"))
                or request.form.get("gradcam", "")
            ).lower()
            in ["true", "1", "yes"]
        )
        include_saliency = (
            str(
                request.args.get("saliency")
                or (request.is_json and request.json.get("saliency"))
                or request.form.get("saliency", "")
            ).lower()
            in ["true", "1", "yes"]
        )

        # Apply OpenCV filter if specified
        if filter_name and filter_name != "none":
            img_bgr = CVProcessor.apply_filter(img_bgr, filter_name)

        # 1. OpenCV Diagnostics
        diagnostics = CVProcessor.get_diagnostics(img_bgr, Config.BLUR_THRESHOLD)

        # 2. Saliency & Object Contour Detection
        saliency_boxes = []
        annotated_b64 = None
        if include_saliency:
            annotated_img, saliency_boxes = CVProcessor.detect_saliency_and_contours(img_bgr)
            annotated_b64 = CVProcessor.encode_image_base64(annotated_img)

        # 3. Model Classification
        classification_result = model_manager.predict(img_bgr, model_name)

        # 4. Optional Grad-CAM Heatmap
        gradcam_b64 = None
        if include_gradcam:
            try:
                top_pred_rank1 = classification_result["predictions"][0]
                _, _, gradcam_b64 = GradCAM.generate_heatmap(img_bgr, model_name)
            except Exception as e:
                app.logger.warning(f"Grad-CAM generation failed: {e}")

        # Original/processed preview base64
        preview_b64 = CVProcessor.encode_image_base64(img_bgr)

        return jsonify({
            "status": "success",
            "source_name": source_name,
            "filter_applied": filter_name,
            "diagnostics": diagnostics,
            "saliency_boxes": saliency_boxes,
            "saliency_image": annotated_b64,
            "image_b64": preview_b64,
            "gradcam_b64": gradcam_b64,
            **classification_result
        })

    except Exception as e:
        app.logger.error(f"Classification error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    """Generate Grad-CAM activation heatmap for the given image."""
    try:
        img_bgr, _ = extract_image_from_request()
        model_name = (
            request.args.get("model")
            or (request.is_json and request.json.get("model"))
            or Config.DEFAULT_MODEL
        )
        _, _, gradcam_b64 = GradCAM.generate_heatmap(img_bgr, model_name)
        return jsonify({"status": "success", "gradcam_b64": gradcam_b64})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/filter", methods=["POST"])
def api_filter():
    """Apply an OpenCV filter and return the transformed image and diagnostics."""
    try:
        img_bgr, _ = extract_image_from_request()
        filter_name = (
            request.args.get("filter")
            or (request.is_json and request.json.get("filter"))
            or "grayscale"
        )
        filtered = CVProcessor.apply_filter(img_bgr, filter_name)
        filtered_b64 = CVProcessor.encode_image_base64(filtered)
        diagnostics = CVProcessor.get_diagnostics(filtered, Config.BLUR_THRESHOLD)
        return jsonify({
            "status": "success",
            "filter": filter_name,
            "image_b64": filtered_b64,
            "diagnostics": diagnostics
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/crop-classify", methods=["POST"])
def api_crop_classify():
    """Crop a bounding box region from an image and classify the crop."""
    try:
        img_bgr, _ = extract_image_from_request()
        data = request.get_json() if request.is_json else request.form
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        w = int(data.get("width", 50))
        h = int(data.get("height", 50))
        model_name = data.get("model", Config.DEFAULT_MODEL)

        cropped_bgr = CVProcessor.crop_region(img_bgr, x, y, w, h)
        cropped_b64 = CVProcessor.encode_image_base64(cropped_bgr)
        classification = model_manager.predict(cropped_bgr, model_name)
        diagnostics = CVProcessor.get_diagnostics(cropped_bgr)

        return jsonify({
            "status": "success",
            "cropped_b64": cropped_b64,
            "diagnostics": diagnostics,
            **classification
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/batch-classify", methods=["POST"])
def api_batch_classify():
    """Batch classify multiple images uploaded at once."""
    try:
        model_name = request.form.get("model", Config.DEFAULT_MODEL)
        files = request.files.getlist("images")

        if not files or files[0].filename == "":
            raise ValueError("No files provided for batch classification.")

        results = []
        for file in files:
            if not allowed_file(file.filename):
                continue
            filename = secure_filename(file.filename)
            file_bytes = file.read()
            img_bgr = CVProcessor.decode_image_bytes(file_bytes)
            classification = model_manager.predict(img_bgr, model_name)
            thumb_b64 = CVProcessor.encode_image_base64(
                cv2.resize(img_bgr, (100, 100), interpolation=cv2.INTER_AREA)
            )

            results.append({
                "filename": filename,
                "thumbnail_b64": thumb_b64,
                "top_label": classification["top_prediction"]["label"],
                "confidence_pct": classification["top_prediction"]["confidence_pct"],
                "badge": classification["top_prediction"]["badge"],
                "inference_time_ms": classification["telemetry"]["inference_time_ms"],
                "all_predictions": classification["predictions"]
            })

        return jsonify({"status": "success", "count": len(results), "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/models", methods=["GET"])
def api_models():
    """Return catalog of available models and specifications."""
    return jsonify({
        "status": "success",
        "default_model": Config.DEFAULT_MODEL,
        "models": Config.AVAILABLE_MODELS
    })


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Return available built-in sample images."""
    samples = []
    if Config.SAMPLE_FOLDER.exists():
        for file in os.listdir(Config.SAMPLE_FOLDER):
            if any(file.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
                clean_name = file.rsplit(".", 1)[0].replace("_", " ").title()
                samples.append({
                    "id": file,
                    "title": clean_name,
                    "url": f"/static/samples/{file}"
                })
    return jsonify({"status": "success", "samples": samples})


@app.route("/api/health", methods=["GET"])
def api_health():
    """System health check and runtime telemetry."""
    import platform
    import tensorflow as tf

    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0

    return jsonify({
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "opencv_version": cv2.__version__,
        "flask_version": app.config.get("ENV", "production"),
        "hardware": {
            "gpu_accelerated": gpu_available,
            "device": "GPU (CUDA)" if gpu_available else "CPU (oneDNN / AVX2 Optimized)"
        }
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
