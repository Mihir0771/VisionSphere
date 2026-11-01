<div align="center">

# 👁️ VisionSphere — Image Recognition System
### Enterprise-Grade Computer Vision & Deep Learning Object Recognition Platform

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.22-orange.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-red.svg?logo=opencv&logoColor=white)](https://opencv.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A full-stack, real-time Computer Vision web application featuring multi-model neural inference, explainable Grad-CAM activation heatmaps, OpenCV visual diagnostics, and interactive studio tooling.*

[✨ Live Features](#-key-features) • [🏛️ Architecture](#-system-architecture) • [🚀 Quickstart](#-quickstart--installation) • [🔌 REST API](#-restful-api-reference) • [📊 Benchmarks](#-model-benchmarks)

---

</div>

## 🌟 Overview

**VisionSphere** is a high-performance deep learning platform engineered to identify objects across 1,000 distinct categories with high accuracy and explainability. Designed for production readiness, the system integrates **TensorFlow/Keras** neural backbones with an **OpenCV** computer vision pipeline and a sleek, cyber-glassmorphic **Flask** interface.

---

## ⚡ Key Features

| Category | Capability | Description |
| :--- | :--- | :--- |
| **🧠 Deep Learning** | **Multi-Model Engine** | Dynamically switch between **MobileNetV3 (Small/Large)**, **ResNet50**, and **EfficientNet-B0** on the fly with lazy weight caching. |
| **🔥 Explainable AI** | **Grad-CAM Heatmaps** | Gradient-weighted Class Activation Mapping visualizes *where* the convolutional network focused its attention using OpenCV JET colormaps. |
| **👁️ Computer Vision** | **OpenCV Diagnostics** | Evaluates image blurriness via **Laplacian variance**, checks lighting/brightness, computes RGB channel distributions, and detects primary object contours. |
| **🎨 Image Filters** | **Pre-Classification Lab** | Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization), **Canny Edge Detection**, **Sharpening**, and **Gaussian Denoising** before inference. |
| **📥 Omnichannel Input** | **Multi-Source Ingestion** | 1-Click File Drag & Drop, **Clipboard Paste (`Ctrl+V`)**, direct **Remote Image URLs**, and **Built-in Benchmark Samples**. |
| **📹 Live Stream** | **Webcam Studio** | High-framerate browser camera streaming with snapshot classification and continuous auto-detection mode. |
| **✂️ Interactive Tools** | **Studio ROI Cropper** | Click and drag directly on the image viewport to crop and classify specific sub-objects in cluttered scenes. |
| **⚖️ Benchmarking** | **Side-by-Side Compare** | Dual-slot comparison mode to benchmark different architectures or compare two images simultaneously. |
| **📦 High-Throughput** | **Batch Classifier** | Process up to 50 images concurrently with progress tracking and one-click export to structured **CSV** and **JSON**. |
| **🔊 Accessibility** | **Voice Announcer** | Text-to-Speech (TTS) engine speaks aloud identified objects via the Web Speech API. |
| **🔌 Developer Ready** | **RESTful Vision API** | Clean, modular JSON endpoints (`/api/classify`, `/api/gradcam`, `/api/filter`, `/api/health`) ready for external apps. |

---

## 🏛️ System Architecture

```text
                                  +---------------------------------------+
                                  |     Omnichannel Ingestion Layer       |
                                  |  (Dropzone / Ctrl+V / Webcam / URL)   |
                                  +---------------------------------------+
                                                     |
                                                     v
                                  +---------------------------------------+
                                  |         Flask Application Core        |
                                  |       (Routing, Telemetry & API)      |
                                  +---------------------------------------+
                                          |                       |
                                          v                       v
                    +---------------------------+   +---------------------------+
                    |    OpenCV Processor Lab   |   |   TensorFlow Model Pool   |
                    |---------------------------|   |---------------------------|
                    | • Laplacian Blur Score    |   | • MobileNetV3 (Small/Lrg) |
                    | • CLAHE Contrast Boost    |   | • ResNet50 Residual Net   |
                    | • Canny Edge Detector     |   | • EfficientNet-B0         |
                    | • Saliency Contours       |   | • ImageNet-1k Softmax     |
                    +---------------------------+   +---------------------------+
                                          \                       /
                                           \                     /
                                            v                   v
                                  +---------------------------------------+
                                  |     Explainable AI Engine (Grad-CAM)  |
                                  |   (Gradient Tape + Colormap Overlay)  |
                                  +---------------------------------------+
                                                     |
                                                     v
                                  +---------------------------------------+
                                  |     Interactive Glassmorphism UI      |
                                  |  (Top-5 Probabilities, HUD, History)  |
                                  +---------------------------------------+
```

---

## 📊 Model Benchmarks

| Backbone Model | Parameters | Target Size | Typical Latency | Primary Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **MobileNetV3 Small** | ~2.5M | 224 x 224 | **~10 - 20 ms** | Real-time continuous webcam streams & edge devices |
| **MobileNetV3 Large** | ~5.4M | 224 x 224 | **~20 - 35 ms** | Balanced general-purpose classification |
| **EfficientNet-B0** | ~5.3M | 224 x 224 | **~30 - 50 ms** | High-precision compound-scaled object identification |
| **ResNet50** | ~25.6M | 224 x 224 | **~45 - 80 ms** | Deep feature extraction in complex cluttered scenes |

---

## 📁 Repository Structure

```text
Image Recognition/
│
├── app.py                     # Main Flask web application & REST API routes
├── config.py                  # Global configurations, model definitions & thresholds
├── requirements.txt           # Python package dependencies
├── LICENSE                    # MIT Open-Source License
├── README.md                  # Project documentation
│
├── core/
│   ├── __init__.py
│   ├── model_manager.py       # TensorFlow model loader, thread-safe cache & Top-K decoder
│   ├── gradcam.py             # Gradient-weighted Class Activation Mapping engine
│   └── cv_processor.py        # OpenCV image filters, blur metric & contour saliency
│
├── static/
│   ├── css/
│   │   └── style.css          # Glassmorphic cyberpunk styling & UI animations
│   ├── js/
│   │   ├── app.js             # Main studio controller, drag-drop, paste & telemetry
│   │   ├── webcam.js          # HTML5 camera stream & continuous classifier
│   │   ├── cropper.js         # Interactive canvas ROI selection & crop classifier
│   │   ├── compare.js         # Side-by-side dual image comparison controller
│   │   └── batch.js           # Multi-file batch classification & CSV exporter
│   ├── samples/               # Built-in benchmark images (Golden Retriever, Tiger, etc.)
│   └── uploads/               # Temporary uploads storage (auto-cleaned)
│
└── templates/
    ├── base.html              # Base HTML template with navbar, footer & toasts
    ├── index.html             # Main Vision Studio dashboard
    ├── compare.html           # Side-by-side comparison page
    ├── batch.html             # Batch classification & dataset exporter
    ├── api_docs.html          # Interactive RESTful API documentation
    └── about.html             # Engineering specs & architecture deep dive
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/image-recognition-system.git
cd image-recognition-system
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🔌 RESTful API Reference

### 1. Classify Image
`POST /api/classify`

**Multipart Form Upload:**
```bash
curl -X POST http://127.0.0.1:5000/api/classify \
  -F "image=@golden_retriever.jpg" \
  -F "model=mobilenet_v3_large" \
  -F "gradcam=true" \
  -F "saliency=true"
```

**JSON Payload (Base64 or URL):**
```bash
curl -X POST http://127.0.0.1:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "imageUrl": "https://example.com/tiger.jpg",
    "model": "mobilenet_v3_large"
  }'
```

**Sample JSON Response:**
```json
{
  "status": "success",
  "model_name": "mobilenet_v3_large",
  "top_prediction": {
    "rank": 1,
    "class_id": "n02129604",
    "label": "Tiger",
    "confidence_pct": 84.67,
    "badge": { "level": "high", "label": "High Confidence", "color": "#10B981" },
    "description": "Identified as a Tiger based on ImageNet visual feature embeddings."
  },
  "predictions": [
    { "rank": 1, "label": "Tiger", "confidence_pct": 84.67 },
    { "rank": 2, "label": "Tiger Cat", "confidence_pct": 4.12 }
  ],
  "diagnostics": {
    "width": 600,
    "height": 400,
    "sharpness_score": 86.4,
    "is_blurry": false,
    "lighting_quality": "Well Balanced"
  },
  "telemetry": {
    "preprocess_time_ms": 1.84,
    "inference_time_ms": 22.15,
    "total_time_ms": 23.99
  }
}
```

### 2. Available Models
`GET /api/models`
Returns list of all registered models, memory parameters, and input dimensions.

### 3. Health Check
`GET /api/health`
Returns system status, active hardware accelerators, and framework versions.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>Built with ❤️ by Mihir</b>
</div>
