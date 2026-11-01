import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Application configuration settings."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "image-recognition-pro-secret-key-2026")
    
    # Upload Directories
    UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
    SAMPLE_FOLDER = BASE_DIR / "static" / "samples"
    
    # Max file size: 16 Megabytes
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    # Supported image extensions
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
    
    # Default model
    DEFAULT_MODEL = "mobilenet_v3_large"
    
    # Model catalog & specifications
    AVAILABLE_MODELS = {
        "mobilenet_v3_large": {
            "name": "MobileNetV3 Large",
            "tagline": "Fast & High Precision",
            "description": "Optimized lightweight convolutional network with hard-swish activations.",
            "target_size": (224, 224),
            "speed": "Ultra Fast (~15-30ms)",
            "accuracy": "High (Top-1: 75.2%)",
            "parameters": "~5.4 Million",
            "recommended_for": "Real-time webcam & rapid image screening"
        },
        "mobilenet_v3_small": {
            "name": "MobileNetV3 Small",
            "tagline": "Ultra Lightweight",
            "description": "Designed for resource-constrained edge inference with minimum latency.",
            "target_size": (224, 224),
            "speed": "Instant (~8-15ms)",
            "accuracy": "Moderate (Top-1: 67.4%)",
            "parameters": "~2.5 Million",
            "recommended_for": "Continuous live camera stream"
        },
        "resnet50": {
            "name": "ResNet50",
            "tagline": "Deep Residual Network",
            "description": "Classic 50-layer deep architecture with residual bottleneck skip connections.",
            "target_size": (224, 224),
            "speed": "Moderate (~40-80ms)",
            "accuracy": "Very High (Top-1: 76.0%)",
            "parameters": "~25.6 Million",
            "recommended_for": "Complex cluttered scenes & high-detail objects"
        },
        "efficientnet_b0": {
            "name": "EfficientNet-B0",
            "tagline": "SOTA Compound Scaled",
            "description": "Neural architecture search scaled uniformly across depth, width, and resolution.",
            "target_size": (224, 224),
            "speed": "Fast (~25-45ms)",
            "accuracy": "State-of-the-Art (Top-1: 77.1%)",
            "parameters": "~5.3 Million",
            "recommended_for": "Optimal balance of accuracy and efficiency"
        }
    }
    
    # OpenCV Computer Vision Settings
    BLUR_THRESHOLD = 100.0  # Variance of Laplacian below this indicates blurry
    
    # Top K Predictions returned to client
    TOP_K = 5
