import base64
import cv2
import numpy as np


class CVProcessor:
    """OpenCV Computer Vision processing toolkit:

    Diagnostics, Filters, Contours, and Transformations.
    """

    @staticmethod
    def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes into an OpenCV BGR numpy array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(
                "Failed to decode image. Ensure the file is a valid image format."
            )
        return img

    @staticmethod
    def encode_image_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
        """Encode an OpenCV BGR image into a base64 data URI string."""
        success, buffer = cv2.imencode(ext, img_bgr)
        if not success:
            raise ValueError("Failed to encode image to base64.")
        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime = "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png"
        return f"data:{mime};base64,{b64_str}"

    @classmethod
    def get_diagnostics(
        cls, img_bgr: np.ndarray, blur_threshold: float = 100.0
    ) -> dict:
        """Compute comprehensive image diagnostics:

        Dimensions, aspect ratio, sharpness/blur metric, and color channel
        stats.
        """
        h, w, c = img_bgr.shape
        aspect_ratio = round(w / h, 2) if h > 0 else 1.0

        # Convert to grayscale for Laplacian blur estimation
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = laplacian_var < blur_threshold

        # Brightness estimation (average luminance)
        brightness = float(np.mean(gray))
        if brightness < 60:
            lighting = "Low Light (Dark)"
        elif brightness > 200:
            lighting = "Overexposed (Bright)"
        else:
            lighting = "Well Balanced"

        # Color channel distributions (BGR)
        mean_b = float(np.mean(img_bgr[:, :, 0]))
        mean_g = float(np.mean(img_bgr[:, :, 1]))
        mean_r = float(np.mean(img_bgr[:, :, 2]))

        # Sharpness score normalized (0 - 100 benchmark)
        sharpness_score = min(100.0, round((laplacian_var / 500.0) * 100, 1))

        return {
            "width": int(w),
            "height": int(h),
            "aspect_ratio": f"{aspect_ratio}:1",
            "channels": int(c),
            "laplacian_variance": round(laplacian_var, 2),
            "is_blurry": bool(is_blurry),
            "sharpness_score": sharpness_score,
            "brightness": round(brightness, 1),
            "lighting_quality": lighting,
            "color_distribution": {
                "red": round(mean_r, 1),
                "green": round(mean_g, 1),
                "blue": round(mean_b, 1),
            },
        }

    @classmethod
    def detect_saliency_and_contours(
        cls, img_bgr: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """Identify significant object contours and primary bounding boxes

        using OpenCV thresholding and contour hierarchy.
        """
        annotated = img_bgr.copy()
        h, w = annotated.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to smooth edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu thresholding after blur
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find external contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected_boxes = []
        min_area = (w * h) * 0.01  # at least 1% of total image area

        # Sort contours by area descending
        sorted_contours = sorted(
            [c for c in contours if cv2.contourArea(c) > min_area],
            key=cv2.contourArea,
            reverse=True,
        )

        # Top 3 major objects detected
        for i, cnt in enumerate(sorted_contours[:3]):
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Avoid full-screen boundary box
            if bw > 0.95 * w and bh > 0.95 * h:
                continue

            area = cv2.contourArea(cnt)
            detected_boxes.append(
                {
                    "id": i + 1,
                    "x": int(x),
                    "y": int(y),
                    "width": int(bw),
                    "height": int(bh),
                    "area_pct": round((area / (w * h)) * 100, 1),
                }
            )

            # Draw sleek HUD-like bounding box
            # Cyan box with corner ticks
            color = (0, 255, 230)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

            # Add label tag
            label = f"Region #{i + 1} ({round((area / (w * h)) * 100, 1)}%)"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                annotated,
                (x, y - 22),
                (x + t_size[0] + 8, y),
                (15, 23, 42),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return annotated, detected_boxes

    @classmethod
    def apply_filter(cls, img_bgr: np.ndarray, filter_name: str) -> np.ndarray:
        """Apply various Computer Vision filters and transformations to an

        image.
        """
        filter_name = filter_name.lower().strip()

        if filter_name == "grayscale":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif filter_name == "clahe":
            # Contrast Limited Adaptive Histogram Equalization on LAB color space
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        elif filter_name == "canny":
            # Edge detection with inverted aesthetic background
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 180)
            # Make neon cyan edges on dark background
            colored_edges = np.zeros_like(img_bgr)
            colored_edges[edges > 0] = [230, 255, 0]  # Cyan in BGR
            return colored_edges

        elif filter_name == "sharpen":
            # High-boost sharpening filter
            kernel = np.array(
                [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
            )
            return cv2.filter2D(img_bgr, -1, kernel)

        elif filter_name == "gaussian":
            # Gaussian blur for noise suppression
            return cv2.GaussianBlur(img_bgr, (15, 15), 0)

        elif filter_name == "sepia":
            # Vintage warm filter
            kernel = np.array(
                [
                    [0.272, 0.534, 0.131],
                    [0.349, 0.686, 0.168],
                    [0.393, 0.769, 0.189],
                ]
            )
            sepia = cv2.transform(img_bgr, kernel)
            return np.clip(sepia, 0, 255).astype(np.uint8)

        elif filter_name == "invert":
            # Negative color inversion
            return cv2.bitwise_not(img_bgr)

        # Default: return original untouched
        return img_bgr

    @classmethod
    def crop_region(
        cls, img_bgr: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """Safely crop a region of interest with bound clamping."""
        img_h, img_w = img_bgr.shape[:2]
        x1 = max(0, min(x, img_w - 1))
        y1 = max(0, min(y, img_h - 1))
        x2 = max(x1 + 1, min(x + w, img_w))
        y2 = max(y1 + 1, min(y + h, img_h))

        return img_bgr[y1:y2, x1:x2]
