import unittest
from app import app


class VisionSphereTestCase(unittest.TestCase):
    """Automated integration tests for VisionSphere endpoints."""

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_pages_render(self):
        """Verify that all web views render successfully."""
        routes = ["/", "/compare", "/batch", "/api-docs", "/about"]
        for route in routes:
            response = self.client.get(route)
            self.assertEqual(
                response.status_code, 200, f"Route {route} failed to render."
            )

    def test_api_health(self):
        """Verify health check endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("tensorflow_version", data)
        self.assertIn("opencv_version", data)

    def test_api_models(self):
        """Verify model catalog endpoint."""
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("mobilenet_v3_small", data["models"])
        self.assertIn("mobilenet_v3_large", data["models"])
        self.assertIn("resnet50", data["models"])
        self.assertIn("efficientnet_b0", data["models"])

    def test_api_samples(self):
        """Verify samples catalog endpoint."""
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(len(data["samples"]) > 0)

    def test_api_classify_sample(self):
        """Verify real classification on a sample image."""
        response = self.client.post(
            "/api/classify",
            json={
                "sampleName": "tiger.jpg",
                "model": "mobilenet_v3_small",
                "gradcam": True,
                "saliency": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Tiger", data["top_prediction"]["label"])
        self.assertIsNotNone(data["gradcam_b64"])
        self.assertIn("diagnostics", data)
        self.assertIn("sharpness_score", data["diagnostics"])

    def test_api_filter(self):
        """Verify OpenCV filter transformation."""
        response = self.client.post(
            "/api/filter", json={"sampleName": "tiger.jpg", "filter": "clahe"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filter"], "clahe")
        self.assertIn("image_b64", data)


if __name__ == "__main__":
    unittest.main()
