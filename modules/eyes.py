import cv2
import base64
import ollama
import os
from datetime import datetime

from modules.utils import timer


class Eyes:
    def __init__(self, config: dict):
        self.config = config.get("vision", {})
        self.camera_index = self.config.get("camera_index", 0)
        self.model = self.config.get("model", "llava")
        self.debug = True

    def capture_frame(self) -> str | None:
        """Capture a single frame from webcam and return as base64."""
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("[Vision] Camera not found")
            return None

        # set higher resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        if self.debug:
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"[Vision] Camera resolution: {actual_w}x{actual_h}")

        # let camera warm up — first few frames are often dark
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("[Vision] Failed to capture frame")
            return None

        save_dir = "/tmp/eyes"
        os.makedirs(save_dir, exist_ok=True)  # creates folder if it doesn't exist
        timestamp = datetime.now().strftime("%H%M%S")
        save_path = f"{save_dir}/atlas_vision_{timestamp}.jpg"
        if self.debug:
            print(f"[Vision] Frame saved to {save_path}")

        # increase JPEG quality to max
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return base64.b64encode(buffer).decode('utf-8')  # warning is PyCharm's static analysis being overly strict

    def check_storage(self) -> str | None:
        """Warn if /tmp/eyes exceeds configured size limit."""
        save_dir = "/tmp/eyes"
        if not os.path.exists(save_dir):
            return None

        max_mb = self.config.get("max_storage_mb", 100)
        total_bytes = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in os.listdir(save_dir)
            if os.path.isfile(os.path.join(save_dir, f))
        )
        total_mb = total_bytes / (1024 * 1024)

        if total_mb > max_mb:
            print(f"[Vision] ⚠️ {total_mb:.1f}MB exceeds {max_mb}MB limit")
            return f"Warning sir, vision storage is {total_mb:.0f} megabytes. You may want to clear the eyes folder."

        print(f"[Vision] Storage: {total_mb:.1f}MB / {max_mb}MB")
        return None

    def analyze(self, prompt: str = "Describe what you see in this image concisely.") -> str:
        """Capture frame and analyze with LLaVA."""
        print("[Eyes] Capturing frame...")
        image_b64 = self.capture_frame()

        if not image_b64:
            return "I couldn't access the camera, sir."

        print(f"[Eyes] Analyzing with {self.model}...")
        try:
            with timer("LLaVA", True):
                response = ollama.chat(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64]
                    }]
                )
            result = response["message"]["content"].strip()
            print(f"[Eyes] Result: {result[:100]}")
            return result

        except Exception as e:
            print(f"[Vision] Error: {e}")
            return "I had trouble analyzing the image, sir."

    def what_do_you_see(self) -> str:
        return self.analyze(
            "You are Jarvis, a British AI assistant. Describe what you see concisely in 1-2 sentences. Address the user as sir."
        )

    def read_text(self) -> str:
        return self.analyze(
            "Read and transcribe any text visible in this image. If no text is visible, say so briefly."
        )

    def identify_object(self) -> str:
        return self.analyze(
            "What is the main object in this image? Describe it briefly in one sentence. Address the user as sir."
        )

    def is_someone_present(self) -> str:
        return self.analyze(
            "Is there a person visible in this image? Answer yes or no and describe briefly. Address the user as sir."
        )

    def describe_scene(self) -> str:
        return self.analyze(
            "Describe the scene in detail — what's in the room, on the desk, "
            "in the background. Be thorough but concise. Address the user as sir."
        )

    def count_people(self) -> str:
        return self.analyze(
            "How many people are visible in this image? Describe them briefly. "
            "If none, say so. Address the user as sir."
        )

    def read_document(self) -> str:
        return self.analyze(
            "Read and transcribe all text visible in this image as accurately as possible. "
            "If no text is visible, say so. Address the user as sir."
        )

    def identify_color(self) -> str:
        return self.analyze(
            "What are the main colors visible in this image? "
            "Describe briefly. Address the user as sir."
        )

    def describe_activity(self) -> str:
        return self.analyze(
            "What activity or action is happening in this image? "
            "Describe what the person appears to be doing. Address the user as sir."
        )

    def analyze_with_question(self, question: str) -> str:
        return self.analyze(
            f"Answer this question about what you see: '{question}'. "
            f"Be brief and direct. Address the user as sir."
        )
