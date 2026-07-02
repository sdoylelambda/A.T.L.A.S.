import cv2
import ollama
import glob
import os
import base64
import asyncio

from datetime import datetime
from modules.utils import timer


class Eyes:
    def __init__(self, config: dict):
        self.config = config.get("vision", {})
        self.response_name = config["personalize"].get("response_name", "")
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
            return f"Warning {self.response_name}, vision storage is {total_mb:.0f} megabytes. You may want to clear the eyes folder."

        print(f"[Vision] Storage: {total_mb:.1f}MB / {max_mb}MB")
        return None

    def analyze(self, prompt: str = "Describe what you see in this image concisely.") -> str:
        """Capture frame and analyze with LLaVA."""
        print("[Eyes] Capturing frame...")
        image_b64 = self.capture_frame()

        if not image_b64:
            return f"I couldn't access the camera, {self.response_name}."

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
            return f"I had trouble analyzing the image, {self.response_name}."

    def what_do_you_see(self) -> str:
        return self.analyze(
            "You are Atlas, a British AI assistant. Describe what you see concisely in 1-2 sentences. Address the user"
            f" as {self.response_name}."
        )

    def read_text(self) -> str:
        return self.analyze(
            "Read and transcribe any text visible in this image. If no text is visible, say so briefly."
        )

    def identify_object(self) -> str:
        return self.analyze(
            "What is the main object in this image? Describe it briefly in one sentence. Address the user as "
            f"{self.response_name}."
        )

    def is_someone_present(self) -> str:
        return self.analyze(
            "Is there a person visible in this image? Answer yes or no and describe briefly. Address the user as "
            f"{self.response_name}."
        )

    def describe_scene(self) -> str:
        return self.analyze(
            "Describe the scene in detail — what's in the room, on the desk, "
            f"in the background. Be thorough but concise. Address the user as {self.response_name}."
        )

    def count_people(self) -> str:
        return self.analyze(
            "How many people are visible in this image? Describe them briefly. "
            f"If none, say so. Address the user as {self.response_name}."
        )

    def read_document(self) -> str:
        return self.analyze(
            "Read and transcribe all text visible in this image as accurately as possible. "
            f"If no text is visible, say so. Address the user as {self.response_name}."
        )

    def identify_color(self) -> str:
        return self.analyze(
            "What are the main colors visible in this image? "
            f"Describe briefly. Address the user as {self.response_name}."
        )

    def describe_activity(self) -> str:
        return self.analyze(
            "What activity or action is happening in this image? "
            f"Describe what the person appears to be doing. Address the user as {self.response_name}."
        )

    def analyze_with_question(self, question: str) -> str:
        return self.analyze(
            f"Answer this question about what you see: '{question}'. "
            f"Be brief and direct. Address the user as {self.response_name}."
        )

    def capture_screen_sync(self) -> str | None:
        """Blocking, safe, cross-session screenshot."""
        import tempfile
        import os
        import base64
        import subprocess
        import re
        from datetime import datetime

        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        screenshot_path = None

        try:
            if session == "wayland":
                save_dir = tempfile.gettempdir()
                timestamp = datetime.now().strftime("%H%M%S")
                screenshot_path = os.path.join(save_dir, f"atlas_screen_{timestamp}.png")

                call = subprocess.run(
                    [
                        "gdbus", "call",
                        "--session",
                        "--dest", "org.freedesktop.portal.Desktop",
                        "--object-path", "/org/freedesktop/portal/desktop",
                        "--method", "org.freedesktop.portal.Screenshot.Screenshot",
                        "", "{'interactive': <false>}"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if call.returncode != 0:
                    if self.debug:
                        print("[Vision] Portal stderr:", call.stderr)
                    return None

                request_match = re.search(r"objectpath '([^']+)'", call.stdout)
                if not request_match:
                    return None

                request_path = request_match.group(1)

                monitor = subprocess.Popen(
                    ["gdbus", "monitor", "--session", "--object-path", request_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True
                )

                screenshot_uri = None
                for line in monitor.stdout:
                    if "Response" in line:
                        uri_match = re.search(r"file://([^']+)", line)
                        if uri_match:
                            screenshot_uri = uri_match.group(1)
                            break
                monitor.terminate()

                if not screenshot_uri:
                    return None

                screenshot_path = screenshot_uri

            else:
                # X11
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    screenshot_path = tmp.name

                for cmd in (["scrot", screenshot_path], ["import", "-window", "root", screenshot_path]):
                    try:
                        result = subprocess.run(cmd, timeout=10)
                        if result.returncode == 0 and os.path.exists(screenshot_path):
                            break
                    except FileNotFoundError:
                        continue

                if not os.path.exists(screenshot_path):
                    return None

            with open(screenshot_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            try:
                os.remove(screenshot_path)
            except Exception:
                pass

            return encoded

        except Exception as e:
            if self.debug:
                print("[Vision] Screen capture error:", e)
            return None

    async def analyze_screen(self, prompt: str = None) -> str:
        """Async analyze screen without blocking."""
        import asyncio

        # Run capture in thread
        image_b64 = await asyncio.to_thread(self.capture_screen_sync)

        if not image_b64:
            return "Allow pop up permission request and try again."

        final_prompt = prompt or (
            "Describe what's on this screen concisely. "
            "What application is open and what is it showing? "
            f"Address the user as {self.response_name}."
        )

        # Run analyzer in thread if heavy
        return await asyncio.to_thread(self._analyze_image, image_b64, final_prompt)

    async def get_latest_screenshot(self) -> str | None:
        """Find and load the most recent screenshot."""
        search_paths = [
            os.path.expanduser("~/Pictures/*.png"),
            os.path.expanduser("~/Pictures/Screenshots/*.jpg"),
            os.path.expanduser("~/Screenshots/*.png"),
            os.path.expanduser("~/Desktop/*.png"),
            "/tmp/eyes/atlas_screen_*.png"
        ]

        all_files = []
        for pattern in search_paths:
            all_files.extend(glob.glob(pattern))

        if not all_files:
            print("[Vision] No screenshots found")
            return None

        latest = max(all_files, key=os.path.getmtime)
        print(f"[Vision] Latest screenshot: {latest}")

        with open(latest, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    async def analyze_screenshot(self, prompt: str = None) -> str:  # ← add async
        image_b64 = await self.get_latest_screenshot()  # ← await directly, already async
        if not image_b64:
            return f"No screenshots found, {self.response_name}. Try taking one first with Print Screen."
        return await asyncio.to_thread(
            self._analyze_image,
            image_b64,
            prompt or f"Describe this screenshot concisely. What does it show? Address the user as {self.response_name}."
        )
    def _analyze_image(self, image_b64: str, prompt: str) -> str:
        """Shared image analysis — used by webcam, screen capture, and screenshots."""
        print(f"[Vision] Analyzing image with {self.model}...")
        try:
            with timer("LLaVA", self.debug):
                response = ollama.chat(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64]
                    }]
                )
            result = response["message"]["content"].strip()
            print(f"[Vision] Result: {result[:100]}")
            return result
        except Exception as e:
            print(f"[Vision] Error: {e}")
            return f"I had trouble analyzing the image, {self.response_name}."
