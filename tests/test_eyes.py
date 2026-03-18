import pytest
import os
import base64
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from modules.eyes import Eyes


@pytest.fixture
def config():
    return {
        "vision": {
            "enabled": True,
            "camera_index": 0,
            "model": "llava",
            "max_storage_mb": 100,
            "resolution_width": 3840,
            "resolution_height": 2160
        }
    }


@pytest.fixture
def eyes(config):
    e = Eyes(config)
    return e


@pytest.fixture
def mock_frame():
    """Create a fake 720p BGR frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ─── capture_frame ────────────────────────────────────────────────────────────

class TestCaptureFrame:

    @patch("modules.eyes.cv2.VideoCapture")
    def test_returns_base64_string(self, mock_cap_class, eyes, mock_frame):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap_class.return_value = mock_cap

        result = eyes.capture_frame()
        assert result is not None
        # verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    @patch("modules.eyes.cv2.VideoCapture")
    def test_returns_none_if_camera_not_found(self, mock_cap_class, eyes):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap

        result = eyes.capture_frame()
        assert result is None

    @patch("modules.eyes.cv2.VideoCapture")
    def test_returns_none_if_read_fails(self, mock_cap_class, eyes):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cap_class.return_value = mock_cap

        result = eyes.capture_frame()
        assert result is None

    @patch("modules.eyes.cv2.VideoCapture")
    def test_sets_resolution(self, mock_cap_class, eyes, mock_frame):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap_class.return_value = mock_cap

        eyes.capture_frame()

        mock_cap.set.assert_any_call(3, 3840)
        mock_cap.set.assert_any_call(4, 2160)

    @patch("modules.eyes.cv2.VideoCapture")
    def test_releases_camera_after_capture(self, mock_cap_class, eyes, mock_frame):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap_class.return_value = mock_cap

        eyes.capture_frame()
        mock_cap.release.assert_called_once()

    @patch("modules.eyes.cv2.VideoCapture")
    def test_saves_frame_to_tmp_eyes(self, mock_cap_class, eyes, mock_frame, tmp_path):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap_class.return_value = mock_cap

        with patch("modules.eyes.cv2.imwrite") as mock_write:
            eyes.capture_frame()
            assert mock_write.called
            save_path = mock_write.call_args[0][0]
            assert "/tmp/eyes" in save_path


# ─── analyze ──────────────────────────────────────────────────────────────────

class TestAnalyze:

    def test_returns_string_response(self, eyes):
        eyes.capture_frame = MagicMock(return_value="fakebase64==")
        with patch("modules.eyes.ollama.chat") as mock_chat:
            mock_chat.return_value = {
                "message": {"content": "I see a desk with a laptop, sir."}
            }
            result = eyes.analyze("what do you see?")
            assert isinstance(result, str)
            assert "desk" in result.lower()

    def test_returns_error_message_if_no_frame(self, eyes):
        eyes.capture_frame = MagicMock(return_value=None)
        result = eyes.analyze("what do you see?")
        assert "camera" in result.lower()

    def test_returns_error_message_on_exception(self, eyes):
        eyes.capture_frame = MagicMock(return_value="fakebase64==")
        with patch("modules.eyes.ollama.chat", side_effect=Exception("model error")):
            result = eyes.analyze("what do you see?")
            assert "trouble" in result.lower()

    def test_passes_prompt_to_llava(self, eyes):
        eyes.capture_frame = MagicMock(return_value="fakebase64==")
        with patch("modules.eyes.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "response"}}
            eyes.analyze("custom prompt here")
            call_args = mock_chat.call_args
            messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][1]
            assert any("custom prompt here" in str(m) for m in messages)


# ─── analysis methods ─────────────────────────────────────────────────────────

class TestAnalysisMethods:

    def setup_method(self):
        """Mock analyze for all method tests."""
        pass

    def test_what_do_you_see(self, eyes):
        eyes.analyze = MagicMock(return_value="I see a room, sir.")
        result = eyes.what_do_you_see()
        assert result == "I see a room, sir."
        eyes.analyze.assert_called_once()

    def test_read_document(self, eyes):
        eyes.analyze = MagicMock(return_value="The text reads: Hello World, sir.")
        result = eyes.read_document()
        assert "hello world" in result.lower()

    def test_identify_object(self, eyes):
        eyes.analyze = MagicMock(return_value="I see a coffee mug, sir.")
        result = eyes.identify_object()
        assert "mug" in result.lower()

    def test_count_people(self, eyes):
        eyes.analyze = MagicMock(return_value="There is one person visible, sir.")
        result = eyes.count_people()
        assert "one" in result.lower()

    def test_identify_color(self, eyes):
        eyes.analyze = MagicMock(return_value="The main colors are blue and white, sir.")
        result = eyes.identify_color()
        assert "blue" in result.lower()

    def test_describe_activity(self, eyes):
        eyes.analyze = MagicMock(return_value="The person appears to be typing, sir.")
        result = eyes.describe_activity()
        assert "typing" in result.lower()

    def test_describe_scene(self, eyes):
        eyes.analyze = MagicMock(return_value="I see a desk with a monitor, sir.")
        result = eyes.describe_scene()
        assert "desk" in result.lower()

    def test_analyze_with_question(self, eyes):
        eyes.analyze = MagicMock(return_value="Yes I can see a phone, sir.")
        result = eyes.analyze_with_question("do you see a phone?")
        assert "phone" in result.lower()
        # verify question was passed through
        call_prompt = eyes.analyze.call_args[0][0]
        assert "do you see a phone" in call_prompt.lower()


# ─── check_storage ────────────────────────────────────────────────────────────

class TestCheckStorage:

    def test_returns_none_if_under_limit(self, eyes, tmp_path):
        with patch("modules.eyes.os.path.exists", return_value=True), \
             patch("modules.eyes.os.listdir", return_value=["img1.jpg"]), \
             patch("modules.eyes.os.path.getsize", return_value=1024 * 1024):  # 1MB
            result = eyes.check_storage()
            assert result is None

    def test_returns_warning_if_over_limit(self, eyes):
        # 200MB of files, limit is 100MB
        with patch("modules.eyes.os.path.exists", return_value=True), \
             patch("modules.eyes.os.listdir", return_value=["img1.jpg", "img2.jpg"]), \
             patch("modules.eyes.os.path.isfile", return_value=True), \
             patch("modules.eyes.os.path.getsize", return_value=100 * 1024 * 1024):
            result = eyes.check_storage()
            assert result is not None
            assert "warning" in result.lower()
            assert "sir" in result.lower()

    def test_returns_none_if_dir_not_exists(self, eyes):
        with patch("modules.eyes.os.path.exists", return_value=False):
            result = eyes.check_storage()
            assert result is None

    def test_warning_includes_size(self, eyes):
        with patch("modules.eyes.os.path.exists", return_value=True), \
             patch("modules.eyes.os.listdir", return_value=["img1.jpg"]), \
             patch("modules.eyes.os.path.isfile", return_value=True), \
             patch("modules.eyes.os.path.getsize", return_value=150 * 1024 * 1024):
            result = eyes.check_storage()
            assert result is not None
            assert "150" in result