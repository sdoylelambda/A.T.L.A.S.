import yaml
import threading
import sys
import asyncio
import os

# load config first so name is available before QApplication
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# must be set before QApplication is created
app_id = config.get("personalize", {}).get("ai_assistant_name", "atlas").lower()
os.environ["QT_WAYLAND_APPID"] = f"{app_id}.assistant"

from PyQt5.QtWidgets import QApplication
from modules.face import FaceController
from modules.window_controller import WindowController
from modules.observer import Observer


def run_async(face, config):
    async def main():
        window_controller = WindowController()
        observer = Observer(face, window_controller, config)
        loop = asyncio.get_running_loop()

        # register API key callback
        from config.api_keys import set_key_request_callback
        set_key_request_callback(observer._request_key_via_gui)

        face.on_cancel = observer._cancel_all
        face.on_mute = lambda muted: setattr(observer.ears, 'paused', muted)

        def handle_text_input(text: str):
            if hasattr(observer, '_waiting_for_key') and observer._waiting_for_key:
                observer._provide_key(text)
            else:
                asyncio.run_coroutine_threadsafe(
                    observer.handle_brain_command(text), loop
                )

        face.on_command = handle_text_input

        await observer.listen_and_respond()

    asyncio.run(main())


if __name__ == "__main__":
    qt_app = QApplication(sys.argv)
    qt_app.setStyle("Fusion")

    face = FaceController(config)
    face.show()

    thread = threading.Thread(target=run_async, args=(face, config), daemon=True)
    thread.start()

    sys.exit(qt_app.exec_())
