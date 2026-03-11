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

        face.on_cancel = observer._cancel_all
        face.on_mute = lambda muted: setattr(observer.ears, 'paused', muted)
        face.on_command = lambda text: asyncio.run_coroutine_threadsafe(
            observer.handle_brain_command(text),
            loop
        )

        await observer.listen_and_respond()

    asyncio.run(main())


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    qt_app = QApplication(sys.argv)
    qt_app.setStyle("Fusion")

    face = FaceController(config)
    face.show()

    thread = threading.Thread(target=run_async, args=(face, config), daemon=True)
    thread.start()

    sys.exit(qt_app.exec_())
