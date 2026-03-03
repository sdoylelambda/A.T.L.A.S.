import asyncio
import yaml
import threading
import sys
import asyncio
import os

# must be set before QApplication is created
os.environ["QT_WAYLAND_APPID"] = "jarvis.assistant"

from PyQt5.QtWidgets import QApplication
from modules.face import FaceController
from modules.window_controller import WindowController
from modules.observer import Observer
from modules.browser_controller import BrowserController


def run_async(face, config):
    async def main():
        window_controller = WindowController()
        observer = Observer(face, window_controller, config)
        # browser = BrowserController()
        # await browser.start()

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

    face = FaceController()
    face.show()

    thread = threading.Thread(target=run_async, args=(face, config), daemon=True)
    thread.start()

    sys.exit(qt_app.exec_())