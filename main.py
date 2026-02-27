import asyncio
import yaml
import threading
from PyQt5.QtWidgets import QApplication
import sys
from modules.face import FaceController
from modules.window_controller import WindowController
from modules.observer import Observer


def run_async(face, config):
    async def main():
        window_controller = WindowController()
        observer = Observer(face, window_controller, config)

        # wire up GUI callbacks
        face.on_cancel = observer._cancel_all
        face.on_mute = lambda muted: setattr(observer.ears, 'paused', muted)
        face.on_command = lambda text: asyncio.create_task(
            observer.handle_brain_command(text)
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