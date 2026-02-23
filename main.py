import asyncio
import yaml
import threading
from vispy import app
from modules.face import FaceController
from modules.window_controller import WindowController
from modules.observer import Observer


def run_async(face, config):
    async def main():
        window_controller = WindowController()
        observer = Observer(face, window_controller, config)
        await observer.listen_and_respond()

    asyncio.run(main())

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    face = FaceController()  # VisPy canvas created on main thread

    # Async logic runs in background thread
    thread = threading.Thread(target=run_async, args=(face, config), daemon=True)
    thread.start()

    # VisPy owns the main thread
    app.run()

