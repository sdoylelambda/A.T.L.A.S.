import yaml
import threading
import sys
import asyncio
import os
import argparse

from modules.window_controller import WindowController

# Android SSH — run without GUI
parser = argparse.ArgumentParser()
parser.add_argument('--no-gui', action='store_true', help='Run without GUI (SSH/headless mode)')
args = parser.parse_args()

# load config first so name is available before QApplication
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# For Termux SSH
async def run_text_only(config):
    from modules.observer import Observer
    from config.api_keys import set_key_request_callback
    from prompt_toolkit import PromptSession

    class MockFace:
        def __init__(self):
            self._current_state = None

        def set_state(self, s):
            if s == self._current_state:
                return
            self._current_state = s
            if s in ["error", "sleeping"]:
                print(f"[State] {s}")

        def set_caption(self, t):
            if t: print(f"[Atlas] {t}")

        def set_heard(self, t):
            print(f"[Heard] {t}")

        on_cancel = None
        on_mute = None
        on_command = None

    face = MockFace()
    window_controller = WindowController()
    observer = Observer(face, window_controller, config)

    set_key_request_callback(observer._request_key_via_gui)

    from prompt_toolkit.patch_stdout import patch_stdout

    async def text_input_loop():
        session = PromptSession()
        print("\n[Atlas] Text input ready — type commands below")
        print("[Atlas] Voice input also active via desktop mic\n")
        while True:
            try:
                with patch_stdout():
                    text = await session.prompt_async(">> ")
                text = text.strip()
                if not text:
                    continue
                if text.lower() in ["exit", "quit", "q"]:
                    print("[Atlas] Goodbye, sir.")
                    sys.exit(0)
                await observer.handle_brain_command(text)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n[Atlas] Goodbye, sir.")
                sys.exit(0)

    await asyncio.gather(
        observer.listen_and_respond(),
        text_input_loop()
    )

def run_async(face, config):
    async def main():
        from config.api_keys import set_key_request_callback
        from modules.observer import Observer

        window_controller = WindowController()
        observer = Observer(face, window_controller, config)
        loop = asyncio.get_running_loop()

        set_key_request_callback(observer._request_key_via_gui)

        face.on_cancel = observer._cancel_all
        face.on_mute = lambda muted: setattr(observer.ears, 'paused', muted)

        def handle_text_input(text: str):
            if hasattr(observer, '_waiting_for_key') and observer._waiting_for_key:
                observer._provide_key(text)
            else:
                asyncio.run_coroutine_threadsafe(
                    observer._text_command_queue.put(text.lower().strip()),
                    loop
                )

        face.on_command = handle_text_input

        await observer.listen_and_respond()

    asyncio.run(main())


if __name__ == "__main__":
    if args.no_gui:
        asyncio.run(run_text_only(config))
    else:
        from PyQt5.QtWidgets import QApplication
        from modules.face import FaceController

        app_id = config.get("personalize", {}).get("ai_assistant_name", "atlas").lower()
        os.environ["QT_WAYLAND_APPID"] = f"{app_id}.assistant"

        qt_app = QApplication(sys.argv)
        qt_app.setStyle("Fusion")

        face = FaceController(config)
        face.show()

        thread = threading.Thread(target=run_async, args=(face, config), daemon=True)
        thread.start()

        sys.exit(qt_app.exec_())
