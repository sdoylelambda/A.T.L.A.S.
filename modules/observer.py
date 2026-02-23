import time
import asyncio
from modules.ears import Ears
from modules.stt.hybrid_stt import HybridSTT
from modules.tts import TTSModule
from modules.app_launcher import AppLauncher


class Observer:
    def __init__(self, face_controller, window_controller, config):
        self.face = face_controller
        self.window_controller = window_controller
        self.paused = False
        self.debug = True

        self.ears = Ears()
        self.stt = HybridSTT(
            whisper_model="small",
            fw_model="small",
            use_gpu=config["system"].get("use_gpu", False),
        )
        self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
        self.launcher = AppLauncher(window_controller)

    async def listen_and_respond(self):
        self.face.set_state("thinking")
        await asyncio.to_thread(self.mouth.speak, "Hello sir, what can I do for you.")
        self.face.set_state("listening")
        print("[Observer] Listening and responding...")

        while True:
            try:
                if self.paused:
                    self.face.set_state("sleeping")
                else:
                    self.face.set_state("listening")

                # 🎧 Listen
                audio_bytes, duration = await self.ears.listen()

                if not audio_bytes:
                    await asyncio.sleep(0.05)
                    continue

                # 🧠 STT
                t0 = time.time()
                text = self.stt.transcribe(audio_bytes, duration)
                print(f"[Timing] STT: {time.time() - t0:.2f}s")

                if not text:
                    continue

                text = text.lower().strip()
                print(f"[Heard]: {text}")

                # 🔑 Wake hot words
                if "jarvis" in text or "you there" in text:
                    if self.paused:
                        self.paused = False
                        self.face.set_state("thinking")
                        t1 = time.time()
                        await asyncio.to_thread(self.mouth.speak, "I'm back online.")
                        print(f"[Timing] TTS: {time.time() - t1:.2f}s")
                        self.face.set_state("listening")
                    continue

                # 😴 Pause command
                if "take a break" in text:
                    self.paused = True
                    self.face.set_state("sleeping")
                    t1 = time.time()
                    await asyncio.to_thread(self.mouth.speak, "Going on a break.")
                    print(f"[Timing] TTS: {time.time() - t1:.2f}s")
                    continue

                if self.paused:
                    continue

                # 🚀 Command handling
                self.face.set_state("thinking")
                try:
                    handled = await self.launcher.handle_command(text)
                except Exception as e:
                    print(f"[Launcher Error]: {e}")
                    handled = False

                # 🔊 TTS response
                t1 = time.time()
                if handled:
                    current_app = self.launcher.get_current_app()
                    if current_app and current_app != "browser":
                        self.window_controller.update_active_window(current_app)
                    await asyncio.to_thread(self.mouth.speak, f"I have: {text}")
                else:
                    await asyncio.to_thread(self.mouth.speak, "Command not recognized.")
                print(f"[Timing] TTS: {time.time() - t1:.2f}s")

                self.face.set_state("listening")

            except Exception as e:
                print(f"[Observer Error]: {e}")
                self.face.set_state("error")

            await asyncio.sleep(0.01)
