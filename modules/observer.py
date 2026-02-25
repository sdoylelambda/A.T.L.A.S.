import time
import asyncio
from modules.ears import Ears
from modules.stt.hybrid_stt import HybridSTT
from modules.tts import TTSModule
from modules.app_launcher import AppLauncher
from modules.brain import Brain
from custom_exceptions import PermissionRequired, ModelUnavailable, PlanExecutionError
from modules.tool_executor import ToolExecutor
from modules.browser_controller import BrowserController


class Observer:
    def __init__(self, face_controller, window_controller, config):
        self.face = face_controller
        self.window_controller = window_controller
        self.config = config
        self.paused = False
        self.debug = True
        self._last_spoken = ""
        self._last_spoken_time = 0

        self.brain = Brain(config)
        self.ears = Ears()
        self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
        self.browser_controller = BrowserController()
        self.launcher = AppLauncher(window_controller, self.browser_controller)
        self.executor = ToolExecutor(self.launcher, self.browser_controller, self.brain)
        self.stt = HybridSTT(
            whisper_model="small",
            fw_model="small",
            use_gpu=config["system"].get("use_gpu", False),
        )

    async def listen_and_respond(self):
        self.face.set_state("thinking")
        # init ears — triggers mic selection and noise floor prints
        try:
            await asyncio.wait_for(self.ears.listen(), timeout=3.0)
        except asyncio.TimeoutError:
            pass
        await self.say("Hello sir, what can I do for you.")
        print("[Observer] Listening and responding...")
        # start auto calibration in background
        interval = self.config["audio"].get("calibration_interval", 30)
        asyncio.create_task(self.ears.auto_calibrate(interval=interval))

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

                # filter hallucinations
                words = text.split()
                unique = set(words)
                if len(unique) <= 2 and len(words) > 6:
                    print(f"[STT] Hallucination detected, skipping: {text[:50]}")
                    continue

                # filter useless single words that aren't commands
                known_short = [
                    # cancel/control
                    "yes", "no", "cancel", "stop", "pause",
                    "never mind", "forget it", "escape", "deselect",
                    # browser navigation
                    "zoom in", "zoom out", "zoom reset", "go down", "go up", "go back", "go forward",
                    "new tab", "close tab", "refresh", "reload", "new window"
                    "full screen", "fullscreen", "find", "search on page",
                    "scroll up", "scroll down", "next", "enter", "press enter",
                    "copy", "paste", "select", "click", "escape"
                    # app shortcuts
                    "save", "run", "clear",
                ]
                if len(words) <= 1 and not any(cmd in text for cmd in known_short):
                    print(f"[STT] Too short, skipping: {text}")
                    continue

                # filter echo of last spoken phrase
                if (self._last_spoken and
                        time.time() - self._last_spoken_time < 5.0 and
                        self._similarity(text, self._last_spoken) > 0.6):
                    print(f"[Observer] Echo detected, skipping: {text[:50]}")
                    continue

                print(f"[Heard]: {text}")

                if self.debug:
                    if self._last_spoken:
                        sim = self._similarity(text, self._last_spoken)
                        print(f"[Observer] Similarity: {sim:.2f} last='{self._last_spoken[:40]}'")
                        if time.time() - self._last_spoken_time < 5.0 and sim > 0.6:
                            print(f"[Observer] Echo detected, skipping")
                            continue

                # ❌ Cancel command
                if any(word in text for word in ["cancel", "stop", "never mind", "forget it"]):
                    self.mouth.stop()  # ← interrupt speech instantly
                    self.face.set_state("listening")
                    await self.say("Cancelled.")
                    continue

                # 🔑 Wake hot words
                if "jarvis" in text or "you there" in text:
                    if self.paused:
                        self.paused = False
                        self.face.set_state("thinking")
                        t1 = time.time()
                        await self.say("For you sir, always.")
                        if self.debug:
                            print(f"[Timing] TTS: {time.time() - t1:.2f}s")
                        self.face.set_state("listening")
                    continue

                # 😴 Pause command
                if any(phrase in text for phrase in ["take a break", "pause", "stop listening", "go to sleep"]):
                    self.paused = True
                    self.face.set_state("sleeping")
                    t1 = time.time()
                    await self.say("Going on a break.")
                    if self.debug:
                        print(f"[Timing] TTS: {time.time() - t1:.2f}s")
                    continue

                if self.paused:
                    continue

                # 🚀 Command handling
                self.face.set_state("thinking")
                try:
                    handled = await self.launcher.dispatch(text)
                except Exception as e:
                    print(f"[Launcher Error]: {e}")
                    handled = False

                # 🔊 TTS response
                t1 = time.time()
                if handled:
                    current_app = self.launcher.get_current_app()
                    if current_app and current_app != "browser":
                        self.window_controller.update_active_window(current_app)
                    await self.say(f"I have: {text}")
                else:
                    # fallthrough to brain instead of failing
                    await self.handle_brain_command(text)

                if self.debug:
                    print(f"[Timing] TTS: {time.time() - t1:.2f}s")

                self.face.set_state("listening")

            except Exception as e:
                print(f"[Observer Error]: {e}")
                self.face.set_state("error")

            await asyncio.sleep(0.01)

    async def handle_brain_command(self, command: str):
        try:
            plan = self.brain.process(command)
            summary = plan.get("summary", "")

            # clean summary — remove any JSON bleed
            if "{" in summary:
                summary = summary.split("{")[0].strip()

            if plan.get("steps"):
                if len(plan["steps"]) > 1:
                    await self.say(summary)
                results = await self.executor.execute_plan(plan)
                for result in results:
                    await self.say(result)
            else:
                await self.say(summary)

        except PermissionRequired as e:
            await self.say(
                f"This command needs to go to {e.model_key}. "
                f"Say yes to send it or no to cancel."
            )
            audio_bytes, duration = await self.ears.listen()
            response = self.stt.transcribe(audio_bytes, duration) if audio_bytes else ""
            response = response.lower().strip() if response else ""

            if any(word in response for word in ["yes", "yeah", "yep", "do it", "send", "sure", "build"]):
                plan = self.brain.process_with_permission(command, e.model_key)
                await self.say(plan["summary"])
            else:
                await self.say("Cancelled. I'll handle it locally instead.")
                plan = self.brain.create_plan(command)
                await self.say(plan["summary"])

        except PlanExecutionError as e:
            print(f"[ToolExecutor Error] step={e.step}, reason={e.reason}")
            await self.say(f"I ran into a problem, sir. {e.reason}")

        except ModelUnavailable as e:
            print(f"[Brain] Model unavailable: {e.model_key}")
            await self.say(
                f"The {e.model_key} model is currently disabled, sir. Handling locally instead."
            )
            plan = self.brain.create_plan(command)
            await self.say(plan["summary"])

        except Exception as e:
            print(f"[Brain Error]: {e}")
            self.face.set_state("error")
            await self.say("I ran into a problem with that one, sir.")

    def _similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        overlap = words_a & words_b
        return len(overlap) / max(len(words_a), len(words_b))

    async def say(self, text: str):
        """Speak and pause ears during playback to prevent echo."""
        if self.debug:
            print(f"[Observer] Pausing ears")
        self.ears.paused = True
        self._last_spoken = text.lower().strip()
        self._last_spoken_time = time.time()
        await self.mouth.speak(text)
        await asyncio.sleep(1.0)  # longer settle time
        # flush any audio captured during speech
        if self.ears.audio_stream:
            try:
                while self.ears.audio_stream.get_read_available() > 0:
                    self.ears.audio_stream.read(
                        self.ears.chunk_size,
                        exception_on_overflow=False
                    )
            except OSError:
                pass
        self.ears.paused = False
        if self.debug:
            print(f"[Observer] Ears resumed")


# import time
# import asyncio
# from modules.ears import Ears
# from modules.stt.hybrid_stt import HybridSTT
# from modules.tts import TTSModule
# from modules.app_launcher import AppLauncher
# from modules.brain import Brain
# from custom_exceptions import PermissionRequired
#
#
# class Observer:
#     def __init__(self, face_controller, window_controller, config):
#         self.face = face_controller
#         self.window_controller = window_controller
#         self.paused = False
#         self.debug = True
#
#         self.brain = Brain(config)
#         self.ears = Ears()
#         self.stt = HybridSTT(
#             whisper_model="small",
#             fw_model="small",
#             use_gpu=config["system"].get("use_gpu", False),
#         )
#         self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
#         self.launcher = AppLauncher(window_controller)
#
#     async def listen_and_respond(self):
#         self.face.set_state("thinking")
#         await asyncio.to_thread(self.mouth.speak, "Hello sir, what can I do for you.")
#         self.face.set_state("listening")
#         print("[Observer] Listening and responding...")
#
#         while True:
#             try:
#                 if self.paused:
#                     self.face.set_state("sleeping")
#                 else:
#                     self.face.set_state("listening")
#
#                 # 🎧 Listen
#                 audio_bytes, duration = await self.ears.listen()
#
#                 if not audio_bytes:
#                     await asyncio.sleep(0.05)
#                     continue
#
#                 # 🧠 STT
#                 t0 = time.time()
#                 text = self.stt.transcribe(audio_bytes, duration)
#                 print(f"[Timing] STT: {time.time() - t0:.2f}s")
#
#                 if not text:
#                     continue
#
#                 text = text.lower().strip()
#                 print(f"[Heard]: {text}")
#
#                 # 🔑 Wake hot words
#                 if "jarvis" in text or "you there" in text:
#                     if self.paused:
#                         self.paused = False
#                         self.face.set_state("thinking")
#                         t1 = time.time()
#                         await asyncio.to_thread(self.mouth.speak, "For you sir, always.")
#                         if self.debug:
#                             print(f"[Timing] TTS: {time.time() - t1:.2f}s")
#                         self.face.set_state("listening")
#                     continue
#
#                 # 😴 Pause command
#                 if "take a break" in text:
#                     self.paused = True
#                     self.face.set_state("sleeping")
#                     t1 = time.time()
#                     await asyncio.to_thread(self.mouth.speak, "Going on a break.")
#                     if self.debug:
#                         print(f"[Timing] TTS: {time.time() - t1:.2f}s")
#                     continue
#
#                 if self.paused:
#                     continue
#
#                 # 🚀 Command handling
#                 self.face.set_state("thinking")
#                 try:
#                     handled = await self.launcher.handle_command(text)
#                 except Exception as e:
#                     print(f"[Launcher Error]: {e}")
#                     handled = False
#
#                 # 🔊 TTS response
#                 t1 = time.time()
#                 if handled:
#                     current_app = self.launcher.get_current_app()
#                     if current_app and current_app != "browser":
#                         self.window_controller.update_active_window(current_app)
#                     await asyncio.to_thread(self.mouth.speak, f"I have: {text}")
#                 else:
#                     # fallthrough to brain instead of failing
#                     await self.handle_command(text)
#                 if self.debug:
#                     print(f"[Timing] TTS: {time.time() - t1:.2f}s")
#
#                 self.face.set_state("listening")
#
#             except Exception as e:
#                 print(f"[Observer Error]: {e}")
#                 self.face.set_state("error")
#
#             await asyncio.sleep(0.01)
#
#     async def handle_command(self, command: str):
#         try:
#             plan = self.brain.process(command)
#             await asyncio.to_thread(self.mouth.speak, plan["summary"])
#
#         except PermissionRequired as e:
#             await asyncio.to_thread(self.mouth.speak,
#                                     f"This command needs to go to {e.model_key}. "
#                                     f"Say yes to send it or no to cancel."
#                                     )
#
#             # listen AND transcribe
#             audio_bytes, duration = await self.ears.listen()
#             response = self.stt.transcribe(audio_bytes, duration) if audio_bytes else ""
#             response = response.lower().strip() if response else ""
#
#             if any(word in response for word in ["yes", "yeah", "yep", "do it", "send it", "sure"]):
#                 plan = self.brain.process_with_permission(command, e.model_key)
#                 await asyncio.to_thread(self.mouth.speak, plan["summary"])
#             else:
#                 await asyncio.to_thread(self.mouth.speak, "Cancelled. I'll handle it locally instead.")
#                 plan = self.brain.create_plan(command)
#                 await asyncio.to_thread(self.mouth.speak, plan["summary"])
#
#         except Exception as e:
#             print(f"[Brain Error]: {e}")
#             self.face.set_state("error")
#             await asyncio.to_thread(self.mouth.speak, "I ran into a problem with that one.")