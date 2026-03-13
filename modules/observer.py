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
from modules.utils import timer


class Observer:
    def __init__(self, face_controller, window_controller, config):
        self.debug = True
        self.face = face_controller
        self.window_controller = window_controller
        self.config = config
        self.paused = False
        self.cancelled = False
        self._processing = False
        self._last_spoken = ""
        self._last_spoken_time = 0
        self._finishing = False

        self.brain = Brain(config)
        self.ears = Ears()  # pass config
        self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
        self.browser_controller = BrowserController(config)
        self.launcher = AppLauncher(window_controller, self.browser_controller)
        self.executor = ToolExecutor(self.launcher, self.browser_controller, self.brain)
        self.stt = HybridSTT(
            whisper_model=config["stt"].get("whisper_model", "small"),
            fw_model=config["stt"].get("fw_model", "small"),
            use_gpu=config["system"].get("use_gpu", False),
        )

    async def listen_and_respond(self):
        self.face.set_state("thinking")
        # init ears AND calibrate before greeting
        try:
            await asyncio.wait_for(self.ears.listen(), timeout=6.0)  # longer to allow calibration
        except asyncio.TimeoutError:
            pass

        # now thresholds are set — safe to start
        print("[Observer] Listening and responding...")
        asyncio.create_task(self.ears.auto_calibrate(interval=30))
        await self.say("Hello sir, what can I do for you.")
        self.face.set_state("listening")

        while True:
            try:
                if self.paused:
                    self.face.set_state("sleeping")
                elif not self._processing and not self._finishing:
                    self.face.set_state("listening")

                # 🎧 Listen
                audio_bytes, duration = await self.ears.listen()

                if not audio_bytes:
                    await asyncio.sleep(0.05)
                    continue

                # 🧠 STT
                with timer("STT", self.debug):
                    text = self.stt.transcribe(audio_bytes, duration)

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
                    "first", "second", "third", "forth", "fifth", "break"
                    # browser navigation
                    "zoom in", "zoom out", "zoom reset", "go down", "go up", "go back", "go forward",
                    "new tab", "close tab", "refresh", "reload", "new window"
                    "full screen", "fullscreen", "find", "search on page",
                    "scroll up", "scroll down", "next", "enter", "press enter",
                    "copy", "paste", "select", "click", "escape",
                    # app shortcuts
                    "save", "run", "clear",
                    # build commands
                    "yeah", "yep", "do",  "it", "proceed",
                    "sure", "go", "ahead", "affirmative", "correct",
                    "build", "sounds", "good"
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
                self.face.set_heard(text)

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
                        with timer("TTS wake", self.debug):
                            await self.say("For you sir, always.")
                        self.face.set_state("listening")
                    continue

                # 😴 Pause command
                if any(phrase in text for phrase in ["take a break", "break", "stop listening", "go to sleep"]):
                    self.paused = True
                    self.face.set_state("sleeping")
                    with timer("TTS sleep", self.debug):
                        await self.say("Going on a break.")
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
                with timer("Total command", self.debug):
                    if handled:
                        current_app = self.launcher.get_current_app()
                        if current_app and current_app != "browser":
                            self.window_controller.update_active_window(current_app)
                        await self.say(f"I have: {text}")
                    else:
                        await self.handle_brain_command(text)

                if not self._processing:
                    self.face.set_state("listening")

            except Exception as e:
                print(f"[Observer Error]: {e}")
                self.face.set_state("error")
                await asyncio.sleep(2)  # show error face for 2 seconds before going back to listening

            await asyncio.sleep(0.01)

    async def handle_brain_command(self, command: str):
        """Main entry point for brain commands."""
        self._processing = True
        self.cancelled = False
        cancel_task = asyncio.create_task(self._listen_for_cancel())
        notice_task = asyncio.create_task(self._thinking_notice())

        try:
            try:
                self.face.set_state("thinking")
                self.face.set_caption("classifying...")  # ← add
                plan = await asyncio.to_thread(self.brain.process, command)
                notice_task.cancel()

                # validate route field
                if plan.get("route") not in ("local", "claude", "gemini", None):
                    print(f"[Brain] Invalid route '{plan.get('route')}', defaulting to local")
                    plan["route"] = "local"
                notice_task.cancel()
                # show what Brain decided
                if plan.get("steps"):
                    self.face.set_caption("planning...")  # ← Mistral made a plan
                else:
                    self.face.set_caption("")  # ← phi3 answered directly
                    await asyncio.sleep(0)  # flush any pending Qt signal updates

                if self.cancelled:
                    await self.say("Cancelled, sir.", next_state="listening")
                    return

                await self._execute_plan_with_confirm(plan, command)

            except PermissionRequired as e:
                notice_task.cancel()
                await self._handle_permission(e, command)

            except PlanExecutionError as e:
                notice_task.cancel()
                await self._handle_plan_error(e, plan)  # fix this or leave it/remove it?

            except ModelUnavailable as e:
                notice_task.cancel()
                await self._handle_model_unavailable(e, command)

            except Exception as e:
                notice_task.cancel()
                print(f"[Brain Error]: {e}")
                self.face.set_state("error")
                if len(command.split()) > 3:
                    await self.say("I ran into a problem with that one, sir.")

        finally:
            self._finishing = True
            self._processing = False
            cancel_task.cancel()
            notice_task.cancel()
            for task in [cancel_task, notice_task]:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.face.set_state("listening")
            self._finishing = False

    async def _thinking_notice(self):
        """Speak 'one moment' if Brain takes too long."""
        await asyncio.sleep(7.5)
        if not self.cancelled:
            await self.say("One moment please, sir.")

    async def _listen_for_cancel(self):
        """Always-on cancel listener running during brain execution."""
        while not self.cancelled:
            try:
                if self.ears.paused:
                    await asyncio.sleep(0.1)
                    continue
                audio_bytes, duration = await asyncio.wait_for(
                    self.ears.listen(), timeout=2.0
                )
                if audio_bytes:
                    text = self.stt.transcribe(audio_bytes, duration).lower().strip()
                    if self.debug:
                        print(f"[STT Raw] '{text}'")  # shows exactly what STT heard before any filtering
                    if any(w in text for w in ["cancel", "stop", "never mind", "forget it"]):
                        print("[Observer] Cancel detected during execution!")
                        self.cancelled = True
                        self.mouth.stop()
                        return
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue

    async def _execute_plan_with_confirm(self, plan: dict, command: str):
        """Speak summary, confirm with user, then execute."""
        summary = plan.get("summary", "")
        if "{" in summary:
            summary = summary.split("{")[0].strip()

        if not plan.get("steps"):
            await self.say(summary, next_state="listening")
            return

        await self.say(summary)
        await self.say("Shall I proceed, sir?")

        self.face.set_caption("waiting for confirmation...")

        audio_bytes, duration = await self.ears.listen()
        response = self.stt.transcribe(audio_bytes, duration).lower().strip() if audio_bytes else ""

        confirmed = any(w in response for w in [
            "yes", "yeah", "yep", "do", "it", "proceed",
            "sure", "ahead", "affirmative", "correct",
            "build", "sounds", "good"
        ])

        await asyncio.sleep(3.5)  # adjust wait time before cancel here (move to config?)

        if not confirmed:
            self.face.set_caption("")
            await self.say("Understood, sir. Cancelled.", next_state="listening")
            return

        self.face.set_caption("")
        await self.say("Building it now, sir.")

        if self.cancelled:
            await self.say("Cancelled, sir.", next_state="listening")
            return

        # show writing code if any step is generate_code
        steps = plan.get("steps", [])
        if any(s.get("action") == "generate_code" for s in steps):
            self.face.set_caption("writing code...")  # ← add this

        results = await self.executor.execute_plan(
            plan,
            cancelled=lambda: self.cancelled,
            on_step=lambda i, total, action: self.face.set_caption(
                f"step {i} of {total}: {action}..."
            )
        )

        self.face.set_caption("")

        if self.cancelled:
            await self.say("Cancelled, sir.")
            return

        if results:
            await self.say(results[-1], next_state="listening")

    async def _handle_permission(self, e: PermissionRequired, command: str):
        """Handle API permission requests."""
        await self.say(
            f"This command needs to go to {e.model_key}. "
            f"Say yes to send it or no to cancel."
        )
        audio_bytes, duration = await self.ears.listen()
        response = self.stt.transcribe(audio_bytes, duration).lower().strip() if audio_bytes else ""

        if any(w in response for w in ["yes", "yeah", "yep", "do it", "send", "sure"]):
            # temporarily disable permission check
            original = self.brain.api_models[e.model_key].get("ask_permission")
            self.brain.api_models[e.model_key]["ask_permission"] = False
            try:
                response = self.brain.query(command, model_key=e.model_key)
                await self.say(response, next_state="listening")
            finally:
                self.brain.api_models[e.model_key]["ask_permission"] = original
        else:
            await self.say("Cancelled. Handling locally instead, sir.")
            await self._run_local_plan(command)

    async def _handle_plan_error(self, e: PlanExecutionError, plan: dict):
        """Handle plan execution errors including overwrite confirmation."""
        print(f"[ToolExecutor Error] step={e.step}, reason={e.reason}")

        if "already exists" in e.reason:
            await self.say(e.reason)
            audio_bytes, duration = await self.ears.listen()
            if audio_bytes:
                response = self.stt.transcribe(audio_bytes, duration).lower().strip()
                if any(w in response for w in [
                    "yes", "overwrite", "over write", "replace", "do it", "sure"
                ]):
                    for step in plan["steps"]:
                        step["params"]["overwrite"] = True
                    results = await self.executor.execute_plan(
                        plan, cancelled=lambda: self.cancelled
                    )
                    if results:
                        await self.say(results[-1])
                else:
                    await self.say("Leaving the existing file untouched, sir.")
        else:
            await self.say(f"I ran into a problem, sir. {e.reason}")

    async def _handle_model_unavailable(self, e: ModelUnavailable, command: str):
        """Handle unavailable model by falling back to local."""
        print(f"[Brain] Model unavailable: {e.model_key}")
        await self.say(
            f"The {e.model_key} model is unavailable, sir. Handling locally instead."
        )
        await self._run_local_plan(command)

    async def _run_local_plan(self, command: str):
        """Run a plan locally without API models."""
        plan = self.brain.create_plan(command)
        summary = plan.get("summary", "")
        if "{" in summary:
            summary = summary.split("{")[0].strip()

        if plan.get("steps"):
            await self.say(summary)
            await self.say("Shall I proceed, sir?")

            audio_bytes, duration = await self.ears.listen()
            response = self.stt.transcribe(audio_bytes, duration).lower().strip() if audio_bytes else ""

            confirmed = any(w in response for w in [
                "yes", "yeah", "yep", "do it", "proceed",
                "sure", "go ahead", "affirmative", "correct", "build"
            ])

            if not confirmed:
                await self.say("Understood, sir. Cancelled.")
                return

            results = await self.executor.execute_plan(
                plan, cancelled=lambda: self.cancelled
            )
            if results:
                await self.say(results[-1])
        else:
            await self.say(summary)

    def _similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        overlap = words_a & words_b
        return len(overlap) / max(len(words_a), len(words_b))

    async def say(self, text: str, next_state: str = None) -> None:
        """Speak and pause ears during playback to prevent echo."""
        if self.debug:
            print(f"[Observer] Pausing ears")
        self.ears.paused = True
        self._last_spoken = text.lower().strip()
        self._last_spoken_time = time.time()
        self.face.set_caption(text)  # ← show in GUI
        self.face.set_state("speaking")
        with timer("TTS", self.debug):
            await self.mouth.speak(text)
        self.face.set_state(next_state or ("thinking" if self._processing else "listening"))        # check if canceled during speech
        if self.cancelled:
            self.ears.paused = False
            return
        await asyncio.sleep(0.5)
        self.ears.paused = False
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

    def _cancel_all(self):
        self.cancelled = True
        try:
            self.mouth.stop()
        except Exception as e:
            print(f"[Observer] Error stopping TTS: {e}")
        self.ears.paused = False
        self.face.set_state("listening")
        print("[Observer] Cancelled via GUI")
