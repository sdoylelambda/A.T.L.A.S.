import time
import asyncio
import threading
import datetime

from modules.ears import Ears
from modules.stt.hybrid_stt import HybridSTT
from modules.tts import TTSModule
from modules.app_launcher import AppLauncher
from modules.brain import Brain
from modules.tool_executor import ToolExecutor
from modules.browser_controller import BrowserController
from modules.utils import timer
from modules.calendar_module import CalendarModule
from modules.eyes import Eyes
from config.api_keys import get_api_key, set_key_request_callback
from custom_exceptions import PermissionRequired, ModelUnavailable, PlanExecutionError


class Observer:
    def __init__(self, face_controller, window_controller, config):
        self.debug = False
        self.face = face_controller
        self.window_controller = window_controller
        self.config = config
        self.paused = False
        self.cancelled = False
        self._processing = False
        self._last_spoken = ""
        self._last_spoken_time = 0
        self._finishing = False
        self._key_event = threading.Event()
        self._pending_key = None
        self._waiting_for_key = False
        self._current_state = None
        set_key_request_callback(self._request_key_via_gui)

        self.brain = Brain(config)
        self.ears = Ears()  # pass config
        self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
        self.browser_controller = BrowserController(config)
        self.launcher = AppLauncher(window_controller, self.browser_controller)
        self.executor = ToolExecutor(self.launcher, self.browser_controller, self.brain)
        calendar_enabled = config.get("integrations", {}).get("google_calendar", {}).get("enabled", False)
        self.calendar = CalendarModule(config) if calendar_enabled else None
        vision_enabled = config.get("vision", {}).get("enabled", False)
        self.eyes = Eyes(config) if vision_enabled else None
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

                if len(words) >= 3 and len(unique) == 1:
                    print(f"[STT] Repetition hallucination detected, skipping: {text[:50]}")
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
                    # calendar commands
                                       "today's events", "next event", "what's next", "this week", "upcoming events"
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
                wake_word = self.config.get("personalize", {}).get("ai_assistant_name", "atlas").lower()

                if wake_word in text or "you there" in text or "wake up" in text or "atlas" in text:
                    if self.paused:
                        self.paused = False
                        self.face.set_state("thinking")
                        with timer("TTS wake", self.debug):
                            await self.say("For you sir, always.")
                        self.face.set_state("listening")
                    continue

                # 😴 Pause command
                if any(phrase in text for phrase in ["take a break", "a break", "stop listening", "go to sleep"]):
                    self.paused = True
                    self.face.set_state("sleeping")
                    with timer("TTS sleep", self.debug):
                        await self.say("Going on a break.")
                    continue

                if self.paused:
                    continue

                if any(phrase in text for phrase in [
                    "what can you do", "what are you capable of",
                    "help", "capabilities", "what do you know"
                ]):
                    await self.say(self._build_capabilities(), next_state="listening")
                    continue

                # 📅 Calendar commands
                if self.calendar:
                    # 📅 Read intent — check BEFORE create intent
                    read_phrases = [
                        "what's on my calendar", "what is on my calendar",
                        "check my calendar", "show my calendar",
                        "what do i have", "what's on my schedule"
                    ]
                    if any(phrase in text for phrase in read_phrases):
                        if any(w in text for w in ["this week", "upcoming", "week"]):
                            events = await asyncio.to_thread(self.calendar.get_upcoming_events, 7)
                            await self.say(self.calendar.format_events_for_speech(events, multi_day=True),
                                           next_state="listening")
                        elif any(w in text for w in ["tomorrow", "tomorrow's"]):
                            events = await asyncio.to_thread(self.calendar.get_tomorrows_events)
                            await self.say(self.calendar.format_events_for_speech(events), next_state="listening")
                        else:
                            # default to today
                            events = await asyncio.to_thread(self.calendar.get_todays_events)
                            await self.say(self.calendar.format_events_for_speech(events), next_state="listening")
                        continue

                    # 📅 Create intent — only if explicit add/schedule word present
                    if any(phrase in text for phrase in [
                        "add ", "schedule ", "create event", "new event",
                        "new appointment", "remind me", "set a reminder"
                    ]) and any(w in text for w in [
                        "today", "tomorrow", "monday", "tuesday", "wednesday",
                        "thursday", "friday", "saturday", "sunday",
                        "tonight", "this evening", "next week"
                    ]):

                        # try to parse directly first
                        parsed = self.calendar.parse_event_from_text(text)

                        if parsed and parsed.get("time"):
                            # have enough info — create directly
                            result = await asyncio.to_thread(
                                self.calendar.create_event,
                                parsed["title"],
                                parsed["date"],
                                parsed.get("time"),
                                parsed.get("duration", 60)
                            )
                            day_str = "today" if parsed["date"] == datetime.date.today().isoformat() else parsed["date"]
                            await self.say(
                                f"Done, sir. {parsed['title']} added on {day_str} at {parsed.get('time')}.",
                                next_state="listening"
                            )
                        elif parsed and not parsed.get("time"):
                            # have title and date but no time — ask for time only
                            await self.say("What time, sir?")
                            audio_bytes, dur = await self.ears.listen()
                            time_text = self.stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""
                            combined = f"{text} {time_text}"
                            parsed = self.calendar.parse_event_from_text(combined)
                            if parsed:
                                result = await asyncio.to_thread(
                                    self.calendar.create_event,
                                    parsed["title"],
                                    parsed["date"],
                                    parsed.get("time"),
                                    parsed.get("duration", 60)
                                )
                                await self.say(
                                    f"Done, sir. {parsed['title']} added.",
                                    next_state="listening"
                                )
                            else:
                                await self.say("Sorry sir, I couldn't parse that. Please try again.",
                                               next_state="listening")
                        else:
                            # not enough info — guided flow
                            await self.say("What's the title of the event, sir?")
                            audio_bytes, dur = await self.ears.listen()
                            title = self.stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

                            await self.say("What day, sir?")
                            audio_bytes, dur = await self.ears.listen()
                            day_text = self.stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

                            await self.say("What time, sir?")
                            audio_bytes, dur = await self.ears.listen()
                            time_text = self.stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

                            combined = f"{title} {day_text} {time_text}"
                            parsed = self.calendar.parse_event_from_text(combined)

                            if parsed:
                                result = await asyncio.to_thread(
                                    self.calendar.create_event,
                                    parsed["title"],
                                    parsed["date"],
                                    parsed.get("time"),
                                    parsed.get("duration", 60)
                                )
                                await self.say(f"Done, sir. {parsed['title']} added.", next_state="listening")
                            else:
                                await self.say("Sorry sir, I wasn't able to create that event.", next_state="listening")

                        continue

                    calendar_words = ["today", "schedule", "events", "calendar", "this week", "upcoming",
                                      "next meeting", "next event"]

                    if any(phrase in text for phrase in calendar_words):
                        if any(w in text for w in ["this week", "upcoming", "week"]):
                            events = await asyncio.to_thread(self.calendar.get_upcoming_events, 7)
                            await self.say(self.calendar.format_events_for_speech(events, multi_day=True),
                                           next_state="listening")
                            continue


                        if any(w in text for w in ["tomorrow", "tomorrow's"]):
                            events = await asyncio.to_thread(self.calendar.get_tomorrows_events)
                            await self.say(self.calendar.format_events_for_speech(events), next_state="listening")
                            continue

                        if any(w in text for w in ["next meeting", "next event", "what's next", "next appointment"]):
                            event = await asyncio.to_thread(self.calendar.get_next_event)
                            response = self.calendar.format_events_for_speech(
                                [event]) if event else "Nothing coming up, sir."
                            await self.say(response, next_state="listening")
                            continue

                        if any(w in text for w in ["today", "schedule today", "my schedule"]):
                            events = await asyncio.to_thread(self.calendar.get_todays_events)
                            await self.say(self.calendar.format_events_for_speech(events), next_state="listening")
                            continue

                # 👁️ Vision commands
                if self.eyes:
                    if self.debug:
                        print(f"[Eyes] checking text: '{text}' eyes={self.eyes}")

                    if any(phrase in text for phrase in [
                        "what do you see", "what can you see", "what's in front",
                        "describe what you see", "look around", "describe the scene",
                        "what's in the room", "what's around", "describe my workspace",
                        "what's on my desk", "what's behind me", "take a picture", "take a photo"
                    ]):
                        self.face.set_caption("looking...")
                        result = await self._vision_check_and_analyze(self.eyes.describe_scene)
                        await self.say(result, next_state="listening")
                        continue

                    if any(phrase in text for phrase in [
                        "what am i holding", "what is this", "identify this",
                        "what's this object", "what object is this"
                    ]):
                        self.face.set_caption("identifying...")
                        result = await self._vision_check_and_analyze(self.eyes.identify_object)
                        await self.say(result, next_state="listening")
                        continue

                    if any(phrase in text for phrase in [
                        "read this", "what does this say", "read the text",
                        "what's written", "transcribe this", "read this document",
                        "what does this paper say", "read the text on screen"
                    ]):
                        self.face.set_caption("reading...")
                        result = await self._vision_check_and_analyze(self.eyes.read_document)
                        await self.say(result, next_state="listening")
                        continue

                    if any(phrase in text for phrase in [
                        "is anyone there", "is someone there", "anyone in the room",
                        "is there someone", "who's there", "who is in the room",
                        "how many people", "is anyone looking", "anyone here"
                    ]):
                        self.face.set_caption("checking...")
                        result = await self._vision_check_and_analyze(self.eyes.count_people)
                        await self.say(result, next_state="listening")
                        continue

                    if any(phrase in text for phrase in [
                        "what am i doing", "what are they doing", "what's happening",
                        "describe the activity", "what is this person doing"
                    ]):
                        self.face.set_caption("observing...")
                        result = await self._vision_check_and_analyze(self.eyes.describe_activity)
                        await self.say(result, next_state="listening")
                        continue

                    if any(phrase in text for phrase in [
                        "what color is this", "what colour is this",
                        "what color am i", "what colour am i"
                    ]):
                        self.face.set_caption("analyzing color...")
                        result = await self._vision_check_and_analyze(self.eyes.identify_color)
                        await self.say(result, next_state="listening")
                        continue

                    if any(phrase in text for phrase in [
                        "do you see", "can you see", "is there a", "is there an",
                        "does this look", "does this seem", "what does this look like"
                    ]):
                        self.face.set_caption("looking...")
                        result = await self._vision_check_and_analyze(
                            lambda: self.eyes.analyze_with_question(text)
                        )
                        await self.say(result, next_state="listening")
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

    async def _vision_check_and_analyze(self, analyze_fn) -> str:
        warning = self.eyes.check_storage()
        if warning:
            await self.say(warning)
        self.face.set_state("thinking")
        return await asyncio.to_thread(analyze_fn)

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
        print(f"[Heard] {response}")

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

    def _request_key_via_gui(self, provider: str) -> str:
        """Called from brain thread — pauses until GUI provides key."""
        self._pending_key = None
        self._waiting_for_key = True
        self._key_event.clear()
        print(f"[APIKeys] Waiting for key input in GUI for {provider}")
        self.face.set_caption(f"{provider} API key required — type in text box and press Send")
        self.face.set_state("thinking")
        self._key_event.wait(timeout=120)
        print(f"[APIKeys] Got key: {bool(self._pending_key)}")
        self._waiting_for_key = False
        if not self._pending_key:
            raise ValueError(f"No API key provided for {provider}")
        return self._pending_key

    def _provide_key(self, key: str):
        """Called from GUI text input when in key-request mode."""
        self._pending_key = key.strip()
        self._key_event.set()

    def _build_capabilities(self) -> str:
        core = (
            "Quite a few things, sir. "
            "I can answer questions, tell jokes, and hold a conversation. "
            "I can open applications, control your browser, search the web, and navigate pages. "
            "I can create files and folders, and generate code in Python, JavaScript, HTML, and more. "
            "I can read files, run scripts, and manage your workspace. "
        )

        if self.eyes:
            core += (
                "I have eyes through your web cam — I can see what's in front of me, "
                "identify objects, read text, and answer questions about what I see. "
            )

        api_caps = []
        if self.brain.api_models.get("gemini", {}).get("enabled"):
            api_caps.append("Gemini for real-time information and long document analysis")
        if self.brain.api_models.get("claude", {}).get("enabled"):
            api_caps.append("Claude for complex reasoning and large codebases")

        if api_caps:
            core += f"I also have access to {' and '.join(api_caps)}. "

        integration_caps = []
        if self.config.get("integrations", {}).get("google_calendar", {}).get("enabled"):
            integration_caps.append("check and manage your calendar")
        if self.config.get("integrations", {}).get("gmail", {}).get("enabled"):
            integration_caps.append("read and send emails via Gmail")

        if integration_caps:
            core += f"I can also {' and '.join(integration_caps)}. "

        core += "What would you like me to do, sir?"
        return core