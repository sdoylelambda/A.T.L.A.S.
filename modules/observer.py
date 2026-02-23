import yaml
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







# import asyncio
# import whisper
#
# class Observer:
#     def __init__(self, ears):
#         self.ears = ears
#         self.model = whisper.load_model("small", device="cpu")  # or tiny/medium
#
#     async def listen_and_respond(self):
#         print("[Observer] Listening and responding...")
#
#         audio_bytes, duration = await self.ears.listen(max_duration=30.0)
#         if not audio_bytes:
#             print("[Observer] No speech detected.")
#             return
#
#         print(f"[Observer] Captured {duration:.2f}s of audio.")
#
#         # Save temp WAV file
#         import tempfile, wave
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#             filename = f.name
#             wf = wave.open(filename, "wb")
#             wf.setnchannels(1)
#             wf.setsampwidth(2)
#             wf.setframerate(self.ears.rate)
#             wf.writeframes(audio_bytes)
#             wf.close()
#
#         # Transcribe English only
#         result = self.model.transcribe(filename, language="en")
#         text = result["text"].strip()
#         print(f"[Heard]: {text}")
#
#         # Here you can integrate with Launcher/TTS/etc.
#         if text:
#             print(f"[TTS] I heard: {text}")
#         else:
#             print("[TTS] Command not recognized.")




# # observer_async.py
# import asyncio
# import time
# from modules.stt.hybrid_stt import HybridSTT
# from modules.tts import TTSModule
# from modules.app_launcher import AppLauncher
# from modules.ears import Ears
#
# class Observer:
#     def __init__(self, face_controller, window_controller, config):
#         self.face = face_controller
#         self.window_controller = window_controller
#         self.paused = False
#
#         self.ears = Ears()
#         self.stt = HybridSTT(
#             whisper_model="small",
#             fw_model="small",
#             use_gpu=config["system"].get("use_gpu", False)
#         )
#         self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
#         self.launcher = AppLauncher(window_controller)
#
#     async def listen_and_respond(self):
#         # Initial greeting
#         self.face.set_state("thinking")
#         await asyncio.to_thread(self.mouth.speak, "Hello sir, what can I do for you.")
#         self.face.set_state("listening")
#         print("[Observer] Listening and responding...")
#
#         while True:
#             if self.paused:
#                 self.face.set_state("sleeping")
#             else:
#                 self.face.set_state("listening")
#
#             audio_bytes, duration = await self.ears.listen()
#             if not audio_bytes:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             try:
#                 if duration < 5:
#                     text = self.stt.transcribe_short(audio_bytes)
#                 else:
#                     text = self.stt.transcribe_long(audio_bytes)
#
#                 if not text:
#                     continue
#
#                 text = text.lower()
#                 print(f"[Heard]: {text}")
#
#                 # Hotwords
#                 if "jarvis" in text or "you there" in text:
#                     if self.paused:
#                         self.paused = False
#                         await asyncio.to_thread(self.mouth.speak, "I'm back online.")
#                         self.face.set_state("listening")
#                     continue
#
#                 if "take a break" in text:
#                     self.paused = True
#                     await asyncio.to_thread(self.mouth.speak, "Going on a break.")
#                     self.face.set_state("sleeping")
#                     continue
#
#                 if self.paused:
#                     continue
#
#                 handled = self.launcher.handle_command(text)
#                 if handled:
#                     current_app = self.launcher.get_current_app()
#                     if current_app and current_app != "browser":
#                         self.window_controller.update_active_window(current_app)
#                     self.face.set_state("thinking")
#                     await asyncio.to_thread(self.mouth.speak, f"I have: {text}")
#                 else:
#                     self.face.set_state("error")
#                     await asyncio.to_thread(self.mouth.speak, "Command not recognized.")
#
#             except Exception as e:
#                 print(f"[Observer Error]: {e}")
#                 self.face.set_state("error")
#
#             await asyncio.sleep(0.01)





# import yaml
# import time
# import asyncio
# from modules.ears import Ears
# from modules.stt.hybrid_stt import HybridSTT
# from modules.tts import TTSModule
# from modules.app_launcher import AppLauncher
#
# class Observer:
#     def __init__(self, face_controller, window_controller):
#         self.face = face_controller
#         self.window_controller = window_controller
#         with open("config.yaml") as f:
#             config = yaml.safe_load(f)
#
#         self.ears = Ears()
#         self.stt = HybridSTT(
#             whisper_model="small",
#             fw_model="small",
#             use_gpu=config["system"].get("use_gpu", False)
#         )
#         self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
#         self.launcher = AppLauncher(window_controller)
#         self.paused = False
#
#     async def listen_and_respond(self):
#         # Initial greeting
#         self.face.set_state("thinking")
#         print("[Observer] Listening and responding...")
#         await asyncio.to_thread(self.mouth.speak, "Hello sir, what can I do for you.")
#
#         while True:
#             if self.paused:
#                 self.face.set_state("sleeping")
#             else:
#                 self.face.set_state("listening")
#
#             audio_bytes, duration = await self.ears.listen(max_duration=3.0, silence_duration=0.4)
#             if not audio_bytes:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             # Choose STT method based on duration
#             try:
#                 if duration < 5:
#                     text = self.stt.transcribe_short(audio_bytes)
#                 else:
#                     text = self.stt.transcribe_long(audio_bytes)
#             except Exception as e:
#                 print(f"[STT Error]: {e}")
#                 continue
#
#             if not text:
#                 continue
#
#             text = text.lower()
#             print(f"[Heard]: {text}")
#
#             # Hotwords
#             if "jarvis" in text or "you there" in text:
#                 if self.paused:
#                     self.paused = False
#                     self.face.set_state("listening")
#                     await asyncio.to_thread(self.mouth.speak, "I'm back online.")
#                 continue
#
#             if "take a break" in text:
#                 self.paused = True
#                 self.face.set_state("sleeping")
#                 await asyncio.to_thread(self.mouth.speak, "Going on a break.")
#                 continue
#
#             # Handle commands
#             handled = self.launcher.handle_command(text)
#             if handled:
#                 current_app = self.launcher.get_current_app()
#                 if current_app and current_app != "browser":
#                     self.window_controller.update_active_window(current_app)
#                 self.face.set_state("thinking")
#                 await asyncio.to_thread(self.mouth.speak, f"I have: {text}")
#             else:
#                 self.face.set_state("error")
#                 await asyncio.to_thread(self.mouth.speak, "Command not recognized.")
#
#             await asyncio.sleep(0.01)










# import yaml
# import time
# from modules.ears import Ears
# from modules.stt.hybrid_stt import HybridSTT
# from modules.tts import TTSModule
# from modules.app_launcher import AppLauncher
#
# class Observer:
#     def __init__(self, face_controller, window_controller):
#         self.window_controller = window_controller
#         with open("config.yaml") as f:
#             config = yaml.safe_load(f)
#
#         self.ears = Ears(
#             # samplerate=config["audio"]["samplerate"],
#             # mic_index=config["audio"].get("mic_index")
#         )
#
#         self.stt = HybridSTT(
#             whisper_model="small",
#             fw_model="small",
#             use_gpu=config["system"].get("use_gpu", False)
#         )
#
#         self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
#         self.launcher = AppLauncher(window_controller)
#         self.face = face_controller
#         self.paused = False
#
#         # Initial greeting
#         self.face.set_state("thinking")
#         print("[Observer] Listening and responding...")
#         self.mouth.speak("Hello sir, what can I do for you.")
#
#     async def listen_and_respond(self):
#         self.face.set_state("listening")
#         while True:
#             if self.paused:
#                 self.face.set_state("sleeping")
#             else:
#                 self.face.set_state("listening")
#
#             print("STARTING EARS.LISTEN")
#             audio_path, duration = await self.ears.listen()
#             print("COMPLETED EARS.LISTEN")
#
#             if not audio_path:
#                 time.sleep(0.1)  # small pause for background noise
#                 continue
#
#             try:
#                 text = self.stt.transcribe_short(audio_path) if duration < 5 else self.stt.transcribe_long(audio_path)
#                 if not text:
#                     continue
#
#                 text = text.lower()
#                 print(f"[Heard]: {text}")
#
#                 # pause/resume hotwords
#                 if "jarvis" in text or "you there" in text:
#                     if self.paused:
#                         self.paused = False
#                         self.mouth.speak("I'm back online.")
#                         self.face.set_state("listening")
#                     continue
#
#                 if "take a break" in text:
#                     self.paused = True
#                     self.mouth.speak("Going on a break.")
#                     self.face.set_state("sleeping")
#                     continue
#
#                 if self.paused:
#                     continue
#
#                 handled = self.launcher.handle_command(text)
#                 if handled:
#                     current_app = self.launcher.get_current_app()
#                     if current_app and current_app != "browser":
#                         self.window_controller.update_active_window(current_app)
#                     self.face.set_state("thinking")
#                     self.mouth.speak(f"I have: {text}")
#                 else:
#                     self.face.set_state("error")
#                     self.mouth.speak("Command not recognized.")
#
#             except Exception as e:
#                 print(f"[Error]: {e}")
#                 self.face.set_state("error")
#
#             time.sleep(0.01)  # CPU/GPU smoothness




# import yaml
# import time
# from modules.ears import Ears
# from modules.stt.hybrid_stt import HybridSTT
# from modules.tts import TTSModule
# from modules.app_launcher import AppLauncher
#
# class Observer:
#     # Sleep color (dark blue)
#     SLEEP_COLOR = [0.0, 0.0, 0.0002, 1.0]
#
#     def __init__(self, face_controller):
#         with open("config.yaml") as f:
#             config = yaml.safe_load(f)
#
#         self.ears = Ears(
#             samplerate=config["audio"]["samplerate"],
#             mic_index=config["audio"].get("mic_index"),
#             duration=7
#         )
#
#         self.stt = HybridSTT(
#             whisper_model="small",
#             fw_model="small",
#             use_gpu=config["system"].get("use_gpu", False)
#         )
#
#         self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
#         self.launcher = AppLauncher()
#         self.face = face_controller
#         self.paused = False
#
#         # Initial greeting
#         self.face.set_state("thinking")
#         print("[Observer] Listening and responding...")
#         self.mouth.speak("Hello sir, what can I do for you.")
#
#     def listen_and_respond(self):
#         while True:
#             # Set visual state
#             if self.paused:
#                 # Dark blue sleep state
#                 self.face.current_color = self.SLEEP_COLOR.copy()
#             else:
#                 self.face.set_state("listening")
#
#             audio_path, duration = self.ears.listen()
#
#             if not audio_path:
#                 time.sleep(0.02)
#                 continue
#
#             try:
#                 if duration < 5:
#                     text = self.stt.transcribe_short(audio_path)
#                 else:
#                     text = self.stt.transcribe_long(audio_path)
#
#                 if not text:
#                     continue
#
#                 text = text.lower()
#                 print(f"[Heard]: {text}")
#
#                 # Hotword to resume
#                 if "jarvis" in text:
#                     if self.paused:
#                         self.paused = False
#                         self.mouth.speak("I'm here.")
#                         self.face.set_state("listening")
#                     continue  # skip command execution this turn
#
#                 # Hotword to pause
#                 if "take a break" in text:
#                     self.paused = True
#                     self.mouth.speak("Going on a break.")
#                     # Start sleep color (dark blue)
#                     self.face.current_color = self.SLEEP_COLOR.copy()
#                     continue
#
#                 if self.paused:
#                     continue  # skip commands while paused
#
#                 # Execute normal commands
#                 handled = self.launcher.handle_command(text)
#                 if not handled:
#                     self.mouth.speak("Command not recognized.")
#                     self.listen_and_respond()
#
#                 print(f"[Heard]: {text}")
#                 self.mouth.speak(f"I will: {text}")
#
#             except Exception as e:
#                 print(f"[Error]: {e}")
#                 self.face.set_state("error")
#
#             # Small sleep for CPU/GPU smoothness
#             time.sleep(0.01)






# import yaml
# import time
# from modules.ears import Ears
# from modules.stt.hybrid_stt import HybridSTT
# from modules.tts import TTSModule
# from modules.app_launcher import AppLauncher
#
# class Observer:
#     def __init__(self, face_controller):
#         with open("config.yaml") as f:
#             config = yaml.safe_load(f)
#
#         self.ears = Ears(
#             samplerate=config["audio"]["samplerate"],
#             mic_index=config["audio"].get("mic_index"),
#             duration=7
#         )
#
#         self.stt = HybridSTT(
#             whisper_model="small",
#             fw_model="small",
#             use_gpu=config["system"].get("use_gpu", False)
#         )
#
#         self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
#         self.launcher = AppLauncher()
#         self.face = face_controller
#         self.paused = False
#
#         self.face.set_state("thinking")
#         print("[Observer] Listening and responding...")
#         self.mouth.speak(f"Hello sir, what can I do for you.")
#
#     def listen_and_respond(self):
#         while True:
#             self.face.set_state("listening")
#             audio_path, duration = self.ears.listen()
#
#             if not audio_path:
#                 time.sleep(0.02)
#                 continue
#
#             self.face.set_state("thinking")
#
#             try:
#                 if duration < 5:
#                     text = self.stt.transcribe_short(audio_path)
#                 else:
#                     text = self.stt.transcribe_long(audio_path)
#
#                 if not text:
#                     continue
#
#                 # Hotword to resume
#                 if "jarvis" in text:
#                     if self.paused:
#                         self.paused = False
#                         self.mouth.speak("I'm back online.")
#                         if self.face:
#                             self.face.set_state("listening")
#                     continue  # don't execute commands this turn
#
#                 # Hotword to pause
#                 if "take a break" in text:
#                     self.paused = True
#                     self.mouth.speak("Going on a break.")
#                     if self.face:
#                         self.face.set_state("error")  # or any color to indicate pause
#                     continue
#
#                 if self.paused:
#                     # If paused, ignore commands
#                     continue
#
#                 handled = self.launcher.handle_command(text)
#                 if not handled:
#                     self.mouth.speak("Command not recognized.")
#                     self.listen_and_respond()
#
#                 print(f"[Heard]: {text}")
#                 self.mouth.speak(f"I will: {text}")
#
#             except Exception as e:
#                 print(f"[Error]: {e}")
#                 self.face.set_state("error")
#
#             # Small sleep to give GUI time
#             time.sleep(0.03)





# # modules/observer.py
# import yaml
# from modules.ears import Ears
# from modules.stt.hybrid_stt import HybridSTT
# from modules.tts import TTSModule
# from modules.app_launcher import AppLauncher
#
# class Observer:
#     def __init__(self, face_controller):
#         with open("config.yaml") as f:
#             config = yaml.safe_load(f)
#
#         self.ears = Ears(
#             samplerate=config["audio"]["samplerate"],
#             mic_index=config["audio"].get("mic_index"),
#             duration=7
#         )
#
#         self.stt = HybridSTT(
#             whisper_model="small",
#             fw_model="small",
#             use_gpu=config["system"].get("use_gpu", False)
#         )
#
#         self.mouth = TTSModule(use_mock=config["audio"].get("use_mock", False))
#         self.launcher = AppLauncher()
#         self.face = face_controller  # Reference to FaceController
#
#     def listen_and_respond(self):
#         print("[Observer] Listening and responding...")
#
#         while True:
#             self.face.set_state("listening")
#             audio_path, duration = self.ears.listen()
#
#             if not audio_path:
#                 continue
#
#             self.face.set_state("thinking")
#
#             try:
#                 if duration < 5:
#                     text = self.stt.transcribe_short(audio_path)
#                 else:
#                     text = self.stt.transcribe_long(audio_path)
#
#                 if not text:
#                     continue
#
#                 print(f"[Heard]: {text}")
#                 self.mouth.speak(f"I will: {text}")
#
#                 handled = self.launcher.handle_command(text)
#                 if not handled:
#                     self.mouth.speak("Command not recognized.")
#
#             except Exception as e:
#                 print(f"[Error]: {e}")
#                 self.face.set_state("error")
#
#
#
#
# # observer.py
# # import yaml
# # from modules.ears import Ears
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# #
# # class Observer:
# #     def __init__(self, config_path="config.yaml"):
# #         with open(config_path) as f:
# #             config = yaml.safe_load(f)
# #
# #         samplerate = config["audio"]["samplerate"]
# #         mic_index = config["audio"].get("mic_index")
# #         use_gpu = config["system"].get("use_gpu", False)
# #         use_mock = config["audio"].get("use_mock", False)
# #
# #         self.stt = HybridSTT(whisper_model="small", fw_model="small", use_gpu=use_gpu)
# #         self.ears = Ears(stt=self.stt, samplerate=samplerate, mic_index=mic_index, duration=7)
# #         self.mouth = TTSModule(use_mock=use_mock)
# #         self.launcher = AppLauncher()
# #
# #     def listen_and_respond(self):
# #         print("[Observer] Listening and responding...")
# #         while True:
# #             result = self.ears.listen()  # could be str or list
# #             if not result:
# #                 continue
# #
# #             # Normalize to list for uniform handling
# #             if isinstance(result, str):
# #                 texts = [result]  # short command
# #             elif isinstance(result, list):
# #                 texts = result  # long dictation
# #             else:
# #                 continue
# #
# #             for text in texts:
# #                 words = len(text.split())
# #                 print(f"[Heard]: {text} ({words} words)")
# #
# #                 if words <= 5:
# #                     # Short command → open apps
# #                     self.launcher.open_app(text)
# #                     self.mouth.speak(text)
# #                 else:
# #                     # Long dictation → just speak
# #                     self.mouth.speak(text)
#
# # observer.py
# # import yaml
# # from modules.ears import Ears
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# #
# # class Observer:
# #     def __init__(self, config_path="config.yaml"):
# #         with open(config_path) as f:
# #             config = yaml.safe_load(f)
# #
# #         samplerate = config["audio"]["samplerate"]
# #         mic_index = config["audio"].get("mic_index")
# #         use_gpu = config["system"].get("use_gpu", False)
# #         use_mock = config["audio"].get("use_mock", False)
# #
# #         self.stt = HybridSTT(whisper_model="small", fw_model="small", use_gpu=use_gpu)
# #         self.ears = Ears(stt=self.stt, samplerate=samplerate, mic_index=mic_index, duration=7)
# #         self.mouth = TTSModule(use_mock=use_mock)
# #         self.launcher = AppLauncher()
# #
# #     def listen_and_respond(self):
# #         """Main loop: record full audio, decide short vs long"""
# #         while True:
# #             text = self.ears.listen()
# #             if not text:
# #                 continue
# #             print(f"[Heard]: {text}")
# #
# #             # Short commands → open apps
# #             if len(text.split()) <= 5:  # heuristic: short command
# #                 self.launcher.open_app(text)
# #
# #             # Speak response
# #             self.mouth.speak(text)
#
#
#
#
#
#
#
#
#
#
# # modules/observer.py
# # import yaml
# # from modules.ears import Ears
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# # import threading
# # import queue
# #
# # class Observer:
# #     def __init__(self):
# #         with open("config.yaml") as f:
# #             config = yaml.safe_load(f)
# #
# #         self.samplerate = config["audio"]["samplerate"]
# #         self.mic_index = config["audio"].get("mic_index")
# #         self.use_mock = config["audio"].get("use_mock", False)
# #         self.use_gpu = config["system"].get("use_gpu", False)
# #
# #         # STT
# #         self.stt = HybridSTT(
# #             whisper_model="small",
# #             fw_model="small",
# #             use_gpu=self.use_gpu,
# #             samplerate=self.samplerate
# #         )
# #
# #         # Ears
# #         self.ears = Ears(
# #             stt=self.stt,
# #             samplerate=self.samplerate,
# #             mic_index=self.mic_index
# #         )
# #
# #         # TTS
# #         self.mouth = TTSModule(use_mock=self.use_mock)
# #
# #         # App Launcher
# #         self.launcher = AppLauncher()
# #
# #         # Queue for long dictation
# #         self.long_audio_queue = queue.Queue()
# #         self.running = True
# #         threading.Thread(target=self._process_long_dictation, daemon=True).start()
# #
# #     def _process_long_dictation(self):
# #         while self.running:
# #             try:
# #                 audio_array = self.long_audio_queue.get(timeout=0.5)
# #             except queue.Empty:
# #                 continue
# #             text = self.stt.transcribe(audio_array, long=True)
# #             if text:
# #                 print(f"[Long Dictation] {text}")
# #                 self.mouth.speak(text)
# #
# #     def listen_and_respond(self):
# #         print("[Observer] Listening and responding...")
# #         for text in self.ears.listen_stream(short_chunk_sec=1.0, long_chunk_sec=5.0):
# #             if not text:
# #                 continue
# #             print(f"[Heard]: {text}")
# #             # Check if short command (<5s)
# #             if len(text.split()) <= 5:
# #                 self.launcher.open_app(text)
# #                 self.mouth.speak(text)
# #             else:
# #                 # Queue long dictation
# #                 print("[Observer] Queueing long dictation...")
# #                 # For simplicity, we can record long audio separately or skip
#
#
#
#
#
#
#
#
#
#
#
#
# # modules/observer.py
# # import yaml
# # from modules.ears import Ears
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# # import threading
# # import queue
# #
# # class Observer:
# #     def __init__(self, config_path="config.yaml"):
# #         with open(config_path) as f:
# #             config = yaml.safe_load(f)
# #
# #         self.samplerate = config["audio"]["samplerate"]
# #         self.mic_index = config["audio"].get("mic_index")
# #         self.use_mock = config["audio"].get("use_mock", False)
# #         self.use_gpu = config["system"].get("use_gpu", False)
# #
# #         # Initialize HybridSTT
# #         self.stt = HybridSTT(
# #             whisper_model="small",
# #             fw_model="small",
# #             use_gpu=self.use_gpu
# #         )
# #
# #         # Initialize Ears
# #         self.ears = Ears(
# #             stt=self.stt,
# #             samplerate=self.samplerate,
# #             mic_index=self.mic_index,
# #             duration=7  # max long dictation buffer
# #         )
# #
# #         # Initialize TTS
# #         self.mouth = TTSModule(use_mock=self.use_mock)
# #
# #         # Initialize App Launcher
# #         self.launcher = AppLauncher()
# #
# #         # Queue for background long dictation
# #         self.long_audio_queue = queue.Queue()
# #         self.running = True
# #
# #         # Background thread to process long dictation
# #         threading.Thread(target=self._process_long_dictation, daemon=True).start()
# #
# #     def _process_long_dictation(self):
# #         """Continuously process long audio in background to avoid blocking short commands."""
# #         while self.running:
# #             try:
# #                 audio_array = self.long_audio_queue.get(timeout=0.5)
# #             except queue.Empty:
# #                 continue
# #
# #             text = self.stt.transcribe(audio_array)
# #             if text:
# #                 print(f"[Long Dictation] {text}")
# #                 self.mouth.speak(text)
# #
# #     def listen_and_respond(self):
# #         """Main streaming loop: handle short commands immediately, queue long audio."""
# #         print("[Observer] Listening and responding...")
# #         for text in self.ears.listen_stream():
# #             if not text:
# #                 continue
# #
# #             # Short commands: send to launcher + TTS immediately
# #             if len(text.split()) <= 6:  # heuristic: few words → short command
# #                 print(f"[Short Command] {text}")
# #                 self.launcher.open_app(text)
# #                 self.mouth.speak(text)
# #             else:
# #                 # Long dictation → queue for background processing
# #                 print(f"[Queueing long dictation] {text[:50]}{'...' if len(text) > 50 else ''}")
# #                 # You can optionally convert back to audio_array if needed
# #                 # For now, we just use text queue for TTS
# #                 self.long_audio_queue.put(text)
# #
# #     def stop(self):
# #         self.running = False
# #         print("[Observer] Stopped.")
#
#
#
#
#
#
#
#
#
#
#
#
# # modules/stt/hybrid_stt.py
# # import numpy as np
# # import whisper
# # from modules.stt.faster_whisper_stt import WhisperModel
# #
# #
# # class HybridSTT:
# #     def __init__(self, whisper_model="small", fw_model="small", use_gpu=False):
# #         self.use_gpu = use_gpu
# #         self.samplerate = 16000
# #
# #         # Whisper CPU for short commands
# #         self.whisper_model = whisper.load_model(
# #             whisper_model,
# #             device="cuda" if use_gpu else "cpu"
# #         )
# #
# #         # Faster Whisper (int8) for long dictation
# #         device = "cuda" if use_gpu else "cpu"
# #         self.fw_model = WhisperModel(
# #             fw_model,
# #             device=device,
# #             compute_type="int8"
# #         )
# #
# #     def transcribe(self, audio_array):
# #         """
# #         Automatically choose model based on duration:
# #         - ≤5s: Whisper (CPU) → short commands
# #         - >5s: Faster Whisper → long dictation
# #         """
# #         duration_sec = len(audio_array) / self.samplerate
# #         audio_float = audio_array.astype(np.float32) / 32768.0  # int16 → float32
# #
# #         # Short command (Whisper)
# #         if duration_sec <= 5.0:
# #             try:
# #                 # Save to file for compatibility
# #                 from scipy.io.wavfile import write
# #                 write("capture.wav", self.samplerate, audio_array)
# #                 result = self.whisper_model.transcribe("capture.wav", language="en")
# #                 text = result.get("text", "").strip()
# #                 return text
# #             except Exception as e:
# #                 print(f"[Whisper] Error: {e}")
# #                 return ""
# #
# #         # Long dictation (Faster Whisper)
# #         else:
# #             try:
# #                 if audio_array.ndim > 1:
# #                     audio_float = audio_array.mean(axis=1)
# #                 segments, _ = self.fw_model.transcribe(
# #                     audio_float,
# #                     language="en",
# #                     temperature=0.0,       # deterministic
# #                     suppress_blank=True,   # avoid extra blanks
# #                     word_timestamps=False, # we don’t need timestamps
# #                     beam_size=1
# #                 )
# #                 return " ".join([seg.text for seg in segments]).strip()
# #             except Exception as e:
# #                 print(f"[Faster Whisper] Error: {e}")
# #                 return ""
#
#
#
#
#
#
# # import yaml
# # from modules.ears import Ears
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# # import threading
# # import queue
# # import time
# #
# #
# # class Observer:
# #     def __init__(self):
# #         # Load config
# #         with open("config.yaml") as f:
# #             config = yaml.safe_load(f)
# #
# #         self.samplerate = config["audio"]["samplerate"]
# #         self.mic_index = config["audio"].get("mic_index", 0)
# #         self.use_mock = config["audio"].get("use_mock", False)
# #         self.use_gpu = config["system"].get("use_gpu", False)
# #
# #         # Initialize hybrid STT
# #         self.stt = HybridSTT(
# #             whisper_model="small",
# #             fw_model="small",
# #             use_gpu=self.use_gpu,
# #             samplerate=self.samplerate
# #         )
# #
# #         # Initialize Ears (streaming)
# #         self.ears = Ears(
# #             stt=self.stt,
# #             samplerate=self.samplerate,
# #             mic_index=self.mic_index,
# #             duration=7  # max buffer length for long dictation
# #         )
# #
# #         # Initialize TTS
# #         self.mouth = TTSModule(use_mock=self.use_mock)
# #
# #         # Initialize App Launcher
# #         self.launcher = AppLauncher()
# #
# #         # Queue for long dictation
# #         self.long_audio_queue = queue.Queue()
# #         self.running = True
# #
# #         # Start background thread for long dictation
# #         threading.Thread(target=self._process_long_dictation, daemon=True).start()
# #
# #     def _process_long_dictation(self):
# #         """Background thread to process long audio."""
# #         while self.running:
# #             try:
# #                 audio_array = self.long_audio_queue.get(timeout=0.5)
# #             except queue.Empty:
# #                 continue
# #             text = self.stt.transcribe(audio_array)
# #             if text:
# #                 print(f"[Long Dictation] {text}")
# #                 self.mouth.speak(text)
# #
# #     def listen_and_respond(self):
# #         """Main loop: stream audio from mic, process short commands immediately, queue long dictation."""
# #         print("[Observer] Listening and responding...")
# #         # Use overlapping streaming from Ears
# #         for audio_array in self.ears.listen_stream(short_chunk_sec=1.0, long_chunk_sec=5.0):
# #             if audio_array is None:
# #                 continue
# #
# #             duration_sec = len(audio_array) / self.samplerate
# #
# #             # Short command → Whisper CPU
# #             if duration_sec <= 5.0:
# #                 try:
# #                     text = self.stt.transcribe(audio_array, long=False)
# #                     if text:
# #                         print(f"[Short Command] {text}")
# #                         self.launcher.open_app(text)
# #                         self.mouth.speak(text)
# #                 except Exception as e:
# #                     print(f"[Observer][Short] STT error: {e}")
# #
# #             # Long dictation → Faster Whisper
# #             else:
# #                 self.long_audio_queue.put(audio_array)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # import yaml
# # from modules.ears import Ears
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# # import threading
# # import queue
# # import time
# #
# # class Observer:
# #     def __init__(self):
# #         with open("config.yaml") as f:
# #             config = yaml.safe_load(f)
# #
# #         self.samplerate = config["audio"]["samplerate"]
# #         self.mic_index = config["audio"].get("mic_index")
# #         self.use_mock = config["audio"].get("use_mock", False)
# #         self.use_gpu = config["system"].get("use_gpu", False)
# #
# #         # Initialize STT
# #         self.stt = HybridSTT(
# #             whisper_model="small",
# #             fw_model="small",
# #             use_gpu=self.use_gpu
# #         )
# #
# #         # Initialize Ears
# #         self.ears = Ears(
# #             stt=self.stt,
# #             samplerate=self.samplerate,
# #             mic_index=self.mic_index
# #         )
# #
# #         # Initialize TTS
# #         self.mouth = TTSModule(use_mock=self.use_mock)
# #
# #         # Initialize App Launcher
# #         self.launcher = AppLauncher()
# #
# #         # Queue for long dictation
# #         self.long_audio_queue = queue.Queue()
# #         self.running = True
# #
# #         # Start background thread for long dictation
# #         threading.Thread(target=self._process_long_dictation, daemon=True).start()
# #
# #     def _process_long_dictation(self):
# #         """Background thread for long audio transcription."""
# #         while self.running:
# #             try:
# #                 audio_array = self.long_audio_queue.get(timeout=0.5)
# #             except queue.Empty:
# #                 continue
# #             text = self.stt.transcribe(audio_array, long=True)
# #             if text:
# #                 print(f"[Long Dictation] {text}")
# #                 self.mouth.speak(text)
# #
# #     def listen_and_respond(self):
# #         """Main loop: stream from mic, check short commands, queue long audio."""
# #         print("[Observer] Listening and responding...")
# #         for audio_array in self.ears.listen_stream():  # 1s slices
# #             if audio_array is None:
# #                 continue
# #
# #             duration_sec = len(audio_array) / self.samplerate
# #
# #             # 1️⃣ Short command detection
# #             if duration_sec <= 1.0:
# #                 text = self.stt.transcribe(audio_array, long=False)
# #                 if text:
# #                     self.launcher.open_app(text)
# #                     self.mouth.speak(text)
# #
# #             # 2️⃣ Long dictation
# #             else:
# #                 # queue for background processing
# #                 self.long_audio_queue.put(audio_array)
#
#
#
#
#
#
#
#
#
# # import yaml
# # from modules.ears import Ears
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# #
# # class Observer:
# #     def __init__(self):
# #         with open("config.yaml") as f:
# #             config = yaml.safe_load(f)
# #
# #         self.samplerate = config["audio"]["samplerate"]
# #         self.mic_index = config["audio"].get("mic_index")
# #         self.use_mock = config["audio"].get("use_mock", False)
# #         self.use_gpu = config["system"].get("use_gpu", False)
# #
# #         # Initialize STT
# #         self.stt = HybridSTT(
# #             whisper_model="small",
# #             fw_model="small",
# #             use_gpu=self.use_gpu
# #         )
# #
# #         # Initialize Ears
# #         self.ears = Ears(
# #             stt=self.stt,
# #             samplerate=self.samplerate,
# #             mic_index=self.mic_index
# #         )
# #
# #         # Initialize TTS
# #         self.mouth = TTSModule(use_mock=self.use_mock)
# #
# #         # Initialize App Launcher
# #         self.launcher = AppLauncher()
# #
# #     def listen_and_respond(self):
# #         print("[Observer] Listening and responding...")
# #         for text in self.ears.listen_stream():
# #             if not text:
# #                 continue
# #
# #             # First, try launching apps if short command
# #             self.launcher.open_app(text)
# #
# #             # Speak back
# #             self.mouth.speak(text)
#
#
#
#
#
#
#
#
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# # from modules.ears import Ears
# #
# # class Observer:
# #     def __init__(self, use_mock=False):
# #         self.stt = HybridSTT(use_gpu=False)
# #         self.tts = TTSModule(use_mock=use_mock)
# #         self.launcher = AppLauncher()
# #         self.ears = Ears(stt=self.stt)
# #
# #     def listen_and_respond(self):
# #         for text in self.ears.listen_stream():
# #             print(f"[Heard]: {text}")
# #
# #             # Launch apps if recognized
# #             if self.launcher.open_app(text):
# #                 self.tts.speak(f"Opening {text}")
# #             else:
# #                 self.tts.speak(text)
#
#
#
#
#
#
#
#
#
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.tts import TTSModule
# # from modules.app_launcher import AppLauncher
# # import numpy as np
# #
# # class Observer:
# #     def __init__(self, use_mock=False):
# #         self.stt = HybridSTT(use_gpu=False)
# #         self.tts = TTSModule(use_mock=use_mock)
# #         self.launcher = AppLauncher()
# #         self.samplerate = self.stt.samplerate
# #
# #     def listen_and_respond(self, duration=5):
# #         audio = self.stt.record(duration)
# #         text = self.stt.transcribe(audio)
# #         print(f"[Heard]: {text}")
# #
# #         # App launcher
# #         if self.launcher.open_app(text):
# #             self.tts.speak(f"Opening {text}")
# #         else:
# #             self.tts.speak(text)
# #
#
#
#
#
#
#
#
#
#
# # from modules.stt.factory import create_stt
# # from modules.stt.faster_whisper_stt import FasterWhisperSTT
# # from modules.stt.hybrid_stt import HybridSTT
# # from modules.app_launcher import AppLauncher
# # from modules.tts import TTSModule
# # from modules.ears import Ears
# # import yaml
# #
# #
# # # Rename as Jarvis?
# # class Observer:
# #     def __init__(self, config_file="config.yaml"):
# #         with open(config_file) as f:
# #             config = yaml.safe_load(f)
# #
# #         self.stt = HybridSTT(
# #             whisper_model=config["stt"].get("whisper_model", "small"),
# #             fw_model=config["stt"].get("fw_model", "small"),
# #             use_gpu=False
# #         )
# #
# #         self.ears = Ears(
# #             self.stt,
# #             mic_index=config["audio"].get("mic_index", 0),
# #             chunk_duration=config["audio"].get("chunk_duration", 1.5)
# #         )
# #         self.tts = TTSModule()
# #         self.launcher = AppLauncher()
# #
# #     def listen_and_respond_stream(self):
# #         buffer = ""
# #         chunk_count = 0
# #         for chunk_text in self.ears.listen_stream():
# #             print(f"[Chunk heard]: {chunk_text}")
# #
# #             # Check for app command first
# #             if self.launcher.open_app(chunk_text):
# #                 self.tts.speak(f"Opening {chunk_text}")
# #                 buffer = ""
# #                 chunk_count = 0
# #                 continue
# #
# #             # Append chunk to buffer
# #             buffer += " " + chunk_text
# #             chunk_count += 1
# #
# #             # Speak buffer every 3 chunks (~4–5s)
# #             if chunk_count >= 3:
# #                 self.tts.speak(buffer)
# #                 buffer = ""
# #                 chunk_count = 0
#
#
#
#
#
#
#
#
#
#
#
# # class Observer:
# #     def __init__(self, config_file="config.yaml"):
# #         with open(config_file) as f:
# #             config = yaml.safe_load(f)
# #
# #         use_gpu = config["system"].get("use_gpu", False)
# #         self.stt = HybridSTT(
# #             whisper_model=config["stt"].get("whisper_model", "small"),
# #             fw_model=config["stt"].get("fw_model", "small"),
# #             use_gpu=use_gpu
# #         )
# #
# #         self.ears = Ears(
# #             self.stt,
# #             mic_index=config["audio"].get("mic_index", 0),
# #             chunk_duration=config["audio"].get("chunk_duration", 1.5)
# #         )
# #         self.tts = TTSModule(gpu=use_gpu)
# #         self.launcher = AppLauncher()
# #
# #     def listen_and_respond(self):
# #         buffer = ""
# #         chunk_count = 0
# #         for chunk_text in self.ears.listen_stream():
# #             print(f"[Chunk heard]: {chunk_text}")
# #
# #             # Check for app command in this chunk first
# #             if self.launcher.open_app(chunk_text):
# #                 self.tts.speak(f"Opening {chunk_text}")
# #                 buffer = ""  # clear buffer after command
# #                 chunk_count = 0
# #                 continue
# #
# #             # Append chunk to buffer
# #             buffer += " " + chunk_text
# #             chunk_count += 1
# #
# #             # Speak buffer every 3 chunks or after long pause
# #             if chunk_count >= 3:
# #                 self.tts.speak(buffer)
# #                 buffer = ""
# #                 chunk_count = 0
# #
# #
# #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     # def __init__(self, config_file="config.yaml"):
#     #     with open(config_file) as f:
#     #         config = yaml.safe_load(f)
#     #
#     #     use_gpu = config["system"].get("use_gpu", False)
#     #     self.stt = HybridSTT(
#     #         whisper_model=config["stt"].get("whisper_model", "small"),
#     #         fw_model=config["stt"].get("fw_model", "small"),
#     #         use_gpu=use_gpu
#     #     )
#     #     self.ears = Ears(self.stt, mic_index=config["audio"].get("mic_index", 0),
#     #                      duration=config["audio"].get("duration", 3))
#     #     self.tts = TTSModule(gpu=use_gpu)
#     #     self.launcher = AppLauncher()
#     #
#     # def listen_and_respond(self):
#     #     buffer = ""
#     #     for chunk_text in self.ears.listen_stream():
#     #         print(f"[Chunk heard]: {chunk_text}")
#     #
#     #         # Append chunk to buffer for long dictation
#     #         buffer += " " + chunk_text
#     #
#     #         # Check if user said an app command (only in the latest chunk)
#     #         if self.launcher.open_app(chunk_text):
#     #             self.tts.speak(f"Opening {chunk_text}")
#     #             buffer = ""  # clear buffer if command executed
#     #             continue
#     #
#     #         # Optional: speak every 2–3 chunks
#     #         if len(buffer.split()) > 15:  # adjust threshold
#     #             self.tts.speak(buffer)
#     #             buffer = ""
#         # text = self.ears.listen()
#         # text = text.strip()
#         # if not text:
#         #     return
#         #
#         # print(f"[Heard]: {text}")
#         #
#         # # Check for app commands first
#         # if self.launcher.open_app(text):
#         #     self.tts.speak(f"Opening {text}")
#         # else:
#         #     self.tts.speak(text)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     # def __init__(self, config_file="config.yaml"):
#     #     with open(config_file) as f:
#     #         config = yaml.safe_load(f)
#     #
#     #     # Initialize STT engine
#     #     engine = config["transcription"]["engine"].lower()
#     #     if engine == "faster-whisper":
#     #         self.stt = FasterWhisperSTT(config)
#     #     else:
#     #         raise ValueError(f"Unknown STT engine: {engine}")
#     #
#     #     # Microphone listener
#     #     self.ears = Ears(self.stt, use_mock=config["audio"].get("use_mock", False))
#     #
#     #     # TTS
#     #     self.tts = TTSModule(
#     #         use_mock=config["audio"].get("use_mock", False),
#     #         gpu=config["system"].get("use_gpu", False)
#     #     )
#     #
#     # def listen_and_speak(self):
#     #     text = self.ears.listen()
#     #     if len(text.strip()) < 2:
#     #         print("[Jarvis] Ignored too short input.")
#     #     else:  # refactor this
#     #         if text.strip():
#     #             print(f"[Heard]: {text}")
#     #             self.tts.speak(text)
#
#
#
#
#
#
#
#
#
#
# # class Observer:
# #     def __init__(self, config_file="config.yaml"):
# #         with open(config_file) as f:
# #             config = yaml.safe_load(f)
# #
# #         engine = config["transcription"]["engine"].lower()
# #         if engine == "faster_whisper":
# #             self.stt = FasterWhisperSTT(config)
# #         elif engine == "whisper":
# #             self.stt = WhisperSTT(config)
# #         else:
# #             raise ValueError(f"Unknown STT engine: {engine}")
# #
# #         self.ears = Ears(self.stt, use_mock=config["audio"].get("use_mock", False))
# #         self.tts = TTSModule(
# #             use_mock=config["audio"].get("use_mock", False),
# #             gpu=config["system"].get("use_gpu", False)
# #         )
# #
# #     def listen_and_speak(self):
# #         text = self.ears.listen()
# #         if text:
# #             print(f"[Heard]: {text}")
# #             self.tts.speak(text)
#
# # class Observer:
# #     def __init__(self):
# #         with open("config.yaml") as f:
# #             config = yaml.safe_load(f)
# #
# #         self.stt = create_stt(config)
# #         self.ears = Ears(
# #             stt=self.stt,
# #             samplerate=config["audio"]["samplerate"],
# #             duration=config["audio"]["duration"],
# #             mic_index=config["audio"].get("mic_index"),
# #             use_mock=config["audio"].get("use_mock", False),
# #         )
# #
# #     def listen(self):
# #         return self.ears.listen()
#
#
#
#
#
# # from modules.stt.__init__ import create_stt
# # import yaml
# # from modules.brain import Brain
# # from modules.hands import Hands
# # import time
# #
# #
# # class Observer:
# #     def __init__(self):
# #         with open("config.yaml") as f:
# #             config = yaml.safe_load(f)
# #
# #         self.stt = create_stt(config)
# #         print("Observer initialized.")
# #
# #     def listen(self, audio):
# #         # audio_path = input("Simulated audio path or text: ")
# #         # For real mic: integrate PyAudio
# #         # return self.stt.transcribe(audio_path)
# #         # return self.stt.ears()
# #         return self.stt.transcribe(audio)
# #
# #     def listen_confirmation(self):
# #         print("[Observer] Awaiting confirmation (yes/no)...")
# #         response = self.stt.ears()
# #         if response:
# #             return response.lower() in ["yes", "yep", "affirmative"]
# #         return False
# #
# #     def speak(self, text):
# #         print(f"Jarvis says: {text}")
# #         self.tts.speak(text, play_audio=True)
# #
# #
#
#
#
#
# # import pyttsx3
# # from modules.brain import Brain
# # from modules.hands import Hands
# # from modules.ears import Ears
# # from modules.mouth import Mouth
#
#
# # class Observer:
# #     def __init__(self):
# #         print("Observer initialized.")
# #
# #     def listen(self):
# #         return input("Simulated STT input: ")
# #
# #     def listen_confirmation(self):
# #         resp = input("Approve plan? (y/n): ")
# #         return resp.lower() in ['y','yes']
# #
# #     def speak(self, text):
# #         print(f"Jarvis says: {text}")
#
#
#
# # class Observer:
# #     def __init__(self):
# #         self.stt = Ears()
# #         self.tts = Mouth()
# #         print("Observer initialized.")
# #
# #     def listen(self):
# #         # return input("Simulated STT input: ")
# #
# #         audio_path = input("Simulated audio path or text: ")
# #         # For real mic: integrate PyAudio
# #         return self.stt.transcribe(audio_path)
# #
# #     def listen_confirmation(self):
# #         resp = input("Approve plan? (y/n): ")
# #         return resp.lower() in ['y','yes']
# #
# #     def speak(self, text):
# #         print(f"Jarvis says: {text}")
# #
# #         self.tts.speak(text)
# #
# #         engine = pyttsx3.init()
# #
# #         # Jarvis-like tuning
# #         engine.setProperty("rate", 175)
# #         engine.setProperty("volume", 1.0)
# #
# #         voices = engine.getProperty("voices")
# #         engine.setProperty("voice", voices[0].id)  # try 0 or 1
# #
# #         engine.say(f"Systems online. {text}, correct?")
# #         engine.runAndWait()
