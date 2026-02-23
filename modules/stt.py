# This logic is now in ears.py



# import whisper
# import os
# import speech_recognition as sr  # Remove this if possible - google api - perhaps whisper instead
# import sounddevice as sd
# from scipy.io.wavfile import write
#
#
# class STT:
#     def __init__(self, use_mock=False, mic_index=0, samplerate=16000, duration=15):
#         """
#         Speech-to-Text class
#         - use_mock: True uses input() instead of microphone
#         - mic_index: optional, select a specific microphone
#         - samplerate: recording sample rate
#         - duration: default recording duration
#         """
#         device = "cpu"  # force CPU usage
#         self.use_mock = use_mock
#         self.duration = duration
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#
#         if not use_mock:
#             print("[Ears] Loading Whisper model...")
#             self.model = whisper.load_model("small", device=device)  # options: tiny, base, small, medium, large
#             print("[Ears] Whisper model loaded.")
#
#             self.recognizer = sr.Recognizer()
#             self.microphone = self._find_microphone(mic_index)
#
#     def _find_microphone(self, mic_index):
#         """Automatically find a working microphone if not specified."""
#         mic_list = sr.Microphone.list_microphone_names()
#         if mic_index is not None:
#             if mic_index < len(mic_list):
#                 print(f"[Ears] Using mic index {mic_index}: {mic_list[mic_index]}")
#                 return sr.Microphone(device_index=mic_index)
#             else:
#                 print("[Ears] Mic index out of range, falling back to default.")
#
#         # fallback: pick the first working mic
#         for i, name in enumerate(mic_list):
#             try:
#                 with sr.Microphone(device_index=i) as source:
#                     print(f"[Ears] Found working mic {i}: {name}")
#                     return sr.Microphone(device_index=i)
#             except OSError:
#                 continue
#
#         raise RuntimeError("No working microphone found.")
#
#     def ears(self, duration=None, filename="capture.wav"):
#         """Record audio from microphone and return the transcribed text."""
#         if self.use_mock:
#             return input("Simulated audio: ")
#
#         duration = duration or self.duration
#         print("[Ears] Listening... Please speak now.")
#
#         # Record audio
#         audio = sd.rec(int(duration * self.samplerate),
#                        samplerate=self.samplerate,
#                        channels=1,
#                        dtype='int16',
#                        device=self.mic_index)
#         sd.wait()
#         write(filename, self.samplerate, audio)
#
#         # Transcribe with Whisper
#         return self.transcribe(filename)
#
#     def transcribe(self, audio_path):
#         """Convert audio file to text using Whisper."""
#         print(f"[Ears] Transcribing {audio_path}...")
#         result = self.model.transcribe(audio_path)
#         text = result.get("text", "").strip()
#         if not text:
#             print("[Ears] Could not understand audio.")
#         else:
#             print(f"[Ears] Heard: {text}")
#         return text
#









# import whisper
# import os
# import speech_recognition as sr
# import sounddevice as sd
# from scipy.io.wavfile import write
#
#
# class STT:
#     def __init__(self, use_mock=False, mic_index=0, samplerate=44100):
#         """
#         Speech-to-Text class
#         - use_mock: True uses input() instead of microphone
#         - mic_index: optional, select a specific microphone
#         """
#         self.use_mock = use_mock
#         if not use_mock:
#             self.recognizer = sr.Recognizer()
#             self.mic_index = mic_index
#             self.microphone = self._find_microphone(mic_index)
#             self.samplerate = samplerate
#
#     def _find_microphone(self, mic_index):
#         """Automatically find a working microphone if not specified."""
#         mic_list = sr.Microphone.list_microphone_names()
#         if mic_index is not None:
#             if mic_index < len(mic_list):
#                 print(f"[Ears] Using mic index {mic_index}: {mic_list[mic_index]}")
#                 return sr.Microphone(device_index=mic_index)
#             else:
#                 print("[Ears] Mic index out of range, falling back to default.")
#         # fallback: pick the first working mic
#         for i, name in enumerate(mic_list):
#             try:
#                 with sr.Microphone(device_index=i) as source:
#                     print(f"[Ears] Found working mic {i}: {name}")
#                     return sr.Microphone(device_index=i)
#             except OSError:
#                 continue
#         raise RuntimeError("No working microphone found.")
#
#     def transcribe(self, audio_path=None):
#         # If you pass a file, read it; else listen via mic
#         if audio_path:
#             # implement file-based STT if needed
#             print('transcribing audio...')
#             return True  # temp return - update to use
#         else:
#             return self.ears()
#
#     def ears(self, duration=5, filename="capture.wav"):
#         if self.use_mock:
#             return input("Simulated audio: ")
#         print("[Ears] Listening...")
#         audio = sd.rec(int(duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='int16',
#                        device=self.mic_index)
#         sd.wait()
#         write(filename, self.samplerate, audio)
#         return filename  # return file path to transcribe later

    # def ears(self):
    #     if self.use_mock:
    #         return input("Simulated audio path or text: ")
    #     # listen from real mic
    #     with self.microphone as source:
    #         print("[Ears] Listening... Please speak now.")
    #         self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
    #         try:
    #             audio = self.recognizer.listen(source, timeout=5)
    #             text = self.recognizer.recognize_google(audio)
    #             print(f"[Ears] Heard: {text}")
    #             return text
    #         except sr.WaitTimeoutError:
    #             print("[Ears] Nothing heard (timeout).")
    #             return None
    #         except sr.UnknownValueError:
    #             print("[Ears] Could not understand audio.")
    #             return None
    #         except sr.RequestError as e:
    #             print(f"[Ears] API error: {e}")
    #             return None

    # def ears(self):
    #     if self.use_mock:
    #         return input("Simulated audio: ")
    #     try:
    #         with sr.Microphone(device_index=self.mic_index) as source:
    #             print("[Ears] Listening... Please speak now.")
    #             self.recognizer.adjust_for_ambient_noise(source, duration=1)
    #             audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
    #         text = self.recognizer.recognize_google(audio)
    #         print(f"[Ears] You said: {text}")
    #         return text
    #     except sr.WaitTimeoutError:
    #         print("[Ears] Timeout, no speech detected.")
    #         return None
    #     except sr.UnknownValueError:
    #         print("[Ears] Could not understand audio.")
    #         return None
    #     except sr.RequestError as e:
    #         print(f"[Ears] API request failed: {e}")
    #         return None

    # def ears(self):
    #     if self.use_mock:
    #         return input("Simulated audio: ")
    #
    #     with sr.Microphone(device_index=self.mic_index) as source:
    #         print("Listening... Please speak now.")
    #         self.recognizer.adjust_for_ambient_noise(source, duration=1)
    #         try:
    #             audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
    #             text = self.recognizer.recognize_google(audio)
    #             print("You said:", text)
    #             return text
    #         except sr.WaitTimeoutError:
    #             print("[Ears] No speech detected.")
    #             return ""
    #         except sr.UnknownValueError:
    #             print("[Ears] Could not understand audio")
    #             return ""
    #         except sr.RequestError as e:
    #             print(f"[Ears] API request error: {e}")
    #             return ""

        # r = sr.Recognizer()
        # try:
        #     with sr.Microphone(device_index=self.mic_index) as source:
        #         print("[Ears] Listening... Please speak now.")
        #         r.adjust_for_ambient_noise(source, duration=1)  # longer to calibrate
        #         audio = r.listen(source, timeout=10, phrase_time_limit=5)
        #         text = r.recognize_google(audio)
        #         return text
        # except sr.UnknownValueError:
        #     return "[Ears] Could not understand audio"
        # except sr.RequestError as e:
        #     return f"[Ears] API request error: {e}"
        # except OSError as e:
        #     print(f"[Ears] Microphone error: {e}. Falling back to default mic.")
        #     with sr.Microphone() as source:
        #         r.adjust_for_ambient_noise(source, duration=0.5)
        #         audio = r.listen(source, timeout=5)
        #         try:
        #             return r.recognize_google(audio)
        #         except:
        #             return "[Ears] Could not understand audio"



# class Ears:
#     def __init__(self):
#         print("STT mock initialized.")
#         # self.model = whisper.load_model(model_name)
#
#     def transcribe(self, audio_path=None):
#         return input("Simulated audio transcription: ")
