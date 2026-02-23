# import os
# import sounddevice as sd
# import numpy as np
# import simpleaudio as sa
# from faster_whisper import WhisperModel
# from TTS.api import TTS
# import yaml
#
#
#
# class FasterWhisperSTT:
#     def __init__(self, config):
#         self.samplerate = 16000
#         self.duration = config["stt"]["duration"]
#         self.mic_index = config["audio"].get("mic_index", 0)
#
#         cpu_threads = config["system"]["cpu_threads"]
#         os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
#
#         device = "cuda" if config["system"].get("use_gpu", False) else "cpu"
#
#         print(f"[STT] Loading Faster Whisper model '{config['stt']['model']}' on {device}...")
#         self.model = WhisperModel(
#             model_size_or_path=config["stt"]["model"],
#             device=device,
#             compute_type="int8",
#             cpu_threads=cpu_threads
#         )
#
#         print("[STT] Faster Whisper loaded.")
#
#     def transcribe(self, audio_array):
#         if audio_array.ndim > 1:
#             audio_array = audio_array.mean(axis=1)  # convert to mono
#
#         audio_float = audio_array.astype(np.float32) / 32768.0  # int16 -> float32
#         segments, _ = self.model.transcribe(
#             audio_float,
#             language="en",
#             temperature=0.0,  # deterministic output
#             suppress_blank=True,  # avoid extra blanks
#             word_timestamps=False,  # we don’t need timestamps
#             max_new_tokens=400,  # limit hallucination - less than 500 - smaller = fewer hallucinations
#             beam_size=1  # turn up to make more accurate at expense of speed. 1-5
#         )
#         print("[STT] Faster Whisper Used.")
#         return " ".join([seg.text for seg in segments])


# import os
# from faster_whisper import WhisperModel
#
# class FasterWhisperSTT:
#     def __init__(self, config):
#         self.duration = config["stt"]["duration"]
#         self.samplerate = 16000
#         self.mic_index = 0
#
#         cpu_threads = config["system"]["cpu_threads"]
#         os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
#
#         # Correct usage of WhisperModel:
#         self.model = WhisperModel(
#             model_size_or_path=config["stt"]["model"],  # ONLY positional argument
#             device="cpu",                               # or "cuda" if GPU
#             compute_type="int8",
#             cpu_threads=cpu_threads
#         )
#
#     def transcribe(self, audio_file):
#         segments, _ = self.model.transcribe(audio_file)
#         return " ".join([seg.text for seg in segments])


# from faster_whisper import WhisperModel
# from .base import BaseSTT
# import sounddevice as sd
# import numpy as np
# import os
#
#
# class FasterWhisperSTT:
#     def __init__(self, config):
#         model_name = config["transcription"]["engine"]
#         device = config["system"]["cpu_threads"]
#         self.duration = config["stt"]["duration"]
#         self.samplerate = 16000
#         self.mic_index = 0
#
#         os.environ["OMP_NUM_THREADS"] = str(config["system"]["cpu_threads"])
#
#         self.model = WhisperModel(
#             config["stt"]["model"],
#             model_name,
#             device=device,
#             compute_type="int8"
#         )
#
#     def transcribe(self, audio_bytes):
#         segments, info = self.model.transcribe(
#             audio_bytes,
#             vad_filter=True
#         )
#         return " ".join(seg.text for seg in segments)
#
#     def listen(self):
#         audio = sd.rec(
#             int(self.duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="float32"
#         )
#         sd.wait()
#
#         audio = audio.flatten()
#         if np.abs(audio).mean() < 0.01:
#             return None
#
#         segments, _ = self.model.transcribe(
#             audio,
#             language="en",
#             beam_size=1,
#             temperature=0.0
#         )
#
#         text = " ".join(s.text for s in segments).strip()
#         return text or None
