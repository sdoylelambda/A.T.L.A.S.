# import whisper
# from .base import BaseSTT
# import sounddevice as sd
# import numpy as np
# import torch
#
#
# class WhisperSTT:
#     """Wrapper for OpenAI's whisper"""
#     def __init__(self, config):
#         self.duration = config["stt"]["duration"]
#         self.samplerate = 16000
#         self.mic_index = 0
#
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = whisper.load_model(
#         config["stt"]["model"],
#         device=device
#         )
#
#     def transcribe(self, audio_bytes):
#         audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#
#         if len(audio_np) < 16000 * 0.5:  # less than 0.5 seconds
#             print("[STT] Audio too short, skipping.")
#             return ""
#
#         result = self.model.transcribe(audio_np, fp16=False)
#         print("[STT] Whisper Used.")
#         return result["text"]

    # def transcribe(self, audio_file):
    #     result = self.model.transcribe(audio_file)
    #     print("[STT] Whisper Used.")
    #     return result["text"]


#
# class WhisperSTT(BaseSTT):
#     def __init__(self, config):
#         model_name = config["transcription"]["model"]
#         self.model = whisper.load_model(model_name)
#
#         # device = "cuda" if torch.cuda.is_available() else "cpu"
#         # self.model = whisper.load_model(
#         #     config["stt"]["model"],
#         #     device=device
#         # )
#         # self.duration = config["stt"]["duration"]
#         # self.samplerate = 16000
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
#         result = self.model.transcribe(
#             audio,
#             language="en",
#             fp16=torch.cuda.is_available()
#         )
#         return result["text"].strip() or None
#
#     def transcribe(self, audio_bytes):
#         result = self.model.transcribe(audio_bytes)
#         return result["text"]
