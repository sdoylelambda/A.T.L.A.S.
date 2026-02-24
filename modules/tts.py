import wave
import simpleaudio as sa
from piper.voice import PiperVoice
import io
import numpy as np


class TTSModule:
    def __init__(self, model_path="en_US-lessac-medium.onnx", use_mock=False):
        self.use_mock = use_mock
        if not use_mock:
            self.voice = PiperVoice.load(model_path)

    async def speak(self, text):
        print(f"[TTS] {text}")
        try:
            # Synthesize
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.voice.config.sample_rate)
                self.voice.synthesize_wav(text, wav_file)

            # Play directly from buffer via simpleaudio
            buf.seek(0)
            audio_data = np.frombuffer(buf.read(), dtype=np.int16)
            play_obj = sa.play_buffer(audio_data, 1, 2, self.voice.config.sample_rate)
            play_obj.wait_done()
        except Exception as e:
            print(f"[TTS] Playback failed: {e}")
            raise

# This works too, just a different option using tacotron2 instead

# from TTS.api import TTS
# import simpleaudio as sa
# import os
#
#
# class TTSModule:
#     def __init__(self, use_mock=False, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
#         self.use_mock = use_mock
#         self.audio_file = "tts_output.wav"
#
#         if use_mock:
#             print("[TTS] Mock mode active.")
#             self.tts = self
#         else:
#             print(f"[TTS] Loading model {model_name}...")
#             self.tts = TTS(model_name)
#             print("[TTS] Model loaded.")
#
#     def tts_to_file(self, text):
#         if os.path.exists(self.audio_file):
#             os.remove(self.audio_file)
#         if self.use_mock:
#             print(f"[MOCK TTS] {text}")
#             with open(self.audio_file, "w") as f:
#                 f.write(f"MOCK: {text}")
#         else:
#             self.tts.tts_to_file(text=text, file_path=self.audio_file)
#
#     def speak(self, text, play_audio=True):
#         self.tts_to_file(text)
#         print(f"[TTS] {text}")
#         if play_audio and not self.use_mock:
#             try:
#                 wave_obj = sa.WaveObject.from_wave_file(self.audio_file)
#                 play_obj = wave_obj.play()
#                 play_obj.wait_done()
#             except Exception as e:
#                 print(f"[TTS] Playback failed: {e}")
