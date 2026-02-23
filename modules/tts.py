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

    def speak(self, text):
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

# This works too, just a different option

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
#         if self.use_mock:
#             print("[TTS] Using mock TTS.")
#             self.tts = self
#         else:
#             print(f"[TTS] Loading model: {model_name}")
#             self.tts = TTS(model_name, progress_bar=False, gpu=False)
#             print("[TTS] English TTS model loaded.")
#
#     def speak(self, text, play_audio=True):
#         text = text.strip()
#         if not text:
#             return
#
#         print(f"[TTS] {text}")
#         if self.use_mock:
#             return
#
#         if os.path.exists(self.audio_file):
#             os.remove(self.audio_file)
#
#         self.tts.tts_to_file(text=text, file_path=self.audio_file)
#
#         if play_audio:
#             try:
#                 wave_obj = sa.WaveObject.from_wave_file(self.audio_file)
#                 play_obj = wave_obj.play()
#                 play_obj.wait_done()
#             except Exception as e:
#                 print(f"[TTS] Could not play audio: {e}")


















# class TTSModule:
#     def __init__(self, use_mock=False, model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False):
#         self.use_mock = use_mock
#         self.audio_file = "tts_output.wav"
#
#         if self.use_mock:
#             print("[TTS] Using mock TTS.")
#             self.tts = self
#         else:
#             print(f"[TTS] Loading model: {model_name} (GPU={gpu})")
#             self.tts = TTS(model_name, progress_bar=False, gpu=gpu)
#             print("[TTS] English TTS model loaded.")
#
#     def speak(self, text, play_audio=True):
#         text = text.strip()
#         if not text:
#             return
#
#         print(f"[TTS] {text}")
#         if self.use_mock:
#             return
#
#         if os.path.exists(self.audio_file):
#             os.remove(self.audio_file)
#
#         self.tts.tts_to_file(text=text, file_path=self.audio_file)
#
#         if play_audio:
#             try:
#                 wave_obj = sa.WaveObject.from_wave_file(self.audio_file)
#                 play_obj = wave_obj.play()
#                 play_obj.wait_done()
#             except Exception as e:
#                 print(f"[TTS] Could not play audio: {e}")



# class TTSModule:
#     def __init__(self, use_mock=False, model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False):
#         self.use_mock = use_mock
#         self.audio_file = "tts_output.wav"
#
#         if self.use_mock:
#             print("[TTS] Using mock TTS.")
#             self.tts = self
#         else:
#             print(f"[TTS] Loading model: {model_name} (GPU={gpu})")
#             self.tts = TTS(model_name, progress_bar=False, gpu=gpu)
#             print("[TTS] English TTS model loaded.")
#
#     def speak(self, text, play_audio=True):
#         text = text.strip()
#         if not text:
#             return
#
#         print(f"[TTS] {text}")
#         if self.use_mock:
#             return
#
#         if os.path.exists(self.audio_file):
#             os.remove(self.audio_file)
#
#         self.tts.tts_to_file(text=text, file_path=self.audio_file)
#
#         if play_audio:
#             try:
#                 wave_obj = sa.WaveObject.from_wave_file(self.audio_file)
#                 play_obj = wave_obj.play()
#                 play_obj.wait_done()
#             except Exception as e:
#                 print(f"[TTS] Could not play audio: {e}")
#


# from TTS.api import TTS
# import simpleaudio as sa
# import os
#
#
# class TTSModule:
#     def __init__(self, use_mock=False, model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False):
#         self.use_mock = use_mock
#         self.audio_file = "tts_output.wav"
#
#         if self.use_mock:
#             print("[TTS] Using mock TTS.")
#             self.tts = self
#         else:
#             print(f"[TTS] Loading model: {model_name} (GPU={gpu})")
#             self.tts = TTS(model_name, progress_bar=False, gpu=gpu)
#             print("[TTS] English TTS model loaded.")
#
#     def speak(self, text, play_audio=True):
#         print(f"[TTS] {text}")
#         if self.use_mock:
#             return
#
#         if os.path.exists(self.audio_file):
#             os.remove(self.audio_file)
#
#         self.tts.tts_to_file(text=text, file_path=self.audio_file)
#
#         if play_audio:
#             try:
#                 wave_obj = sa.WaveObject.from_wave_file(self.audio_file)
#                 play_obj = wave_obj.play()
#                 play_obj.wait_done()
#             except Exception as e:
#                 print(f"[TTS] Could not play audio: {e}")
#



# class TTSModule:
#     def __init__(self, use_mock=False, model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False):  # model_name="tts_models/en/vctk/vits"):
#         """
#         Mouth class for text-to-speech
#         - use_mock: if True, prints text instead of generating audio
#         - model_name: Coqui TTS model
#         """
#         self.use_mock = use_mock
#         self.audio_file = "tts_output.wav"
#
#         if use_mock:
#             print("[Mouth] Using mock TTS.")
#             self.tts = self
#         else:
#             print(f"[Mouth] Loading TTS model: {model_name}")
#             self.tts = TTS(model_name)
#             print("[Mouth] English TTS model loaded.")
#
#     def tts_to_file(self, text):
#         # Remove existing file to avoid conflicts
#         if os.path.exists(self.audio_file):
#             os.remove(self.audio_file)
#
#         if self.use_mock:
#             print(f"[MOCK TTS] '{text}' -> {self.audio_file}")
#             with open(self.audio_file, "w") as f:
#                 f.write(f"MOCK: {text}")
#         else:
#             self.tts.tts_to_file(text=text, file_path=self.audio_file)
#
#     def speak(self, text, play_audio=True):
#         self.tts.tts_to_file(text=text, file_path="tts_output.wav")
#         print(f"[TTS] {text}")
#         if play_audio:
#             try:
#                 wave_obj = sa.WaveObject.from_wave_file("tts_output.wav")
#                 play_obj = wave_obj.play()
#                 play_obj.wait_done()
#             except Exception as e:
#                 print(f"[Mouth] Could not play audio: {e}")

    #     print("Loading TTS model...")
    #     self.use_mock = use_mock
    #
    #     if use_mock:
    #         print("[Mouth] Using mock TTS. No audio will be generated.")
    #         self.tts = self
    #     else:
    #         print(f"[Mouth] Loading real TTS model: {model_name} ...")
    #         self.tts = TTS(model_name)
    #         print("[Mouth] Real English TTS model loaded.")
    #
    # def tts_to_file(self, text, file_path="tts_output.wav"):
    #     if self.use_mock:
    #         print(f"[MOCK TTS] Writing '{text}' to {file_path}")
    #         with open(file_path, "w") as f:
    #             f.write(f"MOCK: {text}")
    #     else:
    #         self.tts.tts_to_file(text=text, file_path=file_path)
    #
    # def speak(self, text, play_audio=False):
    #     print(f"[TTS] {text}")
    #     audio_file = "tts_output.wav"
    #     if os.path.exists(audio_file):
    #         os.remove(audio_file)
    #
    #     self.tts.tts_to_file(text=text, file_path=audio_file)
    #
    #     if play_audio and os.path.exists(audio_file) and not self.use_mock:
    #         wave_obj = sa.WaveObject.from_wave_file(audio_file)
    #         play_obj = wave_obj.play()
    #         play_obj.wait_done()

    # def speak(self, text):
    #     self.tts.tts_to_file(text=text, file_path="tts_output.wav")
    #     # Optional: auto-play
    #     wave_obj = sa.WaveObject.from_wave_file("tts_output.wav")
    #     play_obj = wave_obj.play()
    #     play_obj.wait_done()



# class TTSModule:
#     def __init__(self):
#         print("TTS mock initialized.")
#
#     def speak(self, text):
#         print(f"[TTS] {text}")



# from TTS.api import TTS
# import simpleaudio as sa
#
# class Mouth:
#     def __init__(self):
#         print("Loading TTS model...")
#         self.tts = TTS(model_name)
#         print("TTS mock initialized.")
#
#     def speak(self, text):
#         print(f"[TTS] {text}")
#         self.tts.tts_to_file(text=text, file_path="tts_output.wav")
#         # Optional: auto-play
#         # wave_obj = sa.WaveObject.from_wave_file("tts_output.wav")
#         # play_obj = wave_obj.play()
#         # play_obj.wait_done()
