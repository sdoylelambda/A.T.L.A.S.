import wave
import simpleaudio as sa
from piper.voice import PiperVoice
import io
import numpy as np
from piper.config import SynthesisConfig


class Mouth:
    def __init__(self, model_path="modules/voices/british_man_GB-alan-medium.onnx",  # add voices to config to switch
                 speech_rate=0.8, use_mock=False):
        self.use_mock = use_mock
        self.speech_rate = speech_rate
        self._current_play = None
        if not use_mock:
            self.voice = PiperVoice.load(model_path)
            self.syn_config = SynthesisConfig(
                length_scale=speech_rate,  # speed
                noise_scale=0.5,  # variation in tone (lower = more consistent)
                noise_w_scale=0.2,  # variation in timing (lower = more consistent)
            )

    async def speak(self, text):
        if self.use_mock:
            print(f"[TTS] {text}")
            return

        # print(f"[TTS] {text}")
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.voice.config.sample_rate)
                self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)

            buf.seek(0)
            audio_data = np.frombuffer(buf.read(), dtype=np.int16)
            self._current_play = sa.play_buffer(
                audio_data, 1, 2, self.voice.config.sample_rate
            )
            self._current_play.wait_done()
        except Exception as e:
            # don't raise if we were intentionally stopped
            if self._current_play is None:
                return  # stop() was called, exit silently
            print(f"[TTS] Playback failed: {e}")
            raise
        finally:
            self._current_play = None

    def stop(self):
        """Stop current playback immediately."""
        if self._current_play is not None:
            try:
                self._current_play.stop()
            except Exception:
                pass
            self._current_play = None
