import numpy as np
import whisper
from faster_whisper import WhisperModel
import time


SHORT_THRESHOLD_SECONDS = 10  # use whisper for >10s, faster-whisper for shorter

class HybridSTT:
    def __init__(self, whisper_model="small", fw_model="small", use_gpu=False):
        device = "cuda" if use_gpu else "cpu"

        print("[STT] Loading Whisper...")
        self.whisper = whisper.load_model(whisper_model, device=device)

        print("[STT] Loading Faster-Whisper...")
        self.faster = WhisperModel(
            fw_model,
            device=device,
            compute_type="int8" if not use_gpu else "float16"
        )

    def _to_float32(self, audio_bytes):
        """Convert raw int16 bytes to normalized float32 numpy array."""
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, audio_bytes, duration):
        """Route to whisper or faster-whisper based on clip duration."""
        audio_np = self._to_float32(audio_bytes)

        if duration > SHORT_THRESHOLD_SECONDS:
            print(f"[STT] Using Whisper ({duration:.1f}s)")
            return self._transcribe_short(audio_np)
        else:
            print(f"[STT] Using Faster-Whisper ({duration:.1f}s)")
            return self._transcribe_long(audio_np)

    def _transcribe_short(self, audio_np):
        transcribe_time = time.time()
        result = self.whisper.transcribe(
            audio_np,
            language="en",
            temperature=0.0,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            fp16=False
        )
        transcribe_time = time.time() - transcribe_time
        print('[STT] Transcription time: {:.1f}s'.format(transcribe_time))
        return result.get("text", "").strip()

    def _transcribe_long(self, audio_np):
        transcribe_time = time.time()
        segments, _ = self.faster.transcribe(
            audio_np,
            language="en",
            beam_size=1,
            vad_filter=True
        )
        transcribe_time = time.time() - transcribe_time
        print('[STT] Transcription time: {:.1f}s'.format(transcribe_time))
        return " ".join(seg.text for seg in segments).strip()
