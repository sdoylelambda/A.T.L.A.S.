# Handled in hybrid_stt.py


# def create_stt(config):
#     engine = config["transcription"]["engine"]
#
#     if engine == "faster-whisper":
#         from .faster_whisper_stt import FasterWhisperSTT
#         return FasterWhisperSTT(config)
#
#     elif engine == "whisper":
#         from .whisper_stt import WhisperSTT
#         return WhisperSTT(config)
#
#     else:
#         raise ValueError(f"Unknown STT engine: {engine}")
