# Moved to hybrid_stt.py


# from modules.stt.faster_whisper_stt import FasterWhisperSTT
# from modules.stt.whisper_stt import WhisperSTT
# from .factory import create_stt
#
#
# def create_stt(config):
#     backend = config["stt"]["backend"]
#
#     if backend == "faster_whisper":
#         return FasterWhisperSTT(config)
#
#     if backend == "whisper":
#         return WhisperSTT(config)
#
#     raise ValueError(f"Unknown STT backend: {backend}")
