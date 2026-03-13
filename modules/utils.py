from contextlib import contextmanager
import time

@contextmanager
def timer(label: str, enabled: bool = True):
    if not enabled:
        yield
        return
    t0 = time.time()
    yield
    print(f"[Timing] {label}: {time.time() - t0:.2f}s")


# Usage examples

# with timer("phi3", self.debug):
#     result = self.quick_answer(command)
#
# with timer("Mistral", self.debug):
#     plan = self.create_plan(command)
#
# with timer("Gemini", self.debug):
#     response = self.query(command, model_key="gemini")
#
# from modules.utils import timer
#
# # in observer.py
# with timer("STT", self.debug):
#     text = self.stt.transcribe(audio_bytes, duration)
#
# with timer("TTS", self.debug):
#     await self.mouth.speak(text)