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
