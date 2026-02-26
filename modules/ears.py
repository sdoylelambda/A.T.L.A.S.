import asyncio
import numpy as np
import time
import pyaudio
import os


class Ears:
    """Simple microphone listener that stops shortly after speech ends."""

    def __init__(self, chunk_size=1024, rate=48000, debug=False,
                 pre_speech_timeout=3.0, max_speech_duration=25.0,
                 silence_seconds=1.5):
        self.audio_stream = None
        self.chunk_size = chunk_size
        self.rate = rate
        self.paused = False
        self.debug = debug
        self.speaking = False

        # dynamic — set by calibration
        self.start_threshold = 32767  # max possible value — won't trigger until calibrated
        self.stop_threshold = 16000
        self.noise_floor = None
        self._stream_lock = asyncio.Lock()

        # speech confirmation — prevents AC spikes triggering false starts
        self.speech_confirm_chunks = 3

        self.pre_speech_timeout = pre_speech_timeout
        self.max_speech_duration = max_speech_duration
        self.silence_seconds = silence_seconds
        self.hangover_chunks = 12

    def _find_mic(self, p):
        """Find the first real analog mic input, falling back to pulse."""
        blacklist = ('hdmi', 'dell', 'displayport', 'default',
                     'surround', 'upmix', 'vdown')

        # first pass — look for real analog mic
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name_lower = info['name'].lower()
            if info['maxInputChannels'] == 0:
                continue
            if info['maxInputChannels'] > 8:
                continue
            if any(bad in name_lower for bad in blacklist):
                continue
            print(f"[Ears] Selected mic: [{i}] {info['name']}")
            return i, int(info['defaultSampleRate'])

        # fallback — use pulse (works with PipeWire)
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if 'pulse' in info['name'].lower() and info['maxInputChannels'] > 0:
                print(f"[Ears] Falling back to pulse device: [{i}] {info['name']}")
                return i, int(info['defaultSampleRate'])

        raise RuntimeError("No suitable microphone found")

    async def _ensure_stream(self):
        # suppress ALSA/Jack spam
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)

        try:
            p = pyaudio.PyAudio()  # single instance

            # verify existing stream is healthy
            if self.audio_stream is not None:
                try:
                    self.audio_stream.get_read_available()
                except OSError:
                    print("[Ears] Stream unhealthy, reopening...")
                    try:
                        self.audio_stream.close()
                    except Exception:
                        pass
                    self.audio_stream = None

            # open new stream if needed
            if self.audio_stream is None:
                mic_index, self.rate = self._find_mic(p)
                self.audio_stream = await asyncio.to_thread(
                    p.open,
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.rate,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=self.chunk_size,
                )
                if self.noise_floor is None:
                    await self._calibrate_noise_floor(seconds=2.0, pre_delay=1)

        finally:
            # always restore stderr even if something crashes
            os.dup2(old_stderr, 2)
            os.close(old_stderr)

    async def _calibrate_noise_floor(self, seconds=1.0, pre_delay=1):
        await asyncio.sleep(pre_delay)
        samples = []
        start = time.time()

        async with self._stream_lock:
            while time.time() - start < seconds:
                try:
                    data = await asyncio.to_thread(
                        self.audio_stream.read,
                        self.chunk_size,
                        exception_on_overflow=False
                    )
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
                    samples.append(rms)
                except OSError as e:
                    print(f"[Ears] Calibration read error: {e}")
                    return  # bail out early, don't update thresholds

        if not samples:
            print("[Ears] Calibration got no samples, skipping threshold update")
            return

        # use 90th percentile instead of mean — handles AC spikes better
        self.noise_floor = float(np.percentile(samples, 90))

        if self.noise_floor > 5000:
            print(f"[Ears] WARNING: Noise floor too high ({int(self.noise_floor)}), using static thresholds")
            self.start_threshold = 8000
            self.stop_threshold = 4000
        else:
            self.start_threshold = self.noise_floor * 4.0
            self.stop_threshold = self.noise_floor * 2.0

        print(f"[Ears] Noise floor={int(self.noise_floor)} "
              f"start={int(self.start_threshold)} "
              f"stop={int(self.stop_threshold)}")

    async def auto_calibrate(self, interval: int = 20):
        """Recalibrate noise floor every interval seconds, only during silence."""
        while True:
            await asyncio.sleep(interval)
            try:
                if not self.paused and self.audio_stream and not self.speaking:
                    if self.debug:
                        print("[Ears] Recalibrating noise floor...")
                    await self._calibrate_noise_floor(seconds=2.0, pre_delay=0)
            except OSError as e:
                print(f"[Ears] Calibration failed: {e}, will retry next interval")
                # reset stream so it gets reopened on next listen
                self.audio_stream = None

    async def listen(self, max_duration=30.0):
        if self.debug:
            print(f"[Ears] Thresholds: start={int(self.start_threshold)} stop={int(self.stop_threshold)}")
        # retry up to 3 times with backoff
        for attempt in range(3):
            try:
                await self._ensure_stream()
                break
            except OSError as e:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[Ears] Stream open failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                self.audio_stream = None
                await asyncio.sleep(wait)
        else:
            print("[Ears] Could not open stream after 3 attempts")
            return None, 0.0

        frames = []
        speech_started = False
        silence_chunks = 0
        speech_start_time = None
        consecutive_loud = 0

        # from config params
        max_silence_chunks = int(self.silence_seconds * self.rate / self.chunk_size)
        start_time = time.time()

        while True:
            if self.paused:
                await asyncio.sleep(0.05)
                continue

            try:
                async with self._stream_lock:
                    data = await asyncio.to_thread(
                        self.audio_stream.read,
                        self.chunk_size,
                        exception_on_overflow=False
                    )
            except OSError as e:
                print(f"[Ears] Stream read error: {e}, resetting...")
                self.audio_stream = None
                self.speaking = False
                return None, 0.0

            audio_np = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))

            if self.debug:
                print(f"[Ears RMS] {int(rms)} speech={speech_started}")

            # ---- waiting for speech to start ----
            if not speech_started:
                if time.time() - start_time > self.pre_speech_timeout:
                    return None, 0.0
                if rms >= self.start_threshold:
                    consecutive_loud += 1
                    if consecutive_loud >= self.speech_confirm_chunks:
                        speech_started = True
                        speech_start_time = time.time()
                        self.speaking = True
                        frames.append(data)
                else:
                    consecutive_loud = 0
                continue

            # ---- recording after speech starts ----
            frames.append(data)

            if rms < self.stop_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0  # reset on any sound — keeps recording through pauses

            # stop after sustained silence
            if silence_chunks > max_silence_chunks:
                self.speaking = False
                break

            # hard cap — prevents runaway recording
            if time.time() - speech_start_time > self.max_speech_duration:
                print("[Ears] Max speech duration reached.")
                self.speaking = False
                break

        if not frames:
            self.speaking = False
            return None, 0.0

        audio_bytes = b"".join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        resample_ratio = 16000 / self.rate
        new_length = int(len(audio_np) * resample_ratio)
        audio_resampled = np.interp(
            np.linspace(0, len(audio_np), new_length),
            np.arange(len(audio_np)),
            audio_np
        ).astype(np.int16)

        audio_bytes = audio_resampled.tobytes()
        duration = len(audio_bytes) / 2 / 16000

        return audio_bytes, duration
