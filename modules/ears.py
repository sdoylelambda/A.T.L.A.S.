import asyncio
import numpy as np
import time
import pyaudio
import os


class Ears:
    """Simple microphone listener that stops shortly after speech ends."""
    def __init__(self, chunk_size=1024, rate=48000, debug=False):
        self.audio_stream = None
        self.chunk_size = chunk_size
        self.rate = rate
        self.paused = False
        self.debug = debug
        self.speaking = False

        # Speech detection params
        self.start_threshold = 1200     # speech start RMS
        self.stop_threshold = 700       # speech end RMS
        self.hangover_chunks = 12       # ~0.8s at 1024/16kHz
        self.noise_floor = None

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
        # suppress ALSA spam
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        p = pyaudio.PyAudio()
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

        if self.audio_stream is not None:
            # verify stream is still healthy
            try:
                self.audio_stream.get_read_available()
            except OSError:
                print("[Ears] Stream unhealthy, reopening...")
                try:
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None

        if self.audio_stream is None:
            p = pyaudio.PyAudio()
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
                await self._calibrate_noise_floor()

    async def _calibrate_noise_floor(self, seconds=1.0, pre_delay=1):
        await asyncio.sleep(pre_delay)  # let TTS echo settle
        samples = []
        start = time.time()

        while time.time() - start < seconds:
            data = await asyncio.to_thread(
                self.audio_stream.read,
                self.chunk_size,
                exception_on_overflow=False
            )
            audio_np = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
            samples.append(rms)

        self.noise_floor = float(np.mean(samples))

        # If noise floor is suspiciously high, fall back to static values
        if self.noise_floor > 5000:
            print(f"[Ears] WARNING: Noise floor too high ({int(self.noise_floor)}), using static thresholds")
            self.start_threshold = 1500
            self.stop_threshold = 800
        else:
            self.start_threshold = self.noise_floor * 3.0
            self.stop_threshold = self.noise_floor * 1.5

        print(f"[Ears] Noise floor={int(self.noise_floor)} "
              f"start={int(self.start_threshold)} "
              f"stop={int(self.stop_threshold)}")

    async def auto_calibrate(self, interval: int = 30):
        """Recalibrate noise floor every interval seconds, only during silence."""
        while True:
            await asyncio.sleep(interval)
            try:
                if not self.paused and self.audio_stream and not self.speaking:
                    if self.debug:
                        print("[Ears] Recalibrating noise floor...")
                    await self._calibrate_noise_floor(seconds=0.5, pre_delay=0)
            except OSError as e:
                print(f"[Ears] Calibration failed: {e}, will retry next interval")
                # reset stream so it gets reopened on next listen
                self.audio_stream = None


    async def listen(self, max_duration=30.0):
        """
        Record until speech ends (RMS-based).
        Returns (audio_bytes, duration_seconds)
        """
        try:
            await self._ensure_stream()
        except OSError as e:
            print(f"[Ears] Failed to open stream: {e}")
            self.audio_stream = None
            return None, 0.0

        frames = []
        speech_started = False
        silence_chunks = 0
        max_silence_chunks = int(0.5 * self.rate / self.chunk_size)
        start_time = time.time()

        while True:
            if self.paused:
                await asyncio.sleep(0.05)
                continue

            try:
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

            # ---- speech start detection ----
            if not speech_started:
                if rms >= self.start_threshold:
                    speech_started = True
                    self.speaking = True
                    frames.append(data)
                continue

            # ---- recording after speech start ----
            frames.append(data)

            # silence detection
            if rms < self.stop_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0

            # stop after sustained silence
            if silence_chunks > max_silence_chunks:
                self.speaking = False
                break

            # safety max duration
            if time.time() - start_time > max_duration:
                self.speaking = False
                break

        if not frames:
            self.speaking = False
            return None, 0.0

        audio_bytes = b"".join(frames)

        # Resample from mic rate to Whisper's required 16000
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        resample_ratio = 16000 / self.rate
        new_length = int(len(audio_np) * resample_ratio)
        audio_resampled = np.interp(
            np.linspace(0, len(audio_np), new_length),
            np.arange(len(audio_np)),
            audio_np
        ).astype(np.int16)

        audio_bytes = audio_resampled.tobytes()
        duration = len(audio_bytes) / 2 / 16000  # now measured at 16k

        return audio_bytes, duration
