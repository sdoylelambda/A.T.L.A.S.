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

        # Speech detection params
        self.start_threshold = 1200     # speech start RMS
        self.stop_threshold = 700       # speech end RMS
        self.hangover_chunks = 12       # ~0.8s at 1024/16kHz
        self.noise_floor = None

    def _find_mic(self, p):
        """Find the first real analog mic input, avoiding HDMI/display/virtual devices."""
        blacklist = ('hdmi', 'dell', 'displayport', 'default', 'pulse',
                     'pipewire', 'sysdefault', 'surround', 'upmix', 'vdown')

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name_lower = info['name'].lower()

            if info['maxInputChannels'] == 0:
                continue
            if info['maxInputChannels'] > 8:
                continue  # skip virtual/aggregate devices
            if any(bad in name_lower for bad in blacklist):
                continue

            print(f"[Ears] Selected mic: [{i}] {info['name']}")
            return i, int(info['defaultSampleRate'])

        raise RuntimeError("No suitable microphone found")

    async def _ensure_stream(self):
        # Suppress ALSA/Jack spam
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)

        p = pyaudio.PyAudio()

        # Restore stderr
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

        if self.debug:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  [{i}] {info['name']}  "
                          f"channels={info['maxInputChannels']}  "
                          f"rate={int(info['defaultSampleRate'])}")
            mic_info = p.get_device_info_by_index(0)
            print(f"[Ears] Using device: {mic_info['name']}  "
                  f"native rate={int(mic_info['defaultSampleRate'])}")

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

        # dynamic thresholds
        self.start_threshold = self.noise_floor * 3
        self.stop_threshold = self.noise_floor * 1.5

        print(f"[Ears] Noise floor={int(self.noise_floor)} "
              f"start={int(self.start_threshold)} "
              f"stop={int(self.stop_threshold)}")

    async def listen(self, max_duration=30.0):
        """
        Record until speech ends (RMS-based).
        Returns (audio_bytes, duration_seconds)
        """
        await self._ensure_stream()

        frames = []
        speech_started = False
        silence_chunks = 0
        max_silence_chunks = int(0.5 * self.rate / self.chunk_size)  # ~0.5s silence

        start_time = time.time()

        while True:
            if self.paused:
                await asyncio.sleep(0.05)
                continue

            data = await asyncio.to_thread(
                self.audio_stream.read,
                self.chunk_size,
                exception_on_overflow=False
            )

            audio_np = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))

            if self.debug:
                print(f"[Ears RMS] {int(rms)} speech={speech_started}")

            # ---- speech start detection ----
            if not speech_started:
                if rms >= self.start_threshold:
                    speech_started = True
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
                break

            # safety max duration
            if time.time() - start_time > max_duration:
                break

        if not frames:
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
