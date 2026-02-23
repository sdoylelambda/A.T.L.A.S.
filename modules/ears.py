import asyncio
import pyaudio


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
        p = pyaudio.PyAudio()

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

    # def __init__(self, chunk_size=1024, rate=16000, debug=True):
    #     self.audio_stream = None
    #     self.chunk_size = chunk_size
    #     self.rate = rate
    #     self.paused = False
    #     self.debug = debug
    #
    #     self.noise_floor = None
    #     self.start_threshold = None
    #     self.stop_threshold = None
    #
    # async def listen(self, max_duration=30.0, silence_duration=0.6):
    #     # Ensure thresholds are set
    #     if self.start_threshold is None or self.stop_threshold is None:
    #         raise RuntimeError(
    #             "[Ears] Noise floor not calibrated. "
    #             "Call `await ears.calibrate_noise_floor()` first.")
    #
    # async def _init_stream(self):
    #     if self.audio_stream is None:
    #         p = pyaudio.PyAudio()
    #         mic_info = p.get_default_input_device_info()
    #         self.audio_stream = await asyncio.to_thread(
    #             p.open,
    #             format=pyaudio.paInt16,
    #             channels=1,
    #             rate=self.rate,
    #             input=True,
    #             input_device_index=mic_info["index"],
    #             frames_per_buffer=self.chunk_size,
    #         )
    #
    # async def calibrate_noise_floor(self, seconds=2.0):
    #     """Measure RMS during true silence."""
    #     await self._init_stream()
    #     samples = []
    #     print("[Ears] Calibrating noise floor. Stay silent...")
    #     start = time.time()
    #     while time.time() - start < seconds:
    #         data = await asyncio.to_thread(
    #             self.audio_stream.read,
    #             self.chunk_size,
    #             exception_on_overflow=False
    #         )
    #         audio_np = np.frombuffer(data, dtype=np.int16)
    #         rms = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
    #         samples.append(rms)
    #
    #     self.noise_floor = float(np.mean(samples))
    #     self.start_threshold = self.noise_floor * 2.0  # tuned for speech start
    #     self.stop_threshold = self.noise_floor * 1.5   # tuned for speech stop
    #
    #     print(f"[Ears] Noise floor={int(self.noise_floor)}, "
    #           f"start_threshold={int(self.start_threshold)}, "
    #           f"stop_threshold={int(self.stop_threshold)}")
    #
    # async def listen(self, max_duration=30.0, silence_duration=0.6):
    #     """
    #     Record until speech ends, using RMS thresholds.
    #     Trims silence at beginning and end.
    #     Returns (audio_bytes, duration_in_seconds)
    #     """
    #     await self._init_stream()
    #
    #     frames = []
    #     speech_started = False
    #     silence_counter = 0
    #     start_time = time.time()
    #
    #     while True:
    #         if self.paused:
    #             await asyncio.sleep(0.05)
    #             continue
    #
    #         if time.time() - start_time > max_duration:
    #             break
    #
    #         data = await asyncio.to_thread(
    #             self.audio_stream.read,
    #             self.chunk_size,
    #             exception_on_overflow=False
    #         )
    #         audio_np = np.frombuffer(data, dtype=np.int16)
    #         rms = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
    #         is_speech = rms > self.start_threshold if not speech_started else rms > self.stop_threshold
    #
    #         if self.debug:
    #             print(f"[Ears RMS] {int(rms)} speech={is_speech}")
    #
    #         if is_speech:
    #             frames.append(data)
    #             speech_started = True
    #             silence_counter = 0
    #         elif speech_started:
    #             frames.append(data)
    #             silence_counter += self.chunk_size / self.rate
    #             if silence_counter >= silence_duration:
    #                 break  # stop shortly after speech ends
    #
    #     if not frames:
    #         return None, 0.0
    #
    #     # trim leading/trailing silence
    #     audio_np = np.frombuffer(b"".join(frames), dtype=np.int16)
    #     non_silent = np.where(np.abs(audio_np) > self.noise_floor * 1.2)[0]
    #     if len(non_silent) == 0:
    #         return None, 0.0
    #
    #     start_idx = non_silent[0]
    #     end_idx = non_silent[-1] + 1
    #     trimmed_audio = audio_np[start_idx:end_idx]
    #     duration = len(trimmed_audio) / self.rate
    #     return trimmed_audio.tobytes(), duration
    #
    #
    #
    #
    #














    # async def listen(self, max_duration=30.0):
    #     """
    #     Wait for speech → record → stop after silence.
    #     Returns (audio_bytes, duration_seconds)
    #     """
    #     await self._ensure_stream()
    #
    #     frames = []
    #     speech_started = False
    #     silent_chunks = 0
    #
    #     start_time = time.time()
    #
    #     while True:
    #         if self.paused:
    #             await asyncio.sleep(0.05)
    #             continue
    #
    #         # safety cap
    #         if time.time() - start_time > max_duration:
    #             break
    #
    #         data = await asyncio.to_thread(
    #             self.audio_stream.read,
    #             self.chunk_size,
    #             exception_on_overflow=False
    #         )
    #
    #         audio_np = np.frombuffer(data, dtype=np.int16)
    #         rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
    #
    #         if self.debug:
    #             print(f"[Ears RMS] {int(rms)} speech={speech_started}")
    #
    #         # waiting for speech
    #         if not speech_started:
    #             if rms > self.start_threshold:
    #                 speech_started = True
    #                 frames.append(data)
    #             continue
    #
    #         # recording speech
    #         frames.append(data)
    #
    #         if rms < self.stop_threshold:
    #             silent_chunks += 1
    #         else:
    #             silent_chunks = 0
    #
    #         if silent_chunks >= self.hangover_chunks:
    #             break
    #
    #     if not frames:
    #         return None, 0.0
    #
    #     audio_bytes = b"".join(frames)
    #     duration = len(audio_bytes) / 2 / self.rate  # int16 = 2 bytes
    #
    #     return audio_bytes, duration




# import asyncio
# import time
# import pyaudio
# import numpy as np
#
# class Ears:
#     """Simple microphone listener without VAD."""
#
#     def __init__(self, chunk_size=1024, rate=16000, debug=False):
#         self.audio_stream = None
#         self.chunk_size = chunk_size
#         self.rate = rate
#         self.paused = False
#         self.debug = debug
#
#     async def listen(self, max_duration=30.0):
#         """
#         Record a fixed window of audio (up to max_duration seconds),
#         then trim silence at beginning/end using RMS threshold.
#         Returns (audio_bytes, duration_in_seconds)
#         """
#         if self.audio_stream is None:
#             p = pyaudio.PyAudio()
#             mic_info = p.get_default_input_device_info()
#             self.audio_stream = await asyncio.to_thread(
#                 p.open,
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=self.rate,
#                 input=True,
#                 input_device_index=mic_info["index"],
#                 frames_per_buffer=self.chunk_size,
#             )
#
#         frames = []
#         start_time = time.time()
#
#         while time.time() - start_time < max_duration:
#             if self.paused:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             data = await asyncio.to_thread(
#                 self.audio_stream.read,
#                 self.chunk_size,
#                 exception_on_overflow=False
#             )
#             frames.append(data)
#
#         # convert to numpy array for silence trimming
#         audio_np = np.frombuffer(b"".join(frames), dtype=np.int16)
#
#         # RMS-based silence trim
#         rms_threshold = 1000
#         non_silent = np.where(np.abs(audio_np) > rms_threshold)[0]
#
#         if len(non_silent) == 0:
#             return None, 0.0
#
#         start_idx = non_silent[0]
#         end_idx = non_silent[-1] + 1
#         trimmed_audio = audio_np[start_idx:end_idx]
#
#         duration = len(trimmed_audio) / self.rate
#         return trimmed_audio.tobytes(), duration








# import asyncio
# import time
# import numpy as np
# import pyaudio
# import webrtcvad
# import scipy.signal
#
# class Ears:
#     """Async low-latency microphone listener with WebRTC VAD."""
#
#     def __init__(self, chunk_size=1024, rate=44100, debug=True):
#         self.audio_stream = None
#         self.chunk_size = chunk_size
#         self.rate = rate
#         self.debug = debug
#         self.paused = False
#         self._vad_buffer = np.empty(0, dtype=np.int16)
#         self._is_speaking = False
#
#     async def listen(
#             self,
#             max_duration=30.0,  # Max recording time
#             silence_duration=0.7,  # Stop after this many seconds of silence
#             chunk_size=None,
#             rate=None
#     ):
#         """
#         Returns (audio_bytes, duration) using WebRTC VAD.
#         Stops quickly after silence. Handles short and long commands.
#         """
#         import webrtcvad
#         import numpy as np
#         import time
#         import asyncio
#
#         if chunk_size is None:
#             chunk_size = self.chunk_size
#         if rate is None:
#             rate = self.rate
#
#         # --- ensure audio stream ---
#         if self.audio_stream is None:
#             import pyaudio
#             p = pyaudio.PyAudio()
#             mic_info = p.get_default_input_device_info()
#             self.audio_stream = await asyncio.to_thread(
#                 p.open,
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=int(rate),
#                 input=True,
#                 input_device_index=mic_info["index"],
#                 frames_per_buffer=chunk_size,
#             )
#
#         vad = webrtcvad.Vad(2)  # aggressiveness 0–3
#         FRAME_MS = 30
#         FRAME_BYTES = int(rate * FRAME_MS / 1000) * 2  # bytes per frame
#
#         frames = []
#         speech_started = False
#         silence_start = None
#         speech_start_time = None
#         start_time = time.time()
#
#         buffer = b""
#
#         while True:
#             now = time.time()  # <-- always defined here at start of loop
#
#             if self.paused:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             data = await asyncio.to_thread(
#                 self.audio_stream.read,
#                 chunk_size,
#                 exception_on_overflow=False
#             )
#             buffer += data
#
#             while len(buffer) >= FRAME_BYTES:
#                 frame = buffer[:FRAME_BYTES]
#                 buffer = buffer[FRAME_BYTES:]
#
#                 is_speech = vad.is_speech(frame, int(rate))
#
#                 # ---------- speech start ----------
#                 if not speech_started and is_speech:
#                     speech_started = True
#                     speech_start_time = now
#                     silence_start = None
#                     print("[VAD] Speech start")
#
#                 # ---------- collect frames ----------
#                 if speech_started:
#                     frames.append(frame)
#
#                     # ---------- silence detection ----------
#                     if not is_speech:
#                         if silence_start is None:
#                             silence_start = now
#                         elif now - silence_start >= silence_duration:
#                             print(f"[VAD] Silence end after {now - silence_start:.2f}s")
#                             duration = len(frames) * FRAME_MS / 1000
#                             return b"".join(frames), duration
#                     else:
#                         silence_start = None
#
#                     # ---------- max duration ----------
#                     if now - speech_start_time >= max_duration:
#                         print(f"[VAD] Max duration reached {max_duration}s")
#                         duration = len(frames) * FRAME_MS / 1000
#                         return b"".join(frames), duration
#
#             # ---------- no speech timeout ----------
#             if not speech_started and now - start_time > max_duration:
#                 print("[VAD] No speech detected")
#                 return None, 0.0


# import asyncio
# import time
# import numpy as np
# import pyaudio
# import webrtcvad
# import scipy.signal
#
# class Ears:
#     """Async low-latency microphone listener with WebRTC VAD."""
#
#     def __init__(self, chunk_size=1024, rate=44100, debug=True):
#         self.audio_stream = None
#         self.chunk_size = chunk_size
#         self.rate = rate
#         self.debug = debug
#         self.paused = False
#         self._vad_buffer = np.empty(0, dtype=np.int16)
#         self._is_speaking = False
#
#     async def listen(self, max_duration=30.0, silence_duration=0.7):
#         """
#         Listen until silence detected or max_duration reached.
#         Returns (audio_bytes, duration).
#         Works for short or long commands, multi-language.
#         """
#
#         # --- ensure audio stream ---
#         if self.audio_stream is None:
#             p = pyaudio.PyAudio()
#             mic_info = p.get_default_input_device_info()
#             self.audio_stream = await asyncio.to_thread(
#                 p.open,
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=self.rate,
#                 input=True,
#                 input_device_index=mic_info["index"],
#                 frames_per_buffer=self.chunk_size,
#             )
#
#         vad = webrtcvad.Vad(2)  # aggressiveness 0–3
#         FRAME_MS = 30
#         FRAME_SAMPLES = int(16000 * FRAME_MS / 1000)
#         frames = []
#         speech_started = False
#         silence_start = None
#         start_time = time.time()
#         speech_start_time = None
#
#         while True:
#             if self.paused:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             # --- read audio ---
#             data = await asyncio.to_thread(
#                 self.audio_stream.read,
#                 self.chunk_size,
#                 exception_on_overflow=False
#             )
#             audio_np = np.frombuffer(data, dtype=np.int16)
#
#             # --- resample 44.1k → 16k ---
#             target_len = int(len(audio_np) * 16000 / self.rate)
#             audio_16k = scipy.signal.resample(audio_np, target_len).astype(np.int16)
#
#             # append to buffer
#             self._vad_buffer = np.concatenate([self._vad_buffer, audio_16k])
#
#             # process exact 30ms frames
#             while len(self._vad_buffer) >= FRAME_SAMPLES:
#                 frame = self._vad_buffer[:FRAME_SAMPLES]
#                 self._vad_buffer = self._vad_buffer[FRAME_SAMPLES:]
#                 frame_bytes = frame.tobytes()
#
#                 # --- safe VAD call ---
#                 try:
#                     is_speech = vad.is_speech(frame_bytes, 16000)
#                 except webrtcvad.Error:
#                     if self.debug:
#                         print("[VAD] Invalid frame skipped")
#                     continue
#
#                 rms = int(np.sqrt(np.mean(frame.astype(np.int64)**2)))
#                 now = time.time()
#
#                 if self.debug:
#                     print(f"[VAD DEBUG] RMS={rms}, speech_started={speech_started}")
#
#                 # ---------- speech start ----------
#                 if not speech_started and is_speech:
#                     speech_started = True
#                     speech_start_time = now
#                     silence_start = None
#                     self._is_speaking = True
#                     if self.debug:
#                         print("[VAD] Speech start")
#
#                 # ---------- during speech ----------
#                 if speech_started:
#                     frames.append(frame_bytes)
#
#                     if not is_speech:
#                         if silence_start is None:
#                             silence_start = now
#                         elif now - silence_start > silence_duration:
#                             # stop recording after silence
#                             if self.debug:
#                                 print(f"[VAD] Silence end after {now - speech_start_time:.2f}s")
#                             duration = len(frames) * FRAME_MS / 1000
#                             return b"".join(frames), duration
#                     else:
#                         silence_start = None
#
#                     # ---------- max duration safeguard ----------
#                     if now - speech_start_time > max_duration:
#                         if self.debug:
#                             print(f"[VAD] Max duration reached {max_duration}s")
#                         duration = len(frames) * FRAME_MS / 1000
#                         return b"".join(frames), duration
#
#             # ---------- no speech timeout ----------
#             if not speech_started and now - start_time > max_duration:
#                 if self.debug:
#                     print("[VAD] No speech detected")
#                 return None, 0.0




# import time
# import numpy as np
# import pyaudio
# import asyncio
# import webrtcvad
# import scipy.signal
#
#
# class Ears:
#     """Async low-latency microphone listener with WebRTC VAD."""
#
#     def __init__(self, chunk_size=1024, rate=44100, debug=True):
#         self.audio_stream = None
#         self.chunk_size = chunk_size
#         self.rate = rate
#         self.debug = debug
#         self.paused = False
#         self._vad_buffer = np.empty(0, dtype=np.int16)
#         self._is_speaking = False
#
#     async def listen(self, max_duration=30.0, silence_duration=0.7):
#         """Listen until silence detected or max_duration reached. Returns (audio_bytes, duration)."""
#
#         # --- ensure stream ---
#         if self.audio_stream is None:
#             p = pyaudio.PyAudio()
#             mic_info = p.get_default_input_device_info()
#             self.audio_stream = await asyncio.to_thread(
#                 p.open,
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=self.rate,
#                 input=True,
#                 input_device_index=mic_info["index"],
#                 frames_per_buffer=self.chunk_size,
#             )
#
#         vad = webrtcvad.Vad(2)  # aggressiveness: 0-3
#         FRAME_MS = 30
#         FRAME_SAMPLES = int(16000 * FRAME_MS / 1000)  # 480 samples
#         frames = []
#         speech_started = False
#         silence_start = None
#         start_time = time.time()
#         speech_start_time = None
#
#         while True:
#             if self.paused:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             # --- read audio ---
#             data = await asyncio.to_thread(
#                 self.audio_stream.read, self.chunk_size, exception_on_overflow=False
#             )
#             audio_np = np.frombuffer(data, dtype=np.int16)
#
#             # --- resample to 16kHz exactly ---
#             target_len = int(len(audio_np) * 16000 / self.rate)
#             audio_16k = scipy.signal.resample(audio_np, target_len).astype(np.int16)
#
#             # append to buffer
#             self._vad_buffer = np.concatenate([self._vad_buffer, audio_16k])
#
#             # process exact 30ms frames
#             while len(self._vad_buffer) >= FRAME_SAMPLES:
#                 frame = self._vad_buffer[:FRAME_SAMPLES]
#                 self._vad_buffer = self._vad_buffer[FRAME_SAMPLES:]
#                 frame_bytes = frame.tobytes()
#
#                 # safe VAD call
#                 try:
#                     is_speech = vad.is_speech(frame_bytes, 16000)
#                 except webrtcvad.Error:
#                     if self.debug:
#                         print("[VAD] Invalid frame skipped")
#                     continue
#
#                 rms = int(np.sqrt(np.mean(frame.astype(np.int64)**2)))
#
#                 if self.debug:
#                     print(f"[VAD DEBUG] RMS={rms}, speech_started={speech_started}")
#
#                 now = time.time()
#
#                 # ---------- speech start ----------
#                 if not speech_started and is_speech:
#                     speech_started = True
#                     speech_start_time = now
#                     silence_start = None
#                     self._is_speaking = True
#                     if self.debug:
#                         print("[VAD] Speech start")
#
#                 # ---------- during speech ----------
#                 if speech_started:
#                     frames.append(frame_bytes)
#
#                     if not is_speech:
#                         if silence_start is None:
#                             silence_start = now
#                         elif now - silence_start > silence_duration:
#                             if self.debug:
#                                 print(f"[VAD] Silence end after {now - speech_start_time:.2f}s")
#                             duration = len(frames) * FRAME_MS / 1000
#                             return b"".join(frames), duration
#                     else:
#                         silence_start = None
#
#                     # ---------- max duration safeguard ----------
#                     if now - speech_start_time > max_duration:
#                         if self.debug:
#                             print(f"[VAD] Max duration reached {max_duration}s")
#                         duration = len(frames) * FRAME_MS / 1000
#                         return b"".join(frames), duration
#
#             # ---------- no speech timeout ----------
#             if not speech_started and now - start_time > max_duration:
#                 if self.debug:
#                     print("[VAD] No speech detected")
#                 return None, 0.0



# import time
# import numpy as np
# import webrtcvad
# import asyncio
# import pyaudio
# from scipy.signal import resample_poly
#
# class Ears:
#     def __init__(self):
#         self.audio_stream = None
#         self.paused = False
#         self.chunk_size = 1024
#         self.rate = 44100  # your device sample rate
#         self.debug = True
#         self._vad_buffer = np.empty(0, dtype=np.int16)
#         self._is_speaking = False
#         self._last_speech_time = None
#
#     async def listen(
#         self,
#         max_duration=7.0,      # max total recording time
#         silence_duration=0.5,   # hang on silence threshold
#         chunk_size=None,
#         rate=None,
#     ):
#         """Async listener using WebRTC VAD with proper 16kHz resampling."""
#         if chunk_size is None:
#             chunk_size = self.chunk_size
#         if rate is None:
#             rate = self.rate
#
#         # --- ensure stream ---
#         if self.audio_stream is None:
#             p = pyaudio.PyAudio()
#             mic_info = p.get_default_input_device_info()
#             self.audio_stream = await asyncio.to_thread(
#                 p.open,
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=rate,
#                 input=True,
#                 input_device_index=mic_info["index"],
#                 frames_per_buffer=chunk_size,
#             )
#
#         vad = webrtcvad.Vad(2)  # aggressiveness 0-3
#
#         FRAME_MS = 30
#         FRAME_SIZE = int(16000 * FRAME_MS / 1000)  # samples per frame @16kHz
#
#         frames = []              # collected frames for return
#         speech_started = False   # has speech started yet
#         silence_start = None     # time silence began
#         start_time = time.time() # overall timeout
#         speech_start_time = None # time first speech sample
#
#         buffer = b""
#
#         while True:
#             if self.paused:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             # --- read audio from device ---
#             data = await asyncio.to_thread(
#                 self.audio_stream.read,
#                 chunk_size,
#                 exception_on_overflow=False
#             )
#
#             # --- convert to numpy int16 ---
#             audio_np = np.frombuffer(data, dtype=np.int16)
#
#             # --- resample to 16kHz for VAD ---
#             audio_16k = resample_poly(audio_np, 16000, rate).astype(np.int16)
#
#             # --- append to VAD buffer ---
#             self._vad_buffer = np.concatenate((self._vad_buffer, audio_16k))
#
#             # --- process full 30ms frames ---
#             FRAME_MS = 30
#             FRAME_SIZE = int(rate * FRAME_MS / 1000)
#
#             while len(self._vad_buffer) >= FRAME_SIZE:
#                 frame = self._vad_buffer[:FRAME_SIZE]
#                 self._vad_buffer = self._vad_buffer[FRAME_SIZE:]
#
#                 vad_bytes = frame.tobytes()
#                 is_speech = vad.is_speech(vad_bytes, 16000)
#                 now = time.time()
#
#                 # compute RMS for debug
#                 audio_int = np.frombuffer(vad_bytes, dtype=np.int16)
#                 rms = int(np.sqrt(np.mean(audio_int.astype(np.float32) ** 2)))
#
#                 # rolling RMS indicator (10 chars)
#                 rms_bar = "#" * min(10, rms // 3000)  # adjust 3000 scale if needed
#
#                 if self.debug:
#                     print(f"[DEBUG] RMS={rms:5d} [{rms_bar:<10}] "
#                           f"is_speech={is_speech} _is_speaking={self._is_speaking}")
#
#                 # ---------- speech start ----------
#                 if is_speech:
#                     if not self._is_speaking:
#                         print("[VAD] speech_start")
#                     self._is_speaking = True
#                     self._last_speech_time = now
#                 else:
#                     if self._is_speaking:
#                         silence_time = now - self._last_speech_time
#                         print(f"[VAD] silence {silence_time:.2f}s")
#                         if silence_time >= silence_duration:
#                             print("[VAD] silence_end")
#                             duration = len(frames) * FRAME_MS / 1000
#                             return b"".join(frames), duration
#
#                 # ---------- append frame if speaking ----------
#                 if self._is_speaking:
#                     frames.append(vad_bytes)
#                     if speech_start_time is None:
#                         speech_start_time = now
#
#                     # ---------- max duration ----------
#                     if now - speech_start_time > max_duration:
#                         print("[VAD] Max duration reached")
#                         duration = len(frames) * FRAME_MS / 1000
#                         return b"".join(frames), duration
#
#             # --- no speech detected timeout (optional) ---
#             if not speech_started and (time.time() - start_time) > 5:
#                 print("[VAD] No speech detected")
#                 return None, 0.0
#
#             if self.debug:
#                 print("BUFFER LENGTH:", len(self._vad_buffer))


# # ears_async.py
# import asyncio
# import struct
# import math
# import time
# import pyaudio
# import webrtcvad
# import scipy.signal
#
#
# class Ears:
#     """Async low-latency microphone listener with simple VAD."""
#
#     def __init__(self):
#         self._vad_buffer = None
#         self.audio_stream = None
#         self._is_speaking = False
#         self._silence_start_time = None
#         self.paused = False
#         self.chunk_size = 1024
#         self.rate = 16000
#         self.debug = True
#         self.speech_threshold = None
#         self._last_speech_time = 0
#         self._silence_accum = 0.0
#
#     def _start_debug_loop(self):
#         if self.debug:
#             try:
#                 loop = asyncio.get_running_loop()
#                 loop.create_task(self.test_ears())
#             except RuntimeError:
#                 asyncio.run(self.test_ears())
#
#     async def test_ears(self):
#         if self.debug:
#             while True:
#                 audio, dur = await self.listen()
#                 print(f"Captured => {dur:.2f}s")
#                 print("REAL RATE:", self.audio_stream._rate)
#
#     async def listen(
#             self,
#             max_duration=5.0,
#             silence_duration=0.4,
#             chunk_size=None,
#             rate=None
#     ):
#         if chunk_size is None:
#             chunk_size = self.chunk_size
#         if rate is None:
#             rate = self.rate
#
#         # --- ensure stream ---
#         if self.audio_stream is None:
#             p = pyaudio.PyAudio()
#             mic_info = p.get_default_input_device_info()
#             self.audio_stream = await asyncio.to_thread(
#                 p.open,
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=rate,
#                 input=True,
#                 input_device_index=mic_info["index"],
#                 frames_per_buffer=chunk_size,
#             )
#
#         # --- init VAD state ---
#         if getattr(self, "_vad_buffer", None) is None:
#             self._vad_buffer = np.empty(0, dtype=np.int16)
#         if getattr(self, "_last_speech_time", None) is None:
#             self._last_speech_time = 0
#         if getattr(self, "_is_speaking", None) is None:
#             self._is_speaking = False
#
#         vad = webrtcvad.Vad(2)
#
#         frames = []
#         start_time = time.time()
#
#         while True:
#             if self.paused:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             data = await asyncio.to_thread(
#                 self.audio_stream.read,
#                 chunk_size,
#                 exception_on_overflow=False
#             )
#
#             audio_np = np.frombuffer(data, dtype=np.int16)
#
#             # downsample 44.1k → ~16k
#             audio_16k = scipy.signal.resample_poly(audio_np, 16000, rate).astype(np.int16)
#
#             # append to buffer
#             self._vad_buffer = np.concatenate((self._vad_buffer, audio_16k))
#
#             # process 30ms frames
#             while len(self._vad_buffer) >= 480:
#                 frame = self._vad_buffer[:480]
#                 self._vad_buffer = self._vad_buffer[480:]
#
#                 vad_bytes = frame.tobytes()
#                 is_speech = vad.is_speech(vad_bytes, 16000)
#
#                 now = time.time()
#
#                 if is_speech:
#                     if not self._is_speaking:
#                         print("[VAD] speech_start")
#                     self._is_speaking = True
#                     self._last_speech_time = now
#                     self._silence_accum = 0.0  # ← reset here
#                     frames.append(vad_bytes)
#
#                 else:
#
#                     if self._is_speaking:
#                         self._silence_accum += 0.03
#                         print(f"[VAD] silence {self._silence_accum:.2f}s")
#
#                         if self._silence_accum >= silence_duration:
#                             duration = len(frames) * 0.03
#                             print("[VAD] silence_end")
#                             self._is_speaking = False
#                             self._silence_accum = 0.0
#                             return b"".join(frames), duration
#
#                 # max duration safeguard
#                 if self._is_speaking and (now - self._last_speech_time) > max_duration:
#                     duration = len(frames) * 0.03
#                     print("[VAD] max_duration")
#                     self._is_speaking = False
#                     return b"".join(frames), duration
#
#             # no speech timeout
#             if not self._is_speaking and (time.time() - start_time) > 5:
#                 print("[VAD] no_speech")
#                 return None, 0.0


# import asyncio
# import struct
# import math
# import time
# import pyaudio
#
# class Ears:
#     def __init__(self):
#         self._is_speaking = False
#         self.audio_stream = None
#         self._silence_start_time = None
#         self.paused = False
#         self.out_queue = None
#         self.audio_in_queue = None
#
#     async def listen(self, max_duration=3.0, silence_duration=0.4, chunk_size=1024, rate=16000):
#         """Async listener, returns (audio_bytes, duration). Stops quickly after silence."""
#         if self.audio_stream is None:
#             p = pyaudio.PyAudio()
#             mic_info = p.get_default_input_device_info()
#             self.audio_stream = await asyncio.to_thread(
#                 p.open,
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=rate,
#                 input=True,
#                 input_device_index=mic_info["index"],
#                 frames_per_buffer=chunk_size,
#             )
#
#         frames = []
#         start_time = time.time()
#         self._is_speaking = False
#         self._silence_start_time = None
#         noise_floor = 0
#         START_OFFSET = 4000
#         MIN_THRESHOLD = 8000
#         MAX_THRESHOLD = 30000
#         speech_start_time = None
#
#         while True:
#             if self.paused:
#                 await asyncio.sleep(0.1)
#                 continue
#
#             data = await asyncio.to_thread(self.audio_stream.read, chunk_size, exception_on_overflow=False)
#             frames.append(data)
#             audio = struct.unpack(f"<{len(data)//2}h", data) if len(data) > 0 else [0]
#             rms = int(math.sqrt(sum(s**2 for s in audio)/len(audio)))
#
#             if noise_floor == 0:
#                 noise_floor = rms
#
#             if not self._is_speaking:
#                 noise_floor = 0.95*noise_floor + 0.05*rms
#
#             threshold = max(MIN_THRESHOLD, min(noise_floor + START_OFFSET, MAX_THRESHOLD))
#
#             # Detect speech start
#             if not self._is_speaking and rms >= threshold:
#                 self._is_speaking = True
#                 self._silence_start_time = None
#                 speech_start_time = time.time()
#                 print(f"[Ears] Speech start RMS={rms}")
#
#             # Detect silence after speech
#             if self._is_speaking:
#                 if rms < threshold * 0.6:
#                     if self._silence_start_time is None:
#                         self._silence_start_time = time.time()
#                     elif time.time() - self._silence_start_time > silence_duration:
#                         print(f"[Ears] Silence detected. Ending speech.")
#                         break
#                 else:
#                     self._silence_start_time = None
#
#                 # max duration safeguard
#                 if time.time() - speech_start_time > max_duration:
#                     print(f"[Ears] Max duration reached")
#                     break
#
#             # Global timeout if speech never starts
#             if not self._is_speaking and (time.time() - start_time) > max_duration:
#                 print(f"[Ears] No speech detected")
#                 return None, 0.0
#
#         duration = len(frames)*chunk_size/rate
#         print(f"[Ears] Captured {duration:.2f}s")
#         return b"".join(frames), duration








# import asyncio
# import time
# import struct
# import math
# import pyaudio
#
# class Ears:
#     def __init__(self):
#         self._is_speaking = False
#         self._silence_start_time = None
#         self.audio_stream = None
#         self.paused = False
#         self.out_queue = None
#         self.audio_in_queue = None
#         self._latest_image_payload = None
#
#     async def listen(self, max_duration=6.0, silence_duration=0.5, chunk_size=1024, rate=16000):
#         """
#         Async low-latency microphone listener with adaptive VAD.
#         Returns (audio_bytes, duration_seconds)
#         """
#         pya_inst = pyaudio.PyAudio()
#         mic_info = pya_inst.get_default_input_device_info()
#
#         if self.audio_stream is None:
#             try:
#                 self.audio_stream = await asyncio.to_thread(
#                     pya_inst.open,
#                     format=pyaudio.paInt16,
#                     channels=1,
#                     rate=rate,
#                     input=True,
#                     input_device_index=mic_info["index"],
#                     frames_per_buffer=chunk_size,
#                 )
#             except OSError as e:
#                 print(f"[Ears] Failed to open audio input: {e}")
#                 return None, 0.0
#
#         frames = []
#         speech_started = False
#         speech_start_time = None
#         silence_start = None
#         noise_floor = 0
#         START_OFFSET = 4000
#         MIN_START_THRESHOLD = 8000
#         MAX_START_THRESHOLD = 30000
#         start_time = time.time()
#
#         while True:
#             if self.paused:
#                 await asyncio.sleep(0.1)
#                 continue
#
#             try:
#                 data = await asyncio.to_thread(self.audio_stream.read, chunk_size, exception_on_overflow=False)
#                 frames.append(data)
#                 audio = struct.unpack(f"<{len(data)//2}h", data) if len(data) > 0 else [0]
#                 rms = int(math.sqrt(sum(s**2 for s in audio)/len(audio)))
#
#                 # initialize/adapt noise floor
#                 if noise_floor == 0:
#                     noise_floor = rms
#                 if not speech_started:
#                     noise_floor = 0.95 * noise_floor + 0.05 * rms
#
#                 # adaptive threshold
#                 threshold = max(MIN_START_THRESHOLD, min(noise_floor + START_OFFSET, MAX_START_THRESHOLD))
#
#                 # debug
#                 print(f"[Ears] Noise={noise_floor:.0f} Thr={threshold:.0f} RMS={rms:.0f}", end="\r")
#
#                 # speech detection
#                 if not speech_started and rms >= threshold:
#                     speech_started = True
#                     speech_start_time = time.time()
#                     silence_start = None
#                     print(f"\n[Ears] Speech start RMS={rms}")
#
#                 # silence detection
#                 if speech_started:
#                     if rms < threshold * 0.6:
#                         if silence_start is None:
#                             silence_start = time.time()
#                         elif time.time() - silence_start > silence_duration:
#                             print("\n[Ears] End of speech")
#                             break
#                     else:
#                         silence_start = None
#
#                 # max duration
#                 if speech_started and (time.time() - speech_start_time > max_duration):
#                     print("\n[Ears] Max duration")
#                     break
#
#                 # timeout for no speech
#                 if not speech_started and (time.time() - start_time > max_duration):
#                     print("\n[Ears] No speech detected")
#                     return None, 0.0
#
#             except Exception as e:
#                 print(f"[Ears] Error reading audio: {e}")
#                 await asyncio.sleep(0.05)
#
#         duration = len(frames) * chunk_size / rate
#         print(f"[Ears] Captured {duration:.2f}s")
#         return b"".join(frames), duration







# import asyncio
# import base64
# import struct
# import math
#
#
# class Ears:
#     def __init__(self):
#         self._is_speaking = False
#         self.audio_stream = None
#         self._silence_start_time = None
#         self.audio_in_queue = None
#         self.out_queue = None
#         self.paused = False
#         self._latest_image_payload = None
#         self.session = None
#
#     def clear_audio_queue(self):
#         """Clears the queue of pending audio chunks to stop playback immediately."""
#         try:
#             count = 0
#             while not self.audio_in_queue.empty():
#                 self.audio_in_queue.get_nowait()
#                 count += 1
#             if count > 0:
#                 print(f"[ADA DEBUG] [AUDIO] Cleared {count} chunks from playback queue due to interruption.")
#         except Exception as e:
#             print(f"[ADA DEBUG] [ERR] Failed to clear audio queue: {e}")
#
#     async def send_frame(self, frame_data):
#         # Update the latest frame payload
#         if isinstance(frame_data, bytes):
#             b64_data = base64.b64encode(frame_data).decode('utf-8')
#         else:
#             b64_data = frame_data
#
#             # Store as the designated "next frame to send"
#         self._latest_image_payload = {"mime_type": "image/jpeg", "data": b64_data}
#         # No event signal needed - listen_audio pulls it
#
#     async def send_realtime(self):
#         while True:
#             msg = await self.out_queue.get()
#             await self.session.send(input=msg, end_of_turn=False)
#
#     async def listen(self, chunk_size=1024, rate=16000, vad_threshold=800, silence_duration=0.5, max_duration=6.0):
#         """Async, low-latency VAD-based listener"""
#         if self.audio_stream is None:
#             p = pyaudio.PyAudio()
#             mic_info = p.get_default_input_device_info()
#             self.audio_stream = await asyncio.to_thread(
#                 p.open,
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=rate,
#                 input=True,
#                 input_device_index=mic_info["index"],
#                 frames_per_buffer=chunk_size
#             )
#
#         frames = []
#         speech_started = False
#         speech_start_time = None
#         silence_start = None
#         start_time = time.time()
#
#         while True:
#             if self.paused:
#                 await asyncio.sleep(0.05)
#                 continue
#
#             try:
#                 data = await asyncio.to_thread(self.audio_stream.read, chunk_size, exception_on_overflow=False)
#                 frames.append(data)
#
#                 # Convert to RMS
#                 count = len(data) // 2
#                 shorts = struct.unpack(f"<{count}h", data) if count > 0 else []
#                 rms = int(math.sqrt(sum(s ** 2 for s in shorts) / count)) if count > 0 else 0
#
#                 # Speech detection
#                 if not speech_started and rms >= vad_threshold:
#                     speech_started = True
#                     speech_start_time = time.time()
#                     silence_start = None
#                     print(f"[Ears] Speech start RMS={rms}")
#
#                 # Silence detection
#                 if speech_started:
#                     if rms < vad_threshold * 0.6:
#                         if silence_start is None:
#                             silence_start = time.time()
#                         elif time.time() - silence_start > silence_duration:
#                             print("[Ears] End of speech")
#                             break
#                     else:
#                         silence_start = None
#
#                 # Max duration safety
#                 if speech_started and (time.time() - speech_start_time > max_duration):
#                     print("[Ears] Max duration")
#                     break
#
#                 # Global timeout (no speech)
#                 if not speech_started and (time.time() - start_time > max_duration):
#                     print("[Ears] No speech detected")
#                     return None, 0.0
#
#             except Exception as e:
#                 print(f"[Ears] Error reading audio: {e}")
#                 await asyncio.sleep(0.05)
#
#         duration = len(frames) * chunk_size / rate
#         print(f"[Ears] Captured {duration:.2f}s")
#         return b"".join(frames), duration
        # pya = pyaudio.PyAudio()
        # mic_info = pya.get_default_input_device_info()
        #
        # # Resolve Input Device by Name if provided
        # resolved_input_device_index = None
        #
        # try:
        #     FORMAT = pyaudio.paInt16
        #     CHANNELS = 1
        #     SEND_SAMPLE_RATE = 16000
        #     RECEIVE_SAMPLE_RATE = 24000
        #     CHUNK_SIZE = 1024
        #     self.audio_stream = await asyncio.to_thread(
        #         pya.open,
        #         format=FORMAT,
        #         channels=CHANNELS,
        #         rate=SEND_SAMPLE_RATE,
        #         input=True,
        #         input_device_index=resolved_input_device_index if resolved_input_device_index is not None else mic_info[
        #             "index"],
        #         frames_per_buffer=CHUNK_SIZE,
        #     )
        # except OSError as e:
        #     print(f"[ADA] [ERR] Failed to open audio input stream: {e}")
        #     print("[ADA] [WARN] Audio features will be disabled. Please check microphone permissions.")
        #     return
        #
        # if __debug__:
        #     kwargs = {"exception_on_overflow": False}
        # else:
        #     kwargs = {}
        #
        # # VAD Constants
        # VAD_THRESHOLD = 800  # Adj based on mic sensitivity (800 is conservative for 16-bit)
        # SILENCE_DURATION = 0.5  # Seconds of silence to consider "done speaking"
        #
        # while True:
        #     if self.paused:
        #         await asyncio.sleep(0.1)
        #         continue
        #
        #     try:
        #         data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
        #
        #         # 1. Send Audio
        #         if self.out_queue:
        #             await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
        #
        #         # 2. VAD Logic for Video
        #         # rms = audioop.rms(data, 2)
        #         # Replacement for audioop.rms(data, 2)
        #         count = len(data) // 2
        #         if count > 0:
        #             shorts = struct.unpack(f"<{count}h", data)
        #             sum_squares = sum(s ** 2 for s in shorts)
        #             rms = int(math.sqrt(sum_squares / count))
        #         else:
        #             rms = 0
        #
        #         if rms > VAD_THRESHOLD:
        #             # Speech Detected
        #             self._silence_start_time = None
        #
        #             if not self._is_speaking:
        #                 # NEW Speech Utterance Started
        #                 self._is_speaking = True
        #                 print(f"[ADA DEBUG] [VAD] Speech Detected (RMS: {rms}). Sending Video Frame.")
        #
        #                 # Send ONE frame
        #                 if self._latest_image_payload and self.out_queue:
        #                     await self.out_queue.put(self._latest_image_payload)
        #                 else:
        #                     print(f"[ADA DEBUG] [VAD] No video frame available to send.")
        #
        #         else:
        #             # Silence
        #             if self._is_speaking:
        #                 if self._silence_start_time is None:
        #                     self._silence_start_time = time.time()
        #
        #                 elif time.time() - self._silence_start_time > SILENCE_DURATION:
        #                     # Silence confirmed, reset state
        #                     print(f"[ADA DEBUG] [VAD] Silence detected. Resetting speech state.")
        #                     self._is_speaking = False
        #                     self._silence_start_time = None
        #
        #     except Exception as e:
        #         print(f"Error reading audio: {e}")
        #         await asyncio.sleep(0.1)






# import pyaudio
# import wave
# import struct
# import math
# import time
# import tempfile
#
#
# class Ears:
#     def __init__(
#         self,
#         samplerate=16000,
#         chunk_size=1024,
#         silence_duration=0.5,
#         max_duration=8,
#         mic_index=None,
#     ):
#         self.samplerate = samplerate
#         self.chunk_size = chunk_size
#         self.silence_duration = silence_duration
#         self.max_duration = max_duration
#         self.mic_index = mic_index
#
#         self.pya = pyaudio.PyAudio()
#
#         self.vad_threshold = self._calibrate()
#         print(f"[Ears] VAD threshold: {self.vad_threshold}")
#
#     def _write_wav(self, frames):
#         import wave, tempfile
#         tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         path = tmp.name
#
#         wf = wave.open(path, "wb")
#         wf.setnchannels(1)
#         wf.setsampwidth(self.pya.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(self.samplerate)
#         wf.writeframes(b"".join(frames))
#         wf.close()
#         return path
#
#     # ---------- RMS from int16 ----------
#     def _rms(self, data: bytes) -> int:
#         count = len(data) // 2
#         if count == 0:
#             return 0
#         samples = struct.unpack(f"<{count}h", data)
#         return int(math.sqrt(sum(s*s for s in samples) / count))
#
#     # ---------- Calibration ----------
#     def _calibrate(self):
#         print("[Ears] Calibrating… stay silent")
#
#         stream = self.pya.open(
#             format=pyaudio.paInt16,
#             channels=1,
#             rate=self.samplerate,
#             input=True,
#             frames_per_buffer=self.chunk_size,
#             input_device_index=self.mic_index,
#         )
#
#         rms_vals = []
#         start = time.time()
#
#         while time.time() - start < 1.0:
#             data = stream.read(self.chunk_size, exception_on_overflow=False)
#             rms_vals.append(self._rms(data))
#
#         stream.stop_stream()
#         stream.close()
#
#         noise = sum(rms_vals) / len(rms_vals)
#
#         # speech typically 2–4× noise
#         threshold = int(noise * 2.2)
#
#         print(f"[Ears] Noise floor: {noise:.0f}")
#         return threshold
#
#     # ---------- Listen ----------
#     def listen(self):
#         print("[Ears] Listening… Speak now!")
#
#         stream = self.pya.open(
#             format=pyaudio.paInt16,
#             channels=1,
#             rate=self.samplerate,
#             input=True,
#             frames_per_buffer=self.chunk_size,
#             input_device_index=self.mic_index,
#         )
#
#         NOISE_SAMPLES = 15
#         SILENCE_TIME = 0.4
#         MAX_DURATION = 6
#         SPEECH_RATIO = 2.0  # speech louder than noise
#
#         frames = []
#         speaking = False
#         speech_start = None
#         silence_start = None
#
#         # --- measure noise floor ---
#         noise_vals = []
#         for _ in range(NOISE_SAMPLES):
#             data = stream.read(self.chunk_size, exception_on_overflow=False)
#             noise_vals.append(self._rms(data))
#
#         noise_floor = sum(noise_vals) / len(noise_vals)
#         speech_threshold = noise_floor * SPEECH_RATIO
#
#         print(f"[Ears] Noise floor={noise_floor:.0f}  threshold={speech_threshold:.0f}")
#
#         try:
#             while True:
#                 data = stream.read(self.chunk_size, exception_on_overflow=False)
#                 rms = self._rms(data)
#
#                 if rms > speech_threshold:
#                     if not speaking:
#                         speaking = True
#                         speech_start = time.time()
#                         print(f"[Ears] Speech start RMS={rms:.0f}")
#
#                     frames.append(data)
#                     silence_start = None
#
#                 else:
#                     if speaking:
#                         if silence_start is None:
#                             silence_start = time.time()
#
#                         elif time.time() - silence_start > SILENCE_TIME:
#                             print("[Ears] Speech end")
#                             break
#
#                 if speaking and speech_start:
#                     if time.time() - speech_start > MAX_DURATION:
#                         print("[Ears] Max duration")
#                         break
#
#         finally:
#             stream.stop_stream()
#             stream.close()
#
#         if not frames:
#             print("[Ears] No speech")
#             return None, 0.0
#
#         path = self._write_wav(frames)
#         duration = len(frames) * self.chunk_size / self.samplerate
#         print(f"[Ears] Captured {duration:.2f}s")
#
#         return path, duration


# Old version

# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# import queue
# import time
#
# class Ears:
#     def __init__(
#         self,
#         samplerate=16000,
#         mic_index=None,
#         frame_duration_ms=30,
#         silence_duration_ms=500,
#         start_threshold_ratio=.8  # relative to ambient RMS
#     ):
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#         self.frame_duration = frame_duration_ms / 1000
#         self.frame_samples = int(self.samplerate * self.frame_duration)
#         self.silence_frames = int(silence_duration_ms / frame_duration_ms)
#         self.start_threshold_ratio = start_threshold_ratio
#         self.ambient_rms = None
#
#     def _rms(self, data):
#         return np.sqrt(np.mean(data ** 2))
#
#     def _calibrate(self, seconds=1.0):
#         print("[Ears] Calibrating ambient noise...")
#         frames_needed = int(seconds / self.frame_duration)
#         q = queue.Queue()
#
#         def callback(indata, frames, time_info, status):
#             q.put(indata.copy())
#
#         ambient_data = []
#         try:
#             with sd.InputStream(
#                 samplerate=self.samplerate,
#                 blocksize=self.frame_samples,
#                 device=self.mic_index,
#                 channels=1,
#                 dtype='float32',
#                 callback=callback
#             ):
#                 for _ in range(frames_needed):
#                     try:
#                         data = q.get(timeout=1.0)
#                         ambient_data.append(data)
#                     except queue.Empty:
#                         continue
#         except sd.PortAudioError as e:
#             print(f"[Ears] Calibration error: {e}")
#             self.ambient_rms = 0.01
#             return
#
#         if ambient_data:
#             all_data = np.concatenate(ambient_data, axis=0).flatten()
#             self.ambient_rms = self._rms(all_data)
#         else:
#             self.ambient_rms = 0.01
#         print(f"[Ears] Ambient RMS: {self.ambient_rms:.5f}")
#
#     def listen(self, filename="capture.wav", max_duration=10.0):
#         if self.ambient_rms is None:
#             self._calibrate()
#
#         threshold = self.ambient_rms * self.start_threshold_ratio
#         q = queue.Queue()
#         recording = []
#         silence_counter = 0
#         started = False
#         start_time = time.time()
#
#         def callback(indata, frames, time_info, status):
#             if status:
#                 print(f"[Ears] Stream warning: {status}")
#             q.put(indata.copy())
#
#         try:
#             with sd.InputStream(
#                 samplerate=self.samplerate,
#                 blocksize=self.frame_samples,
#                 device=self.mic_index,
#                 channels=1,
#                 dtype='float32',
#                 callback=callback
#             ) as stream:
#                 print("[Ears] Listening... Speak now!")
#                 while True:
#                     try:
#                         data = q.get(timeout=1.0)
#                     except queue.Empty:
#                         continue
#
#                     rms = self._rms(data)
#
#                     if not started and rms > threshold:
#                         started = True
#                         recording.append(data)
#                         silence_counter = 0
#                         print("[Ears] Loud sound detected, recording...")
#                         continue
#
#                     if started:
#                         recording.append(data)
#                         if rms < threshold:
#                             silence_counter += 1
#                         else:
#                             silence_counter = 0
#
#                         if silence_counter >= self.silence_frames:
#                             print("[Ears] Silence detected, stopping recording.")
#                             break
#                         if time.time() - start_time > max_duration:
#                             print("[Ears] Max duration reached, stopping.")
#                             break
#
#         except sd.PortAudioError as e:
#             print(f"[Ears] PortAudio error: {e}")
#             return None, 0.0
#
#         if not recording:
#             print("[Ears] No speech detected.")
#             return None, 0.0
#
#         audio = np.concatenate(recording, axis=0).flatten()
#         write(filename, self.samplerate, (audio * 32767).astype(np.int16))
#         duration = len(audio) / self.samplerate
#         return filename, duration
#


import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import queue
import time
import pyaudio


# class Ears:
#     def __init__(self, samplerate=16000, mic_index=None, frame_ms=30, silence_ms=500, energy_threshold=0.05):
    #     self.samplerate = samplerate
    #     self.mic_index = mic_index
    #     self.frame_ms = frame_ms
    #     self.frame_samples = int(samplerate * frame_ms / 1000)
    #     self.silence_frames = int(silence_ms / frame_ms)
    #     self.energy_threshold = energy_threshold  # louder noise trigger
    #
    #     FORMAT = pyaudio.paInt16
    #     CHANNELS = 1
    #     RATE = 16000
    #     CHUNK = 1024
    #
    #     self.p = pyaudio.PyAudio()
    #
    #     self.audio_stream = self.p.open(
    #         format=FORMAT,
    #         channels=CHANNELS,
    #         rate=RATE,
    #         input=True,
    #         input_device_index=None,
    #         frames_per_buffer=CHUNK,
    #     )
    #
    # def listen(
    #         self,
    #         max_duration=6.0,
    #         silence_duration=0.6,
    #         chunk=1024,
    #         rate=16000,
    # ):
    #     """
    #     Low-latency microphone listener with adaptive VAD.
    #     Returns captured PCM bytes.
    #     """
    #
    #     print("STARTING EARS.LISTEN")
    #     print("[Ears] Listening… Speak now!")
    #
    #     stream = self.audio_stream
    #
    #     frames = []
    #     speech_started = False
    #     speech_start_time = None
    #     silence_start = None
    #
    #     noise_floor = 0
    #
    #     START_OFFSET = 4000
    #     MIN_START_THRESHOLD = 8000
    #     MAX_START_THRESHOLD = 30000
    #
    #     start_time = time.time()
    #
    #     while True:
    #         data = stream.read(chunk, exception_on_overflow=False)
    #         frames.append(data)
    #
    #         audio = np.frombuffer(data, dtype=np.int16)
    #
    #         if len(audio) == 0:
    #             continue
    #
    #         rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    #
    #         # initialize noise floor
    #         if noise_floor == 0:
    #             noise_floor = rms
    #
    #         # adapt noise floor only when idle
    #         if not speech_started:
    #             noise_floor = 0.95 * noise_floor + 0.05 * rms
    #
    #         # threshold
    #         threshold = noise_floor + START_OFFSET
    #         threshold = max(MIN_START_THRESHOLD, min(threshold, MAX_START_THRESHOLD))
    #
    #         # debug line
    #         print(
    #             f"[Ears] Noise={noise_floor:.0f} Thr={threshold:.0f} RMS={rms:.0f}",
    #             end="\r",
    #         )
    #
    #         # speech start
    #         if not speech_started and rms >= threshold:
    #             speech_started = True
    #             speech_start_time = time.time()
    #             silence_start = None
    #             print(f"\n[Ears] Speech start RMS={rms:.0f}")
    #
    #         # silence detection after speech
    #         if speech_started:
    #             if rms < threshold * 0.6:
    #                 if silence_start is None:
    #                     silence_start = time.time()
    #                 elif time.time() - silence_start > silence_duration:
    #                     print("\n[Ears] End of speech")
    #                     break
    #             else:
    #                 silence_start = None
    #
    #         # max duration safety
    #         if speech_started and (time.time() - speech_start_time > max_duration):
    #             print("\n[Ears] Max duration")
    #             break
    #
    #         # global timeout (no speech)
    #         if not speech_started and (time.time() - start_time > max_duration):
    #             print("\n[Ears] No speech detected")
    #             return None
    #
    #     print(f"[Ears] Captured {len(frames) * chunk / rate:.2f}s")
    #     print("COMPLETED EARS.LISTEN")
    #
    #     return b"".join(frames)

# # modules/ears.py
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# import time
#
#
# class Ears:
#     def __init__(self, samplerate=16000, mic_index=None, duration=15):
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#         self.duration = duration
#
#     def _has_speech(self, audio, threshold=0.01):
#         rms = np.sqrt(np.mean(audio ** 2))
#         return rms > threshold
#
#     def listen(self, filename="capture.wav"):
#         print("[Ears] Listening... Please speak now.")
#         # start_time = time.time()
#
#         audio = sd.rec(
#             int(self.duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="int16",
#             device=self.mic_index
#         )
#         sd_wait_time = time.time()
#         sd.wait()
#         sd_end_time = time.time()
#         elapsed_time = sd_end_time - sd_wait_time
#         print(f"[SD] wait time {elapsed_time} seconds.")
#
#         audio = audio.flatten().astype(np.float32) / 32768.0
#
#         if not self._has_speech(audio):
#             print("[Ears] Silence detected. Ignoring.")
#             return None, 0.0
#
#         write(filename, self.samplerate, (audio * 32768).astype(np.int16))
#         # end_time = time.time()
#         # elapsed_time = end_time - start_time
#         # print(f"[EARS] Listening... Done in {elapsed_time} seconds.")
#         return filename, len(audio) / self.samplerate
#
#







# # modules/ears.py
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
#
# class Ears:
#     def __init__(self, stt, samplerate=16000, mic_index=None, duration=7, short_threshold=5.0):
#         """
#         stt: HybridSTT instance
#         samplerate: recording sample rate
#         mic_index: optional device index
#         duration: max recording duration for capture.wav
#         short_threshold: max seconds considered a 'short command'
#         """
#         self.stt = stt
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#         self.duration = duration
#         self.short_threshold = short_threshold
#
#     def listen(self, duration=None, filename="capture.wav"):
#         duration = duration or self.duration
#         print("[Ears] Listening... Please speak now.")
#
#         audio = sd.rec(
#             int(duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="int16",
#             device=self.mic_index
#         )
#         sd.wait()
#
#         audio = audio.flatten().astype(np.float32) / 32768.0
#
#         def has_speech(audio, threshold=0.01):
#             rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
#             return rms > threshold
#
#         # 🔴 SILENCE GUARD
#         if not has_speech(audio):
#             print("[Ears] No speech detected. Ignoring.")
#             return None
#
#         write(filename, self.samplerate, (audio * 32768).astype(np.int16))
#         return filename
#
#     def listen_stream(self, audio_array=None, chunk_sec=1.0):
#         """
#         Streaming generator for long dictation only.
#         Yields partial transcription every chunk_sec seconds.
#         """
#         if audio_array is None:
#             print("[Ears] Recording long dictation for streaming...")
#             audio_array = sd.rec(int(self.duration * self.samplerate),
#                                  samplerate=self.samplerate,
#                                  channels=1,
#                                  dtype="int16",
#                                  device=self.mic_index)
#             sd.wait()
#
#         buffer = np.zeros((0,), dtype=np.int16)
#         chunk_samples = int(chunk_sec * self.samplerate)
#
#         for start in range(0, len(audio_array), chunk_samples):
#             chunk = audio_array[start:start+chunk_samples].flatten()
#             buffer = np.concatenate((buffer, chunk))
#
#             if len(buffer) >= chunk_samples:
#                 audio_float = buffer.astype(np.float32) / 32768.0
#                 text = self.stt.transcribe_long(audio_float)
#                 if text:
#                     yield text
#                 buffer = np.zeros((0,), dtype=np.int16)








# modules/ears.py
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
#
# class Ears:
#     def __init__(self, stt, samplerate=16000, mic_index=None, duration=7):
#         self.stt = stt
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#         self.duration = duration  # max recording duration
#
#     def listen(self, filename="capture.wav"):
#         """Record full audio to capture.wav"""
#         print("[Ears] Listening... Please speak now.")
#         audio = sd.rec(int(self.duration * self.samplerate),
#                        samplerate=self.samplerate,
#                        channels=1,
#                        dtype="int16",
#                        device=self.mic_index)
#         sd.wait()
#         write(filename, self.samplerate, audio)
#
#         # Decide which STT to use based on duration
#         duration_sec = len(audio) / self.samplerate
#         if duration_sec <= 5.0:
#             # Short command
#             text = self.stt.transcribe_short(filename)
#         else:
#             # Long dictation
#             text = self.stt.transcribe_long(audio)
#         return text





# import numpy as np
# import sounddevice as sd
#
#
# class Ears:
#     def __init__(self, stt, samplerate=16000, mic_index=None, duration=7):
#         self.stt = stt
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#         self.max_duration = duration  # max recording length
#         self.short_chunk_sec = 1.0
#         self.long_chunk_sec = 5.0
#
#     def listen(self):
#         """Record audio and return transcription (short/long auto-detect)."""
#         print(f"[Ears] Listening for up to {self.max_duration}s...")
#         audio = sd.rec(int(self.max_duration * self.samplerate),
#                        samplerate=self.samplerate,
#                        channels=1,
#                        dtype='int16',
#                        device=self.mic_index)
#         sd.wait()
#         duration_sec = len(audio) / self.samplerate
#
#         if duration_sec <= 5.0:
#             return self.stt.transcribe(audio, long=False)
#         else:
#             return self.stt.transcribe(audio, long=True)
#
#     def listen_stream(self, short_chunk_sec=1.0, long_chunk_sec=5.0):
#         """Yield transcribed text continuously."""
#         self.short_chunk_sec = short_chunk_sec
#         self.long_chunk_sec = long_chunk_sec
#
#         short_buffer = np.zeros((0,), dtype=np.int16)
#         long_buffer = np.zeros((0,), dtype=np.int16)
#
#         try:
#             with sd.InputStream(
#                 samplerate=self.samplerate,
#                 channels=1,
#                 dtype="int16",
#                 device=self.mic_index,
#                 blocksize=int(self.samplerate * self.short_chunk_sec)
#             ) as stream:
#                 print("[Ears] Streaming started...")
#                 while True:
#                     chunk, _ = stream.read(int(self.samplerate * self.short_chunk_sec))
#                     chunk = chunk.flatten()
#                     short_buffer = np.concatenate((short_buffer, chunk))
#                     long_buffer = np.concatenate((long_buffer, chunk))
#
#                     # Process short buffer
#                     if len(short_buffer) >= int(self.short_chunk_sec * self.samplerate):
#                         text = self.stt.transcribe(short_buffer, long=False)
#                         if text:
#                             yield text
#                         short_buffer = np.zeros((0,), dtype=np.int16)
#
#                     # Process long buffer
#                     if len(long_buffer) >= int(self.long_chunk_sec * self.samplerate):
#                         text = self.stt.transcribe(long_buffer, long=True)
#                         if text:
#                             yield text
#                         long_buffer = np.zeros((0,), dtype=np.int16)
#
#         except KeyboardInterrupt:
#             print("[Ears] Stopped listening.")
#         except Exception as e:
#             print(f"[Ears] Error: {e}")
#
#











# modules/ears.py
# import numpy as np
# import sounddevice as sd
#
# class Ears:
#     def __init__(self, stt, samplerate=16000, mic_index=None, duration=7):
#         """
#         Streaming microphone listener with short and long buffers.
#         - stt: HybridSTT instance
#         - samplerate: microphone sample rate
#         - mic_index: optional mic index
#         - duration: max recording length for long buffer
#         """
#         self.stt = stt
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#         self.duration = duration  # max buffer for long dictation in seconds
#
#         # Buffers
#         self.short_buffer = np.zeros((0,), dtype=np.int16)
#         self.long_buffer = np.zeros((0,), dtype=np.int16)
#
#         # Short/long chunk sizes
#         self.short_chunk_sec = 1.0   # Whisper short commands
#         self.long_chunk_sec = 5.0    # Faster Whisper long dictation
#
#     def listen_stream(self):
#         """Continuously read from mic and yield transcribed text for short and long audio."""
#         print("[Ears] Starting streaming...")
#
#         try:
#             with sd.InputStream(
#                 samplerate=self.samplerate,
#                 channels=1,
#                 dtype="int16",
#                 device=self.mic_index,
#                 blocksize=int(self.samplerate * 0.25)  # 250ms blocks for responsiveness
#             ) as stream:
#
#                 while True:
#                     # Read a block
#                     chunk, _ = stream.read(int(self.samplerate * 0.25))
#                     chunk = chunk.flatten()
#
#                     # Append to buffers
#                     self.short_buffer = np.concatenate((self.short_buffer, chunk))
#                     self.long_buffer = np.concatenate((self.long_buffer, chunk))
#
#                     # Process short buffer
#                     if len(self.short_buffer) >= int(self.short_chunk_sec * self.samplerate):
#                         audio_short = self.short_buffer[:int(self.short_chunk_sec * self.samplerate)]
#                         text = self.stt.transcribe(audio_short)
#                         if text:
#                             yield text
#                         self.short_buffer = np.zeros((0,), dtype=np.int16)
#
#                     # Process long buffer
#                     if len(self.long_buffer) >= int(self.long_chunk_sec * self.samplerate):
#                         audio_long = self.long_buffer[:int(self.long_chunk_sec * self.samplerate)]
#                         text = self.stt.transcribe(audio_long)
#                         if text:
#                             yield text
#                         self.long_buffer = np.zeros((0,), dtype=np.int16)
#
#         except KeyboardInterrupt:
#             print("[Ears] Stopped listening.")
#         except Exception as e:
#             print(f"[Ears] Error: {e}")














# ears.py
# import numpy as np
# import sounddevice as sd
# from scipy.io.wavfile import write
#
#
# class Ears:
#     def __init__(self, stt, samplerate=16000, mic_index=0, duration=7):
#         """
#         Streaming microphone wrapper.
#         - stt: HybridSTT instance
#         - samplerate: recording sample rate
#         - mic_index: microphone device index
#         - duration: maximum buffer length for long dictation
#         """
#         self.stt = stt
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#         self.duration = duration
#
#         # Buffers for streaming
#         self.short_buffer = np.zeros((0,), dtype=np.int16)
#         self.long_buffer = np.zeros((0,), dtype=np.int16)
#
#     def listen(self, duration=None, filename="capture.wav"):
#         """
#         Record a single slice and transcribe immediately (for short commands).
#         """
#         duration = duration or self.duration
#         print("[Ears] Listening... Please speak now.")
#         audio = sd.rec(int(duration * self.samplerate),
#                        samplerate=self.samplerate,
#                        channels=1,
#                        dtype='int16',
#                        device=self.mic_index)
#         sd.wait()
#         write(filename, self.samplerate, audio)
#
#         audio_array = audio.flatten()
#         return self.stt.transcribe(audio_array, long=(duration > 5.0))
#
#     def listen_stream(self, short_chunk_sec=1.0, long_chunk_sec=5.0):
#         """
#         Streaming input from mic.
#         - short_chunk_sec: duration of short buffer (Whisper)
#         - long_chunk_sec: duration of long buffer (Faster Whisper)
#         Yields transcribed text chunks.
#         """
#         print("[Ears] Starting overlapping streaming...")
#         short_chunk_samples = int(short_chunk_sec * self.samplerate)
#         long_chunk_samples = int(long_chunk_sec * self.samplerate)
#
#         try:
#             with sd.InputStream(
#                 samplerate=self.samplerate,
#                 channels=1,
#                 dtype="int16",
#                 device=self.mic_index,
#                 blocksize=short_chunk_samples
#             ) as stream:
#
#                 while True:
#                     chunk, _ = stream.read(short_chunk_samples)
#                     chunk = chunk.flatten()
#
#                     # Append to buffers
#                     self.short_buffer = np.concatenate((self.short_buffer, chunk))
#                     self.long_buffer = np.concatenate((self.long_buffer, chunk))
#
#                     # Process short buffer
#                     if len(self.short_buffer) >= short_chunk_samples:
#                         audio_float = self.short_buffer.astype(np.float32) / 32768.0
#                         try:
#                             text = self.stt.transcribe(audio_float, long=False)
#                             if text:
#                                 yield text
#                         except Exception as e:
#                             print(f"[Ears][Short] STT error: {e}")
#                         self.short_buffer = np.zeros((0,), dtype=np.int16)
#
#                     # Process long buffer
#                     if len(self.long_buffer) >= long_chunk_samples:
#                         audio_float = self.long_buffer.astype(np.float32) / 32768.0
#                         try:
#                             text = self.stt.transcribe(audio_float, long=True)
#                             if text:
#                                 yield text
#                         except Exception as e:
#                             print(f"[Ears][Long] STT error: {e}")
#                         self.long_buffer = np.zeros((0,), dtype=np.int16)
#
#         except KeyboardInterrupt:
#             print("[Ears] Stopped listening.")
#         except Exception as e:
#             print(f"[Ears] Error: {e}")
#









# import sounddevice as sd
# import numpy as np
#
#
# class Ears:
#     def __init__(self, stt, mic_index=0, duration=5, samplerate=16000, use_mock=False):
#         self.stt = stt
#         self.mic_index = mic_index
#         self.duration = duration
#         self.samplerate = samplerate
#         self.use_mock = use_mock
#
#     def listen(self, duration=None):
#         """Record audio and return transcription."""
#         if self.use_mock:
#             return input("Simulated audio: ")
#
#         duration = duration or self.duration
#         print("[Ears] Listening... Please speak now.")
#         audio = sd.rec(
#             int(duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="int16",
#             device=self.mic_index
#         )
#         sd.wait()
#         print("[Ears] Processing transcription...")
#         return self.stt.transcribe(audio)


















# class Ears:
#     """
#     Continuous audio capture with overlapping short/long buffers.
#     Short: ≤1s → Whisper CPU for commands.
#     Long: ≥3s → Faster Whisper for dictation.
#     """
#     def __init__(self, stt, samplerate=16000, mic_index=None):
#         self.stt = stt
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#
#         self.short_buffer = np.zeros((0,), dtype=np.int16)
#         self.long_buffer = np.zeros((0,), dtype=np.int16)
#
#         self.short_chunk_sec = 0.5  # process every 0.5s for short commands
#         self.long_chunk_sec = 3.0   # process every 3s for long dictation
#
#     def listen_stream(self):
#         """
#         Generator that yields transcribed text in short and long chunks.
#         Uses internal buffers for short and long speech detection.
#         """
#         print("[Ears] Starting overlapping streaming...")
#         self.short_buffer = np.zeros((0,), dtype=np.int16)
#         self.long_buffer = np.zeros((0,), dtype=np.int16)
#
#         try:
#             with sd.InputStream(
#                     samplerate=self.samplerate,  # <- correct
#                     channels=1,
#                     dtype="int16",
#                     device=self.mic_index,
#                     blocksize=int(self.samplerate * self.short_chunk_sec)  # samples per read
#             ) as stream:
#
#                 while True:
#                     chunk, _ = stream.read(int(self.samplerate * self.short_chunk_sec))
#                     chunk = chunk.flatten()
#
#                     # Append to both buffers
#                     self.short_buffer = np.concatenate((self.short_buffer, chunk))
#                     self.long_buffer = np.concatenate((self.long_buffer, chunk))
#
#                     # Process short buffer
#                     if len(self.short_buffer) >= int(self.short_chunk_sec * self.samplerate):
#                         audio_float = self.short_buffer.astype(np.float32) / 32768.0
#                         try:
#                             text = self.stt.transcribe(audio_float, short=True)
#                             if text:
#                                 yield text
#                         except Exception as e:
#                             print(f"[Ears][Short] STT error: {e}")
#                         self.short_buffer = np.zeros((0,), dtype=np.int16)
#
#                     # Process long buffer
#                     if len(self.long_buffer) >= int(self.long_chunk_sec * self.samplerate):
#                         audio_float = self.long_buffer.astype(np.float32) / 32768.0
#                         try:
#                             text = self.stt.transcribe(audio_float, short=False)
#                             if text:
#                                 yield text
#                         except Exception as e:
#                             print(f"[Ears][Long] STT error: {e}")
#                         self.long_buffer = np.zeros((0,), dtype=np.int16)
#
#         except KeyboardInterrupt:
#             print("[Ears] Stopped listening.")
#         except Exception as e:
#             print(f"[Ears] Error: {e}")










# class Ears:
#     """
#     Continuous audio capture from microphone, yielding chunks for STT.
#     Automatically splits audio into short (≤3s) and long (>3s) segments.
#     """
#     def __init__(self, stt, samplerate=16000, mic_index=None):
#         self.stt = stt
#         self.samplerate = samplerate
#         self.mic_index = mic_index
#         self.chunk_duration = 1.0  # 1 second chunk for streaming
#
#     def listen_stream(self):
#         """
#         Generator yielding transcribed text from live mic in near real-time.
#         """
#         print("[Ears] Starting continuous listening...")
#         buffer = np.zeros((0,), dtype=np.int16)
#
#         try:
#             with sd.InputStream(samplerate=self.samplerate,
#                                 channels=1,
#                                 dtype="int16",
#                                 device=self.mic_index,
#                                 blocksize=int(self.samplerate * self.chunk_duration)) as stream:
#
#                 while True:
#                     chunk, _ = stream.read(int(self.samplerate * self.chunk_duration))
#                     chunk = chunk.flatten()
#                     buffer = np.concatenate((buffer, chunk))
#
#                     duration_sec = len(buffer) / self.samplerate
#
#                     # Decide if it's short or long command
#                     if duration_sec >= 3.0:
#                         # Long command → Faster Whisper
#                         text = self.stt.transcribe(buffer)
#                         buffer = np.zeros((0,), dtype=np.int16)  # reset buffer
#                         if text:
#                             yield text
#                     elif duration_sec >= 0.5:
#                         # Short command → Whisper CPU (partial streaming)
#                         text = self.stt.transcribe(buffer)
#                         buffer = np.zeros((0,), dtype=np.int16)
#                         if text:
#                             yield text
#
#         except KeyboardInterrupt:
#             print("[Ears] Stopped listening.")
#         except Exception as e:
#             print(f"[Ears] Error: {e}")
#









# import sounddevice as sd
#
#
# class Ears:
#     def __init__(self, stt, mic_index=0, chunk_duration=1.5):
#         self.stt = stt
#         self.samplerate = stt.samplerate
#         self.chunk_duration = chunk_duration
#         self.mic_index = mic_index
#
#     def listen_stream(self):
#         """Yields transcribed text chunk-by-chunk."""
#         print("[Ears] Listening... Speak now.")
#         try:
#             while True:
#                 audio = sd.rec(
#                     int(self.chunk_duration * self.samplerate),
#                     samplerate=self.samplerate,
#                     channels=1,
#                     dtype="int16",
#                     device=self.mic_index
#                 )
#                 sd.wait()
#                 text = self.stt.transcribe(audio)
#                 if text:
#                     yield text
#         except KeyboardInterrupt:
#             print("[Ears] Stopped listening.")



















# class Ears:
#     def __init__(self, stt, mic_index=0, chunk_duration=1.5):
#         self.stt = stt
#         self.samplerate = stt.samplerate
#         self.chunk_duration = chunk_duration
#         self.mic_index = mic_index
#
#     def listen_stream(self):
#         """Yields transcribed text chunk-by-chunk."""
#         print("[Ears] Listening... Speak now.")
#         try:
#             while True:
#                 audio = sd.rec(
#                     int(self.chunk_duration * self.samplerate),
#                     samplerate=self.samplerate,
#                     channels=1,
#                     dtype="int16",
#                     device=self.mic_index
#                 )
#                 sd.wait()
#                 text = self.stt.transcribe(audio)
#                 if text:
#                     yield text
#         except KeyboardInterrupt:
#             print("[Ears] Stopped listening.")









# class Ears:
#     def __init__(self, stt, mic_index=0, chunk_duration=1.5):
#         """
#         chunk_duration: seconds per slice of audio to feed to STT
#         """
#         self.stt = stt
#         self.samplerate = stt.samplerate
#         self.chunk_duration = chunk_duration
#         self.mic_index = mic_index
#
#     def listen_stream(self):
#         """
#         Generator yielding transcribed text in near-real-time
#         """
#         print("[Ears] Listening... Speak now.")
#         try:
#             while True:
#                 audio = sd.rec(
#                     int(self.chunk_duration * self.samplerate),
#                     samplerate=self.samplerate,
#                     channels=1,
#                     dtype="int16",
#                     device=self.mic_index
#                 )
#                 sd.wait()
#                 text = self.stt.transcribe(audio).strip()
#                 if text:
#                     yield text
#         except KeyboardInterrupt:
#             print("[Ears] Stopped listening.")













# class Ears:
#     def __init__(self, stt, mic_index=0, duration=3):
#         self.stt = stt
#         self.samplerate = stt.samplerate
#         self.duration = duration
#         self.mic_index = mic_index
#
#     def listen(self):
#         print("[Ears] Listening... Please speak now.")
#         audio = sd.rec(
#             int(self.duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="int16",
#             device=self.mic_index
#         )
#         sd.wait()
#         print("[Ears] Processing transcription...")
#         return self.stt.transcribe(audio)





# import sounddevice as sd
#
#
# class Ears:
#     def __init__(self, stt, use_mock=False):
#         self.stt = stt
#         self.samplerate = stt.samplerate
#         self.duration = stt.duration
#         self.mic_index = stt.mic_index
#         self.use_mock = use_mock
#
#     def listen(self, duration=None):
#         duration = duration or self.duration
#
#         if self.use_mock:
#             return input("Simulated audio: ")
#
#         print("[Ears] Listening... Please speak now.")
#         audio = sd.rec(
#             int(duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="int16",
#             device=self.mic_index,
#         )
#         sd.wait()
#         print("[Ears] Processing transcription...")
#         return self.stt.transcribe(audio)
#
#
#







# import os
# import sounddevice as sd
# import numpy as np
# # import wavio  # for saving WAV files if needed
# from modules.stt.factory import create_stt  # assuming your factory is in stt/factory.py
# from scipy.io.wavfile import write
#
#
# class Ears:
#     """
#     Microphone interface for recording and transcribing audio.
#     Supports both Whisper and FasterWhisper via the STTFactory.
#     """
#
#     def __init__(self, stt, samplerate=16000, duration=5, mic_index=None, use_mock=False):        # STT engine
#         # self.stt = create_stt(config)
#         #
#         # # Audio settings
#         # self.samplerate = 16000
#         # self.duration = config["stt"]["duration"]
#         # self.mic_index = 0
#         # self.use_mock = use_mock
#         self.stt = stt
#         self.samplerate = samplerate
#         self.duration = duration
#         self.mic_index = mic_index
#         self.use_mock = use_mock
#
#     def listen(self, duration=None, filename="capture.wav"):
#         """
#         Record audio from the microphone and return transcribed text.
#         """
#         if self.use_mock:
#             return input("Simulated audio: ")
#
#         duration = duration or self.duration
#         print("[Ears] Listening... Please speak now.")
#
#         # Record audio
#         audio = sd.rec(
#             int(duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="int16",
#             device=self.mic_index,
#         )
#         sd.wait()
#
#         # Save audio to WAV file (optional, useful for FasterWhisper)
#         # wavio.write(filename, audio, self.samplerate, sampwidth=2)
#         write(filename, self.samplerate, audio)
#
#         print("[Ears] Processing transcription...")
#         text = self.stt.transcribe(filename)
#         return text
#



# import sounddevice as sd
# from scipy.io.wavfile import write
#
# class Ears:
#     def __init__(self, stt, samplerate=16000, duration=5, mic_index=None, use_mock=False):
#         self.stt = stt
#         self.samplerate = samplerate
#         self.duration = duration
#         self.mic_index = mic_index
#         self.use_mock = use_mock
#
#     def listen(self, duration=None, filename="capture.wav"):
#         """Record audio from microphone and return transcribed text."""
#         if self.use_mock:
#             return input("Simulated audio: ")
#
#         duration = duration or self.duration
#         print("[Ears] Listening... Please speak now.")
#
#         audio = sd.rec(
#             int(duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="int16",
#             device=self.mic_index,
#         )
#         sd.wait()
#
#         write(filename, self.samplerate, audio)
#
#         return self.stt.transcribe(filename)
