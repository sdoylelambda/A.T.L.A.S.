# A.T.L.A.S. Project Context
*Condensed conversation summary for Claude — use this to resume work*

---

## Developer
**Sean Doyle** — Solo developer, Pop!_OS 24.04, COSMIC desktop, GTX 1060 6GB

---

## Project Overview
**A.T.L.A.S.** (Autonomous Task and Local AI System) — fully local, voice-controlled AI assistant built in Python. Privacy-first: all processing local, cloud opt-in with confirmation.

**Repo:** `~/dev/A.T.L.A.S.`
**Mobile repo:** `~/dev/atlas_mobile` (Flutter Android)
**Username:** `zero`

---

## Tech Stack
- **Language:** Python 3.11, Dart/Flutter
- **GUI:** PyQt5 + vispy 3D particle orb
- **STT:** Whisper + Faster-Whisper hybrid
- **TTS:** Piper TTS (British male — alan)
- **LLM Routing:** Ollama (local) + Gemini API
- **Models:** phi3:mini (classifier), Mistral 7B (orchestrator), qwen2.5-coder:7b (code gen), LLaVA (vision)
- **Browser:** Playwright
- **Memory:** MemPalace + ChromaDB
- **Calendar/Email:** Google Calendar + Gmail API (OAuth2)
- **Vision:** OpenCV + LLaVA
- **Remote Access:** SSH + Tailscale + Termux (Android)
- **Mobile:** Flutter Android app (Samsung S10+)
- **API:** FastAPI + uvicorn (port 8000)

---

## Key File Locations
```
A.T.L.A.S/
├── main.py                         # entry point, --no-gui for SSH/headless
├── config.yaml                     # local config (gitignored)
├── api/
│   ├── fastapi_server.py           # FastAPI mobile API server
│   ├── atlas-api.service           # systemd service (not yet installed)
│   ├── gen_qr.py                   # generates QR code (server URL + API key JSON bundle)
│   └── wake_server.py              # (not used — SSH approach chosen instead)
├── modules/
│   ├── brain.py                    # LLM routing, create_plan(), query()
│   ├── observer/
│   │   ├── observer.py             # main loop, keyword layer, Brain routing
│   │   ├── light_handler.py        # BLE LED light handler (NEW — not yet wired)
│   │   └── ...other handlers
│   ├── mouth.py                    # Piper TTS — wait_done() uses asyncio.to_thread()
│   ├── ears.py                     # mic input
│   ├── light_controller.py         # BLE LED controller (NEW — not yet tested)
│   └── ...
├── tests/
│   └── test_brain_dynamic_ctx.py   # Mistral ctx tuning test suite
└── config/
    └── api_keys.py                 # keyring-based secure key storage

atlas_mobile/
└── lib/
    ├── main.dart
    ├── screens/
    │   ├── home_screen.dart        # main UI — orb, STT, TTS, conversation
    │   ├── settings_screen.dart    # server URL, API key (QR scan), SSH host/user
    │   └── setup_screen.dart       # first-launch SSH key install
    ├── services/
    │   ├── atlas_service.dart      # HTTP client + SSH wake
    │   └── ssh_key_manager.dart    # ed25519 keypair generation + storage
    └── widgets/
        ├── orb_painter.dart        # CustomPainter 3D particle orb
        └── conversation_drawer.dart # slide-up chat history
```

---

## Architecture

### Command Flow
```
Voice/Text Input
  → STT (Whisper/FasterWhisper) or Flutter STT
  → _text_command_queue (unified)
  → Keyword Layer (observer.py)
  → phi3:mini classifier → ESCALATE if needed
  → Mistral 7B → JSON execution plan
  → ToolExecutor → create_file, generate_code, browser, calendar, email
  → TTS (Piper) → speak response
  → Memory.remember_conversation()
```

### FastAPI Mobile Flow
```
Flutter app opens
  → checks GET /status (already running? skip SSH)
  → if not running: password popup → SSH → bashrc fires → uvicorn starts
  → polls /status until atlas_running=true (8s initial delay + 45x3s polls)
  → POST /command → observer._text_command_queue.put(text)
  → Observer processes (full pipeline)
  → say() → _api_response_queue.put(text) + speaks on PC
  → Flutter TTS speaks response on phone
```

---

## Important Brain.py Details

### create_plan() — key method
```python
def create_plan(self, command: str, num_ctx_override: int = None) -> dict:
    # num_ctx_override added by Sean to allow test suite injection
    num_ctx = num_ctx_override or self._get_num_ctx(command)
    result = self.query(command, model_key="orchestrator", system=system, num_ctx_override=num_ctx)
    # returns dict: {"summary": "...", "route": "local|claude|gemini", "steps": [...]}
```

### _get_num_ctx() — tuned via test suite
```python
# After tuning: most commands need only 1024 tokens
# Current buckets (to be updated based on test results):
# simple (<=10 words): 1024
# medium (<=20 words): 1024
# long (>20 words): 4096
# code simple: 1024
# code complex (>=15 words): 4096
```

### Known bugs in _get_num_ctx (to fix):
- **BUG-1:** "backend"/"frontend" in code_keywords misclassifies plain-English commands
- **BUG-2:** `words > 15` should be `words >= 15` (lightbulb command was exactly 15 words and truncated in prod)

---

## Observer.py Key Additions
```python
# Added to __init__:
self._api_response_queue  = asyncio.Queue()
self._api_request_pending = False

# Added to say() after self._last_spoken = text.lower().strip():
if self._api_request_pending:
    self._api_response_queue.put_nowait(text)
    self._api_request_pending = False
```

### mouth.py fix (prevents blocking event loop):
```python
# Changed from:
self._current_play.wait_done()
# To:
await asyncio.to_thread(self._current_play.wait_done)
```

---

## FastAPI Server Key Details
- **Port:** 8000
- **Auth:** X-API-Key header — key stored in `~/.config/atlas/api_key` (chmod 600)
- **Endpoints:** GET /status, POST /command, POST /cancel
- **Rate limiting:** slowapi (30/min commands, 60/min cancels)
- **Binding:** Tailscale IP only via `$(tailscale ip -4 | head -n1)`
- **Audio:** `config['audio']['use_mock'] = True` when headless (prevents PyAudio errors)
- **Start command:**
```bash
TAILSCALE_IP=$(tailscale ip -4 | head -n1)
nohup /home/zero/dev/A.T.L.A.S./.venv/bin/uvicorn api.fastapi_server:app \
  --host $TAILSCALE_IP --port 8000 \
  > ~/.atlas/logs/fastapi.log 2>&1 &
```

### ~/.bashrc SSH block:
```bash
if [ -n "$SSH_CONNECTION" ]; then
    cd /home/zero/dev/A.T.L.A.S.
    if ! pgrep -f "uvicorn api.fastapi_server:app" > /dev/null; then
        TAILSCALE_IP=$(tailscale ip -4 | head -n1)
        mkdir -p /home/zero/.atlas/logs
        nohup /home/zero/dev/A.T.L.A.S./.venv/bin/uvicorn api.fastapi_server:app \
            --host $TAILSCALE_IP --port 8000 \
            > /home/zero/.atlas/logs/fastapi.log 2>&1 &
    fi
    exit
fi
```

---

## Flutter App Key Details

### Dependencies
```yaml
dependencies:
  http, speech_to_text, flutter_tts, flutter_secure_storage,
  mobile_scanner, dartssh2, cryptography, qr_flutter (removed)
```

### Secure Storage Keys
- `atlas_api_key` — API key
- `atlas_server_url` — FastAPI URL e.g. http://100.x.x.x:8000
- `atlas_ssh_host` — Tailscale IP
- `atlas_ssh_user` — Linux username
- `atlas_ssh_fingerprint` — TOFU host fingerprint
- `atlas_ssh_private_pem` — ed25519 private key (Android Keystore)
- `atlas_ssh_public_pem` — public key authorized_keys line
- `atlas_ssh_public_b64` — public key base64

### SSH Security Model
- ed25519 keypair generated on phone first launch
- Public key installed to `~/.ssh/authorized_keys` via SetupScreen (one time)
- Every launch: password popup (two-factor: password + key)
- TOFU host fingerprint pinning
- If already running: skip SSH, connect directly

### STT — Chunked Accumulation
Android enforces ~5s hard STT timeout. Fix:
- `_listenOnce()` loops while button held
- Each chunk appends to `_accumulatedText`
- Full string sent on button release
- `cancelOnError: false`, `listenMode: ListenMode.dictation`

### Orb States
```dart
listening → blue   (#3366FF) — 400 particles
thinking  → green  (#33FF66) — 650 particles
speaking  → cyan   (#66CCFF) — 550 particles
error     → red    (#FF3333) — 400 particles
sleeping  → yellow (#EECC33) — 75 particles
```

---

## Config.yaml Structure
```yaml
personalize:
  ai_assistant_name: "A.T.L.A.S."
  response_name: ""

llm:
  models:
    classifier:   {name: phi3:mini}
    orchestrator: {name: mistral}
    code:         {name: qwen2.5-coder:7b-instruct-q4_K_M}
  api_models:
    claude:  {enabled: false}
    gemini:  {enabled: true, ask_permission: true}

audio:
  use_mock: false
  samplerate: 16000

light:
  enabled: true
  device_address: null   # set after first BLE connect

api_server:
  api_key: ""    # not used — key in ~/.config/atlas/api_key
  port: 8000
```

---

## BLE Light Controller (modules/light_controller.py)
- **Status:** Written, NOT yet tested (BLE connection issues with Pop!_OS BlueZ)
- **Bulb:** Vibe E-ssential LED — Bluetooth, unnamed BLE device (no stable MAC)
- **Protocol:** Tries Tuya/Triones/ELK-BLEDOM in order
- **Issue:** BlueZ on Pop!_OS has trouble with random MAC rotation
- **Last tried fix:** `Experimental = true` in `/etc/bluetooth/main.conf`
- **Voice commands:** on/off, set color to X, dim to X%, flash/strobe/fade/pulse
- **NOT yet wired into observer.py** — needs `register_light(self, config)` added to `__init__`

---

## Storage Locations
```
~/.config/atlas/api_key          # FastAPI API key (chmod 600)
~/.config/atlas/google_*         # OAuth tokens
~/.atlas/logs/fastapi.log        # FastAPI/uvicorn logs
~/.atlas/errors/ctx_test_errors/ # Mistral test truncation logs
~/.ssh/authorized_keys           # Flutter SSH public key
~/.ssh/known_hosts               # Tailscale IP fingerprint
~/.mempalace/                    # MemPalace persistent memory
```

---

## Current Known Issues / TODO
- [ ] BLE light connection not working on Pop!_OS (BlueZ random MAC issue)
- [ ] light_handler.py not wired into observer.py yet
- [ ] brain_handler.py not extracted from observer.py
- [ ] debug=True still in several files
- [ ] FunctionGemma integration in progress
- [ ] Option D audio (PC mic + phone simultaneously) — not yet implemented
- [ ] _get_num_ctx BUG-1 and BUG-2 not yet fixed in brain.py
- [ ] systemd atlas-api.service not yet installed

---

## Roadmap (Priority Order)
1. Fix BLE light controller connection
2. Wire light_handler into observer.py
3. Fix _get_num_ctx BUG-1 and BUG-2
4. Option D audio — PC mic + phone simultaneously
5. FunctionGemma — live weather/stocks/news
6. Email monitor — background inbox watcher
7. FastAPI → launch desktop GUI + terminal on SSH connect
8. Flutter UI improvements (always improving)
9. RAG over local notes/files
10. Mac/PC versions, Docker, monetization

---

## Resume / Interview Notes
Sean is actively interviewing. Key talking points:
- Multi-model AI orchestration (phi3/Mistral/qwen/Gemini)
- Privacy-first: all processing local, cloud opt-in
- RAG memory via ChromaDB/FAISS
- Agentic execution planning (Mistral → JSON → ToolExecutor)
- FastAPI backend, Flutter mobile, SSH security, OAuth2
- Full test suite including ctx window benchmarking
- Android STT chunked accumulation (solved 5s platform limit)
- Two-factor SSH security (ed25519 key + password)
