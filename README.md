# J.A.R.V.I.S
### Just A Rather Very Intelligent System

A fully local, voice-controlled AI assistant for Linux. Jarvis listens for voice commands, understands natural language, and can create files, write code, search the web, open applications, and much more — all with near-instant response for simple commands and intelligent escalation to more powerful models for complex tasks.

---

## Overview

Jarvis is built around a layered intelligence architecture:

1. **Fast keyword layer** — instant response for known commands (open apps, wake words, pause)
2. **phi3:mini** — handles simple conversational questions directly, escalates everything else
3. **Mistral 7B** — orchestrates complex multi-step tasks and generates structured execution plans
4. **DeepSeek Coder 6.7B** — dedicated code generation model, handles all code files
5. **Cloud APIs** — Claude and Gemini available for long-context reasoning and real-time information (opt-in, permission required)

All core functionality runs **completely locally** on your machine. No data leaves your computer unless you explicitly approve it.

---

## Platform Support

| Feature | Linux | Mac | Windows |
|---------|-------|-----|---------|
| Core AI/LLM | ✅ | ✅ | ✅ |
| Voice I/O | ✅ | ✅ | ⚠️ |
| File/code tools | ✅ | ✅ | ✅ |
| App launching | ✅ | ⚠️ | ⚠️ |
| Browser control | ✅ | ✅ | ✅ |
| GUI | ✅ | ✅ | ✅ |

⚠️ = works with minor path configuration

**Mac:** App paths in `app_launcher.py` are Linux paths — update to `/Applications/...` style. Replace `xdg-open` with `open`.

**Windows:** Not recommended — use a VM instead. `xdg-open` doesn't exist, all app paths break, PyAudio installation is painful.

---

## System Requirements

- Linux (tested on Pop!_OS with COSMIC desktop)
- Python 3.11+
- 16GB RAM minimum
- GPU optional (NVIDIA sm_70+ for CUDA) — falls back to CPU automatically
- Microphone (headset strongly recommended — internal mics pick up background noise)
- Speakers or headphones

---

## Setup Checklist

### System Dependencies
- [ ] Python 3.11+
- [ ] `pip` and `venv`
- [ ] `ffmpeg` — required by Whisper for audio processing
  ```bash
  sudo apt install ffmpeg
  ```
- [ ] `portaudio` — required for microphone input
  ```bash
  sudo apt install portaudio19-dev
  ```
- [ ] Playwright system dependencies
  ```bash
  playwright install firefox
  playwright install-deps
  and/or then updage config.yaml 'use_chrome: true', 'use_firefox: false'
  playwright install chromium
  playwright install-deps
  ```

### Ollama + Local Models
- [ ] Install Ollama
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- [ ] Pull Mistral (orchestrator)
  ```bash
  ollama pull mistral
  ```
- [ ] Pull phi3:mini (fast classifier)
  ```bash
  ollama pull phi3:mini
  ```
- [ ] Pull DeepSeek Coder (code generation)
  ```bash
  ollama pull deepseek-coder:6.7b
  ```
- [ ] Verify models are running
  ```bash
  ollama list
  ollama run mistral "say hello"
  ollama run phi3:mini "say hello"
  ollama run deepseek-coder:6.7b "say hello"
  ```

### Python Environment
- [ ] Create and activate virtual environment
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- [ ] Install Python dependencies
  ```bash
  pip install ollama
  pip install openai-whisper
  pip install faster-whisper
  pip install piper-tts
  pip install pyaudio
  pip install simpleaudio
  pip install numpy
  pip install scipy
  pip install PyQt5
  pip install playwright
  pip install pyyaml
  pip install anthropic               # Claude API (optional)
  pip install google-genai            # Gemini API (optional)
  ```
- [ ] Install Playwright browser
  ```bash
  playwright install chromium
  ```

### Configuration
- [ ] Copy and edit config
  ```bash
  cp config.example.yaml config.yaml
  ```
- [ ] Set `audio.use_mock: false` for real microphone
- [ ] Set `system.use_gpu: true` if you have a compatible NVIDIA GPU (sm_70+)
- [ ] Add API keys if using cloud models (optional)
  ```yaml
  api_keys:
    anthropic: "your-key-here"
    google: "your-key-here"
  ```

### API Keys (Optional)
- [ ] Anthropic (Claude) — https://console.anthropic.com
- [ ] Google (Gemini) — https://console.cloud.google.com
- [ ] Set `api_models.claude.enabled: true` and/or `api_models.gemini.enabled: true` in config to activate

### Workspace
- [ ] `workspace/` directory is created automatically on first run
- [ ] Add to `.gitignore`:
  ```
  workspace/
  .venv/
  config.yaml
  .window_pos
  ```
  ### Browser Profile (First Run)
Jarvis creates a persistent browser profile automatically on first use.
When you say your first browser command, a Firefox window will open.
Log into YouTube, Gmail, or any sites you want Jarvis to access.
Your session is saved — you won't be asked again.

To use Chrome instead, update config.yaml:
```yaml
browser:
  use_chrome: true
  use_firefox: false
  chrome_profile_path: "/home/YOUR_USER/.config/google-chrome/Jarvis"
```

---

## Project Structure

```
J.A.R.V.I.S/
├── main.py                  # Entry point — PyQt5 app + async Observer thread
├── config.yaml              # Your local config (gitignored)
├── config.example.yaml      # Template config to share
├── custom_exceptions.py     # PermissionRequired, ModelUnavailable, PlanExecutionError
├── workspace/               # Where Jarvis creates files (gitignored)
├── modules/
│   ├── observer.py          # Main loop — listens, routes, responds, command queue
│   ├── brain.py             # LLM routing and plan generation
│   ├── tool_executor.py     # Executes plans (create files, run scripts, etc.)
│   ├── app_launcher.py      # Fast keyword-based app launching
│   ├── browser_controller.py # Playwright browser automation
│   ├── ears.py              # Microphone input with dynamic noise calibration
│   ├── tts.py               # Text-to-speech (Piper)
│   ├── face.py              # PyQt5 GUI — orb, captions, controls
│   └── stt/
│       └── hybrid_stt.py    # Speech-to-text (Whisper + Faster-Whisper)
```

---

## config.yaml Reference

```yaml
# ---- Node identity ----
node:
  name: "g7"
  role: "single_node" # single_node | observer | brain | hands

# ---- Networking (future use) ----
network:
  brain_ip: "127.0.0.1"
  brain_port: 8000
  hands_ip: "127.0.0.1"
  hands_port: 8001
  observer_ip: "127.0.0.1"
  observer_port: 8002

# ---- LLM (disabled for now) ----
llm:
  enabled: true
  # local | remote
  backend: "local"

  models:
    classifier:
      name: phi3:mini
      # set for 3 sentence max responses - double if cut off
      num_ctx: 2048
      # deterministic - no creativity needed, just classify
      temperature: 0.0
      # hard output cap - 3 sentence max - increase for longer responses from phi3 (basic AI)
      max_tokens: 300

    orchestrator:
      name: mistral
      # set for shorter commands - double if cut off
      num_ctx: 8092
      # low-medium - consistent JSON output, handles varied phrasing
      temperature: 0.2
      max_tokens: 500

    code:
      # deepseek-r1 - Deep system analysis
      name: deepseek-coder:6.7b
      # longer context for full code output - bump to 8192 if truncating complex outputs
      num_ctx: 8092
      # nearly deterministic - correct code over creative code
      temperature: 0.1

  api_models:
    claude:
      enabled: false
      model: claude-opus-4-6
      max_tokens: 1000
      ask_permission: true
    gemini:
      enabled: true
      # models/gemini-2.5-flash-lite -
      model: gemini-2.5-flash
      ask_permission: true

  # legacy/unused for now
  model_path: "./models/mpt-7b/"

# ---- Memory ----
memory:
  enabled: false # enable later when adding FAISS
  vector_db_path: "./memory/vector_db/"
  short_term_db: "./memory/short_term.db"
  recall_top_k: 5

# ---- Audio ----
audio:
  samplerate: 16000
  # Max length of message that can be heard in seconds
  duration: 30
  mic_index: null
  use_mock: false
  calibration_interval: 30
  # give up waiting to START speaking after 3s
  pre_speech_timeout: 3.0
  # hard cap — allows long commands
  max_speech_duration: 25.0
  # wait 1.5s of silence before stopping mid-speech
  silence_seconds: 1.5

transcription:
  engine: faster-whisper

  tts:
    # mock TTS
    enabled: false
    engine: "coqui"
    voice: "default"

    # update

# ---- Execution permissions ----
permissions:
  allow_file_write: true
  # safer default
  allow_command_exec: false
  allow_network: false

# ---- Logging ----
logging:
  level: "INFO"
  file: "./logs/jarvis.log"

# ---- Development ----
dev:
  # IMPORTANT: keeps everything lightweight
  mock_mode: true
  auto_approve: false

stt:
  language: en
  # Length of message that can be heard in seconds
  duration: 30
   # tiny | base | small | medium | large
  whisper_model: small
  # tiny | base | small | medium | large
  fw_model: small

system:
  cpu_threads: 6
  use_gpu: false

browser:
  profile: "jarvis"
  chrome_profile_path: "/home/{user}/.config/google-chrome/jarvis"
  use_chrome: false
  firefox_profile_path: "/home/{user}/.mozilla/firefox/jarvis"
  use_firefox: true
```

---

## Running Jarvis

```bash
source .venv/bin/activate
python main.py
```

Jarvis will calibrate the microphone noise floor, open the GUI, then say **"Hello sir, what can I do for you"** when ready.

---

## GUI

The GUI is a PyQt5 window with a vispy 3D particle orb embedded above a control panel.

### Orb States
| State | Color | Animation |
|-------|-------|-----------|
| Listening | Blue | Slow gentle pulse |
| Thinking | Green | Medium pulse, more particles |
| Speaking | Light blue | Fast pulse |
| Sleeping | Yellow | Very slow deep breath, sparse particles |
| Error | Red | Rapid jitter |

The orb features per-particle color variation, depth-based size variation, and beam line connections between nearby particles for a holographic look. Each state has distinct particle density, breathing speed, and connection density.

### Caption Area
Displays real-time status as Jarvis works:
```
classifying...
planning...
step 1 of 3: create_dir...
step 2 of 3: generate_code...
waiting for confirmation...
```
Shows spoken text when Jarvis is speaking, clears automatically after.

### Controls
| Control | Function |
|---------|----------|
| Text input + Send | Type commands directly, bypasses voice |
| Cancel button | Stops current task, returns to listening |
| Mute button | Toggles microphone on/off, turns red when active |

### GUI Notes
- Window positioning and always-on-top not supported under Wayland by design
- On COSMIC desktop: right-click titlebar → Sticky to pin above other windows
- On X11: window position saves automatically every 5 seconds and restores on next launch

---

## Voice Commands

### Wake / Sleep
| Say | Result |
|-----|--------|
| `Jarvis` or `you there` | Wake from sleep |
| `take a break` or `pause` | Go to sleep |

### Cancel / Confirm
| Say | Result |
|-----|--------|
| `cancel`, `stop`, `never mind`, `forget it` | Cancel current action |
| `yes`, `yeah`, `do it`, `go ahead` | Confirm pending action |
| `overwrite`, `over write`, `replace` | Confirm overwriting existing file |

### Apps
| Say | Result |
|-----|--------|
| `open pycharm` | Launches PyCharm |
| `open vscode` | Launches VS Code |
| `open browser` | Launches browser |
| `open terminal` | Launches terminal |

### Files & Folders
| Say | Result |
|-----|--------|
| `create a file called notes.txt` | Creates `workspace/notes.txt` |
| `create a folder called projects` | Creates `workspace/projects/` |
| `create a folder called api with a file called routes.py` | Multi-step plan |

### Code Generation
| Say | Result |
|-----|--------|
| `create a python file called calculator.py with add and subtract methods` | Generates working Python class |
| `create a file called backend.py with flask routes` | Generates Flask app |
| `create a homepage.html with about and contact sections using inline css` | Generates styled HTML page |
| `create a folder called bot with a file called bot.py that has an AI assistant class` | Multi-step with code generation |

### Web & Browser
| Say | Result |
|-----|--------|
| `google latest AI news` | Opens browser, searches Google |
| `youtube lo-fi music` | Opens YouTube search |
| `navigate to github.com` | Opens URL directly |
| `scroll down` / `go down` | Scroll page down |
| `scroll up` / `go up` | Scroll page up |
| `go back` / `go forward` | Browser navigation |
| `refresh` / `reload` | Reload page |
| `new tab` / `close tab` | Tab management |
| `click first result` | Clicks first search result |
| `zoom in` / `zoom out` / `zoom reset` | Page zoom |
| `full screen` | Toggle fullscreen |

### Keyboard Shortcuts (Context-Aware)
| Say | Result |
|-----|--------|
| `save` | Ctrl+S |
| `copy` / `paste` | Ctrl+C / Ctrl+V |
| `undo` / `redo` | Ctrl+Z / Ctrl+Shift+Z |
| `find` | Ctrl+F |
| `select all` | Ctrl+A |
| `close` | Ctrl+W |

### Conversation & Facts
| Say | Result |
|-----|--------|
| `what's the capital of France` | Instant answer via phi3 |
| `tell me a joke` | Dry British wit |
| `how are you today` | Jarvis responds in character |

---

## Architecture: How a Command Flows

```
Your voice
    ↓
Whisper STT — transcribes audio to text
    ↓
Hallucination filter — drops repetitive/too-short/echo audio
    ↓
Fast keyword layer (AppLauncher)
    ├── matched → execute instantly (open app, wake, pause, cancel)
    └── no match ↓
Command Queue — voice keeps listening while Brain processes
    ↓
phi3:mini classifier
    ├── simple fact/conversation → answer directly (~2-4 seconds)
    └── ESCALATE ↓
Mistral orchestrator
    ├── generates JSON execution plan (dynamic ctx: 1024–8192)
    ├── speaks summary → "Shall I proceed, sir?"
    ├── waits for voice confirmation
    └── confirmed ↓
ToolExecutor
    ├── create_file, create_dir
    ├── generate_code → DeepSeek Coder writes actual code
    ├── read_file, run_script, list_dir, delete_file
    └── web_search, browser_navigate, browser_search
```

---

## Command Queue

Jarvis processes commands concurrently — voice keeps listening while Brain is working. Say a second command while Jarvis is executing the first and it will be queued and processed immediately after.

```
"create a flask app"  →  Brain starts working
"what's 5 times 5"   →  queued while Brain works
                         answered immediately after first command finishes
```

Commands are processed one at a time in order — no commands are dropped.

---

## Dynamic Context Windows

Mistral's context window scales automatically based on command complexity:

| Command Type | ctx | Example |
|---|---|---|
| Simple (≤10 words) | 1024 | "create a file called hello.txt" |
| Medium (≤20 words) | 2048 | "create a folder called projects" |
| Code task (≤15 words) | 4096 | "create a flask file with routes" |
| Complex code (>15 words) | 8192 | "create a flask backend with auth and database models" |
| Complex multi-step | 4096 | long commands with and/then/also |

---

## Noise Floor Calibration

Jarvis automatically calibrates microphone thresholds at startup and every 20 seconds:

- Uses the **95th percentile** of samples — handles AC/fan spikes better than mean
- Dynamically sets `start_threshold` and `stop_threshold` based on your environment
- Prints results: `[Ears] Noise floor=1842 start=7368 stop=3684`
- Warns if noise floor exceeds 5000 RMS and uses conservative static thresholds
- Stream lock prevents concurrent calibration and listening from causing ALSA crashes

**Tip:** A headset or USB microphone dramatically improves accuracy in noisy environments.

---

## Plan → Confirm → Execute

For any command that involves executing steps, Jarvis will:

1. **Speak the plan** — "Creating flask folder with backend.py inside."
2. **Ask for confirmation** — "Shall I proceed, sir?"
3. **Wait for your response** — say yes/yeah/do it/go ahead to confirm, anything else cancels
4. **Execute** — runs each step, updates GUI caption with live progress, checks for cancel between steps
5. **Report** — "Done, sir. Code written to workspace/backend.py"

**File exists?** Jarvis says "backend.py already exists, sir. Say overwrite to replace it." Say overwrite/replace/yes to proceed or anything else to keep the existing file.

**Slow response?** If Brain takes more than 5 seconds, Jarvis says "One moment please, sir." and continues working.

---

## Feature Roadmap

### Complete
- [x] Voice input (Whisper + Faster-Whisper hybrid STT)
- [x] Voice output (Piper TTS) with British male voice, speed and pitch control
- [x] PyQt5 GUI with embedded vispy 3D particle orb
- [x] Five orb states — listening, thinking, speaking, sleeping, error
- [x] Per-particle color variation and depth-based size variation
- [x] Beam line connections for holographic look
- [x] Smooth sin-wave breathing animation per state
- [x] Real-time caption display — Brain status and spoken text
- [x] Cancel button — stops current task instantly
- [x] Mute button — toggles microphone
- [x] Text input field — type commands directly
- [x] Wake word / sleep commands
- [x] Fast keyword command layer
- [x] App launching
- [x] Browser control (Playwright) with scroll, zoom, click, navigate, YouTube
- [x] Context-aware keyboard shortcuts
- [x] Ollama local LLM integration
- [x] phi3:mini fast classifier with ESCALATE guardrails
- [x] Mistral orchestrator with structured JSON plans
- [x] Dynamic context window sizing based on command complexity
- [x] DeepSeek Coder for all code generation (Python, JS, HTML, CSS, Dart, etc.)
- [x] Tool execution layer (files, folders, code, browser, web search)
- [x] Privacy-first API permission system
- [x] Hallucination and echo loop filters
- [x] Echo cancellation (ears paused during TTS + buffer flush)
- [x] Plan → Confirm → Execute flow with live step progress in GUI
- [x] File overwrite confirmation
- [x] Cancel between execution steps
- [x] Command queue — speak next command while Brain is working
- [x] "One moment please" for slow Brain responses
- [x] Dynamic noise floor calibration (auto-adjusts to environment every 20s)
- [x] ALSA stream recovery with exponential backoff
- [x] Stream lock preventing concurrent mic access crashes (SIGABRT/SIGSEGV)
- [x] DeepSeek explanation stripper — fluff auto-commented at bottom of file

### Planned
- [ ] Claude / Gemini API routing
- [ ] Self-expanding fast keyword layer
- [ ] RAG over local notes and files
- [ ] Screen / vision support (LLaVA)
- [ ] Android client over SSH
- [ ] Persistent memory and user preferences
- [ ] Push-to-talk mode

---

## Troubleshooting

**Jarvis mishears commands**
- Speak clearly and at a moderate pace
- For file extensions say "dot p y" not "dot py"
- Consider upgrading Whisper model from `small` to `medium` in config
- A headset mic dramatically reduces background noise issues

**"Max speech duration reached" constantly firing**
- Background noise exceeds your start threshold
- Check `[Ears] Noise floor=...` in logs — if above 5000, environment is very noisy
- Increase `start_multiplier` in config (default 4.0, try 5.0 or 6.0)
- A headset mic is the most reliable fix

**ALSA/SIGABRT/SIGSEGV crashes**
- Stream accessed concurrently — ensure `_stream_lock` is in place in ears.py
- Disable `auto_calibrate` temporarily to confirm it's the source
- Restart audio: `systemctl --user restart pipewire pipewire-pulse wireplumber`

**Mic not found after audio crash**
- Run: `systemctl --user restart pipewire pipewire-pulse wireplumber`
- ears.py falls back to `pulse` device automatically if analog mic isn't found

**Response is very slow**
- Expected on CPU — 7B models take 20-30 seconds without GPU
- Simple questions via phi3 should be 2-4 seconds
- Code generation via DeepSeek takes 30-90 seconds on CPU
- GPU upgrade (sm_70+) is the most impactful hardware improvement

**phi3 trying to handle computer tasks itself**
- ESCALATE keyword triggers fallthrough to Mistral
- Code block detector in `quick_answer()` catches most cases
- Add examples to phi3 system prompt for edge cases

**File created with explanation text instead of code**
- DeepSeek added commentary — the stripper handles this automatically
- Explanation lines are commented out at bottom of file, not deleted
- Check `[ToolExecutor] ⚠️ Stripped N lines` in logs

**Mistral returning truncated JSON**
- Dynamic ctx sizing handles this automatically
- Check `[Brain] num_ctx:` log line
- If still occurring, increase manually in config

**"Shall I proceed" confirmation not heard**
- Ears resumes after TTS — speak clearly after Jarvis finishes asking
- If misheard, it will cancel — just repeat the command

**GUI window position not saving**
- Position save/restore only works on X11 — not supported under Wayland by design
- On COSMIC desktop: right-click titlebar → Sticky to pin the window

**GUI loads in the middle of the screen**
- Expected on Wayland — compositor controls window placement
- Drag to preferred position and use compositor sticky/pin feature

---

## Running Tests
```bash
source .venv/bin/activate
pytest tests/ -v
```

### Test Structure
```
tests/
├── conftest.py              # shared fixtures — mock config, mock audio, fake Observer
├── test_brain.py            # LLM routing, plan generation, context window sizing
├── test_tool_executor.py    # file creation, code generation, plan execution
├── test_observer.py         # command routing, cancel, confirmation flow
├── test_ears.py             # noise calibration, hallucination filters
└── test_stt.py              # transcription, echo detection
```

### Running Specific Tests
```bash
pytest tests/test_brain.py -v          # brain tests only
pytest tests/test_tool_executor.py -v  # tool executor only
pytest -k "test_cancel" -v             # any test with "cancel" in the name
pytest -x                              # stop on first failure
```

### Test Coverage
```bash
pip install pytest-cov
pytest tests/ --cov=modules --cov-report=term-missing
```

---

## Privacy

- All processing is **local by default** — nothing leaves your machine
- Cloud API calls (Claude, Gemini) are **disabled by default**
- When enabled, Jarvis **asks permission before every API call**
- Anthropic and Google do not use API calls to train their models by default

---

## Credits

Built with: Ollama, Whisper, Faster-Whisper, Piper TTS, simpleaudio, PyAudio, PyQt5, vispy, scipy, Playwright, phi3:mini, Mistral 7B, DeepSeek Coder 6.7B, numpy, PyYAML
