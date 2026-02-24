# J.A.R.V.I.S
### Just A Rather Very Intelligent System

A fully local, voice-controlled AI assistant for Linux. Jarvis listens for voice commands, understands natural language, and can create files, write code, search the web, open applications, and much more — all with near-instant response for simple commands and intelligent escalation to more powerful models for complex tasks.

---

## Overview

Jarvis is built around a layered intelligence architecture:

1. **Fast keyword layer** — instant response for known commands (open apps, wake words, pause)
2. **phi3:mini** — handles simple conversational questions directly, escalates everything else
3. **Mistral 7B** — orchestrates complex multi-step tasks and generates structured execution plans
4. **Cloud APIs** — Claude and Gemini available for long-context reasoning and real-time information (opt-in, permission required)

All core functionality runs **completely locally** on your machine. No data leaves your computer unless you explicitly approve it.

---

## System Requirements

- Linux (tested on Pop!_OS)
- Python 3.11+
- 16GB RAM minimum
- GPU optional (NVIDIA recommended) — falls back to CPU automatically
- Microphone
- Speakers or headphones

---

## Setup Checklist

### System Dependencies
- [ ] Python 3.11+
- [ ] `pip` and `venv`
- [ ] `npm` (for any JS tooling)
- [ ] `ffmpeg` — required by Whisper for audio processing
  ```bash
  sudo apt install ffmpeg
  ```
- [ ] `portaudio` — required for microphone input
  ```bash
  sudo apt install portaudio19-dev
  ```
- [ ] `playwright` system dependencies
  ```bash
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
- [ ] Pull DeepSeek Coder (code generation — optional, step 9)
  ```bash
  ollama pull deepseek-coder:6.7b
  ```
- [ ] Verify models are running
  ```bash
  ollama list
  ollama run mistral "say hello"
  ollama run phi3:mini "say hello"
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
  pip install TTS                    # Piper TTS
  pip install pyaudio
  pip install playwright
  pip install sentence-transformers
  pip install faiss-cpu
  pip install pyyaml
  pip install anthropic               # Claude API (optional)
  pip install google-generativeai     # Gemini API (optional)
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
- [ ] Set `llm.enabled: true` in `config.yaml`
- [ ] Set `audio.use_mock: false` for real microphone
- [ ] Set `system.use_gpu: true` if you have a compatible GPU
- [ ] Add API keys if using cloud models (optional)
  ```yaml
  api_keys:
    anthropic: "your-key-here"
    google: "your-key-here"
  ```

### API Keys (Optional)
- [ ] Anthropic (Claude) — https://console.anthropic.com
- [ ] Google (Gemini) — https://makersuite.google.com
- [ ] Set `api_models.claude.enabled: true` and/or `api_models.gemini.enabled: true` in config to activate

### Workspace
- [ ] `workspace/` directory is created automatically on first run
- [ ] Add to `.gitignore`:
  ```
  workspace/
  .venv/
  config.yaml
  ```

---

## Project Structure

```
J.A.R.V.I.S/
├── main.py                  # Entry point
├── config.yaml              # Your local config (gitignored)
├── config.example.yaml      # Template config to share
├── custom_exceptions.py     # PermissionRequired, ModelUnavailable, PlanExecutionError
├── workspace/               # Where Jarvis creates files (gitignored)
├── modules/
│   ├── observer.py          # Main loop — listens, routes, responds
│   ├── brain.py             # LLM routing and plan generation
│   ├── tool_executor.py     # Executes plans (create files, run scripts, etc.)
│   ├── app_launcher.py      # Fast keyword-based app launching
│   ├── browser_controller.py # Playwright browser automation
│   ├── ears.py              # Microphone input
│   ├── tts.py               # Text-to-speech (Piper)
│   └── stt/
│       └── hybrid_stt.py    # Speech-to-text (Whisper + Faster-Whisper)
```

---

## config.yaml Reference

```yaml
llm:
  enabled: true
  backend: local

  models:
    orchestrator:
      name: mistral
      num_ctx: 512        # increase to 1024 if responses get cut off
      temperature: 0.2    # low-medium: consistent JSON output
    classifier:
      name: phi3:mini
      num_ctx: 512
      temperature: 0.0    # deterministic: no creativity needed
    code:
      name: deepseek-coder:6.7b
      num_ctx: 2048       # longer context for full code output
      temperature: 0.1    # nearly deterministic: correct over creative

  api_models:
    claude:
      enabled: false      # set true + add API key to activate
      model: claude-opus-4-6
      max_tokens: 1000
      ask_permission: true  # always ask before sending data externally
    gemini:
      enabled: false
      model: gemini-pro
      ask_permission: true

system:
  use_gpu: false          # set true if you have a compatible GPU

audio:
  use_mock: false         # set true to disable mic for testing
```

---

## Running Jarvis

```bash
source .venv/bin/activate
python main.py
```

Jarvis will say **"Hello sir, what can I do for you"** when ready.

---

## Voice Commands

### Wake / Sleep
| Say | Result |
|-----|--------|
| `Jarvis` or `you there` | Wake from sleep |
| `take a break` or `pause` | Go to sleep |

### Cancel
| Say | Result |
|-----|--------|
| `cancel`, `stop`, `never mind`, `forget it` | Cancel current action immediately |

### Apps
| Say | Result |
|-----|--------|
| `open pycharm` | Launches PyCharm |
| `open vscode` / `code` | Launches VS Code |
| `open browser` / `firefox` | Launches Firefox |
| `open terminal` | Launches terminal |

### Files & Folders
| Say | Result |
|-----|--------|
| `create a file called notes.txt` | Creates `workspace/notes.txt` |
| `create a folder called projects` | Creates `workspace/projects/` |
| `create a folder called project and add a file called main.py` | Multi-step execution |

### Code Generation
| Say | Result |
|-----|--------|
| `create a python file called calculator.py with add and subtract methods` | Writes working Python class |
| `create a python file called app.py with a Flask hello world route` | Generates Flask boilerplate |
| `write a class called Database with connect and query methods` | Writes class to workspace |

### Web & Browser
| Say | Result |
|-----|--------|
| `google latest AI news` | Opens browser, searches Google |
| `search for python tutorials` | Google search |
| `scroll down` / `scroll up` | Scroll current page |
| `go back` | Browser back |
| `new tab` | Opens new tab |
| `click first result` | Clicks first search result |

### Conversation & Facts
| Say | Result |
|-----|--------|
| `what's the capital of France` | Instant answer via phi3 |
| `what's the boiling point of water` | Instant answer via phi3 |
| `how are you today` | Jarvis responds in character |
| `tell me a joke` | Dry British wit |

---

## Architecture: How a Command Flows

```
Your voice
    ↓
Whisper STT — transcribes audio to text
    ↓
Fast keyword layer (AppLauncher)
    ├── matched → execute instantly (open app, wake, pause, cancel)
    └── no match ↓
phi3:mini classifier
    ├── simple fact/conversation → answer directly (~2-4 seconds)
    └── ESCALATE ↓
Mistral orchestrator
    ├── generates JSON execution plan
    ├── local task → ToolExecutor runs the plan
    │     ├── create_file, create_dir, write_code
    │     ├── read_file, run_script, list_dir
    │     └── web_search, browser_navigate
    └── complex/realtime → API models (with your permission)
          ├── Claude — long context, complex reasoning
          └── Gemini — real-time info, current events
```

---

## Feature Roadmap

### Core (Complete)
- [x] Voice input (Whisper STT)
- [x] Voice output (Piper TTS)
- [x] Wake word / sleep commands
- [x] Fast keyword command layer
- [x] App launching
- [x] Browser control (Playwright)
- [x] Ollama local LLM integration
- [x] phi3:mini fast classifier
- [x] Mistral orchestrator
- [x] Structured JSON execution plans
- [x] Tool execution layer (files, folders, code)
- [x] Privacy-first API permission system
- [x] Hallucination filter

### In Progress
- [ ] Plan → Approve → Execute confirmation loop
- [ ] phi3 classifier tuning
- [ ] Claude / Gemini API routing
- [ ] DeepSeek Coder for code generation

### Planned
- [ ] Self-expanding fast keyword layer
- [ ] RAG over local notes and files
- [ ] Screen / vision support (LLaVA)
- [ ] Android client over SSH
- [ ] Persistent memory and user preferences

---

## Troubleshooting

**Jarvis mishears commands**
- Try speaking clearly and at a moderate pace
- For file extensions say "dot p y" instead of "dot py"
- Consider upgrading Whisper model from `small` to `medium` in config
- Check microphone input levels

**Response is very slow**
- Expected on CPU — 7B models take 7-25 seconds without GPU
- Simple questions handled by phi3 should be 2-4 seconds
- Complex file/code tasks hit Mistral and take 20-30 seconds

**"command not understood" still firing**
- Check that `llm.enabled: true` in config.yaml
- Verify Ollama is running: `ollama list`

**phi3 trying to handle computer tasks itself**
- This is a known issue with small models being overconfident
- The system prompt uses few-shot examples to discourage this
- If it persists, the ESCALATE keyword in the response triggers fallthrough to Mistral

**File created in wrong location**
- All files default to `workspace/` directory
- Use absolute paths to specify a different location

**Playwright browser not launching**
- Run `playwright install chromium`
- Run `playwright install-deps`

---

## Privacy

- All processing is **local by default** — nothing leaves your machine
- Cloud API calls (Claude, Gemini) are **disabled by default**
- When enabled, Jarvis **asks permission before every API call**
- API calls are only made when Mistral determines local models are insufficient
- Anthropic and Google do not use API calls to train their models by default

---

## Credits

Built with: Ollama, Whisper, Faster-Whisper, Piper TTS, Playwright, phi3:mini, Mistral 7B, sentence-transformers, FAISS
