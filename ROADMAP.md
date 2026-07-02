# Atlas Feature Priority Roadmap
### Scored by Reward ÷ Time (highest first)

> **Reward** = resume value + daily usefulness + impressiveness  
> **Time** = realistic build time for this codebase  
> **Score** = Reward (1-10) ÷ Time (1-10) — higher is better

---

## 🔴 Do First — Highest ROI

| # | Feature | Reward | Time | Score | Notes |
|---|---------|--------|------|-------|-------|
| 1 | **Fix first word cutoff** | 8 | 1 | 8.0 | Already diagnosed — VAD pre-buffer fix. Makes every interaction better. |
| 2 | **Fix broken tests** | 7 | 1 | 7.0 | Audit + mock STT/TTS. Unlocks CI/CD later. |
| 3 | **Rename repo + GPL v3 license** | 6 | 1 | 6.0 | 20 minutes. Protects the project before going public. |
| 4 | **Wake word (openWakeWord)** | 9 | 2 | 4.5 | Transforms daily usability. No more always-on STT. Huge demo moment. |
| 5 | **"What can you do?" response** | 5 | 1 | 5.0 | Hardcoded response. 1 hour. Makes demo look polished. |

---

## 🟡 High Value — Build Next

| # | Feature | Reward | Time | Score | Notes |
|---|---------|--------|------|-------|-------|
| 6 | **Google Calendar API** | 8 | 2 | 4.0 | "What do I have today?" is the #1 daily use case. Direct API, no browser needed. |
| 7 | **Gmail read + send** | 8 | 2 | 4.0 | Pairs naturally with calendar. Shows real workflow automation. |
| 8 | **Memory / ChromaDB** | 9 | 3 | 3.0 | Follow-up questions, context retention. Biggest single leap in intelligence. High resume value. |
| 9 | **Multi-command stack** | 7 | 2 | 3.5 | "Play music, create a file, remind me in 20 minutes." Impressive demo, moderate build. |
| 10 | **Read file** | 6 | 1 | 6.0 | "Read notes.txt" → summarize. Already has file write, read is trivial to add. |
| 11 | **Screenshot → analyze** | 8 | 2 | 4.0 | Capture screen → vision model → "here's the error and fix." Extremely impressive live. |
| 12 | **n8n integration** | 7 | 2 | 3.5 | Webhook trigger. Unlocks 400+ integrations for free. Automates complex workflows. |

---

## 🟢 Strong Features — Medium Term

| # | Feature | Reward | Time | Score | Notes |
|---|---------|--------|------|-------|-------|
| 13 | **Multi-model routing (classifier)** | 8 | 3 | 2.7 | Rule-based first, then LLM classifier. Resume gold — shows orchestration architecture. |
| 14 | **Context-aware projects** | 8 | 3 | 2.7 | "Create project GSX-R" → all commands scope to it. Needs memory first. |
| 15 | **Research {topic} → MD + audio** | 7 | 2 | 3.5 | Scrape → generate MD → speak summary. Already partially built. |
| 16 | **Send text via Twilio** | 6 | 2 | 3.0 | Simple API. Good demo. Pairs with Gmail. |
| 17 | **Slack alerts** | 6 | 2 | 3.0 | Webhook-based. Easy to add. Good for productivity angle. |
| 18 | **YouTube keyboard shortcuts** | 5 | 1 | 5.0 | pause/play/skip/mute via keyboard — 2 hours, makes YouTube control feel complete. |
| 19 | **Config Linux/Mac/Windows** | 6 | 2 | 3.0 | Platform utils layer. Needed before any public release. |
| 20 | **Top-level debug flag** | 4 | 1 | 4.0 | `--debug` CLI flag. Already have per-file debug, just needs global toggle. |

---

## 🔵 Big Swings — Later

| # | Feature | Reward | Time | Score | Notes |
|---|---------|--------|------|-------|-------|
| 21 | **VSCode voice plugin** | 9 | 5 | 1.8 | "Line 43, create a function that does X." Developer tooling + LLM = very hot combo. |
| 22 | **FastAPI WebSocket server** | 8 | 3 | 2.7 | Wraps existing app. Enables mobile. Mostly additive, minimal refactor. |
| 23 | **Create image (DALL-E/SD)** | 6 | 2 | 3.0 | "Create an image of X" → save locally. Simple API call. |
| 24 | **Analyze PDF/document** | 7 | 2 | 3.5 | Upload to Claude API with doc context. Return synopsis. |
| 25 | **Flutter mobile app** | 9 | 8 | 1.1 | Best animation, best long term. Needs FastAPI server first. Save for last. |

---

## 📊 Summary — Recommended Sprint Order

```
Sprint 1  →  Fix word cutoff + fix tests + rename + license + "what can you do"
Sprint 2  →  Wake word
Sprint 3  →  Calendar API + Gmail
Sprint 4  →  Read file + YouTube shortcuts + research topic
Sprint 5  →  Memory / ChromaDB
Sprint 6  →  Multi-command stack + n8n
Sprint 7  →  Screenshot analysis + multi-model routing
Sprint 8  →  Context-aware projects + platform config
Sprint 9  →  VSCode plugin + FastAPI server
Sprint 10 →  Flutter mobile app
```

---

## 🎯 Impact by Feature

These are the features that will generate the most interview conversation:

1. **Memory / ChromaDB** — shows you understand stateful AI systems
2. **Multi-model routing** — shows orchestration architecture thinking
3. **VSCode voice plugin** — developer tooling + LLM is a rare combo
4. **FastAPI WebSocket server** — distributed systems, real-time streaming
5. **Flutter mobile app** — turns a desktop app into a platform
6. **Wake word** — shows audio DSP depth beyond basic mic input
7. **Screenshot → analyze** — multimodal AI, vision models

---

## Project Structure

```
A.T.L.A.S/
├── main.py                  # Entry point — PyQt5 app + async Observer thread
├── config.yaml              # Your local config (gitignored)
├── config.example.yaml      # Template config to share
├── custom_exceptions.py     # PermissionRequired, ModelUnavailable, PlanExecutionError
├── config/
│      └── api_keys.py      # Secure API key management via OS keyring
├── modules/
│   └── gmail/
│       ├── email_drafter.py # Drafts email responses with Mistral
│       ├── email_monitor.py # Monitor emails and notifies user with important emails analyzed by Mistral
│       └── gmail_module.py  # Authenticate on first run, check emails, write replies, find contacts, etc.
│   └── observer/
│       ├── calendar_handler.py # Google calendar. Auth of first run. Extrapolates voice/text commands.
│       ├── document_handler.py # Extrapolates voice/text commands for analyzing PDF. Finds file automatically.
│       ├── eyes_handler.py  # Extrapolates voice/text commands for analyzing what computer camera sees.
│       ├── gmail_handler.py  # Extrapolates voice/text commands for gmail commands.
│       └── observer.py  # Main loop — listens, routes, responds, command queue
│   └── stt/
│       └── hybrid_stt.py    # Speech-to-text (Whisper + Faster-Whisper)
│   └── voices/
│       └── voice files      # TTS voice files (Piper)
│   ├── app_launcher.py      # Fast keyword-based app launching
│   ├── brain.py             # LLM routing and plan generation
│   ├── browser_controller.py # Playwright browser automation
│   ├── calendar_module.py   # Google calendar - check schedule and add events
│   ├── ears.py              # Microphone input with dynamic noise calibration
│   ├── eyes.py              # Webcam vision with LLaVA
│   ├── face.py              # PyQt5 GUI — orb, facial expressions, captions, controls
│   ├── mouth.py               # Text-to-speech (Piper)
│   ├── tool_executor.py     # Executes plans (create files, run scripts, etc.)
│   └── window_controller.py # Voice/text commands for navigating programs
│   └── notes/
│       └── note files      # Planned update details. Known errors.
│   └── screenshots/
│       └── saved image files # Created on first use. From Llava eyes image analization. Warned when file becomes large.
│   └── tests/
│       └── test files      # Test components. See tests section below for more details.
│   └── workspace/
│       └── project files      # Created on first use. Stores projects created by Deepseek and other models.
```

---