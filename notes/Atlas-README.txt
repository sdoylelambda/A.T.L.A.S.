# A.T.L.A.S.
An AI assistant based on Iron Man's J.A.R.V.I.S.

# Setup
Uses Wayland. If you are using x11 switch code in WindowController. X11 code is commented out below. Same for open_app in app_launcher. 
Wayland code is in use currently.
Create venv - 
Install requirements.txt - pip install -r requirements.txt
Run main.py - python main.py or python3 main.py

# Basic use
[x] - Speak open (program name) ie: browser - browser loads
[x] - Speak 'take a break' Atlas does not respond until you speak 'Atlas' or 'you there?'
[x] - Atlas 'face', spinning orb of lights, reacts when in different states. (sleeping, listening, thinking, error)

### MVP
[x] - Voice Commands
[x] - Voice Responses
[x] - Test dummy LLM data -> Small LLM -> Larger (simply replace LLM folder with desired LLM)
[x] - Open/use Programs
Enter data
[x] - Answer general questions

### Stretch Goals
Multiple nodes
    3 Nodes setup
    1 - Brain - AI decisions - Primary Computer
    2 - Work Station - Handle AI decisions/commands - Secondary Computer
    3 - Peripherals - Remote use - Cellphone
        a. Add-ons such as cameras, temperature gauge, 

### Use Case
Using cellphone, fire up app. 
Say build me a python project that prints hello world.
Cell phone sends request to Primary Computer.
Primary computer creates a plan and sends back to the cell phone for user review. 
Cell phone either speaks plan aloud, or shows in text. (Depending on preference)
User confirms plan. Says build it.
Cell phone sends plan accepted message to Primary Computer.
Primary Computer sends plan to work station.
Work station opens Pycharm, creates project. Sends screenshot to cell phone.
User views screenshot on cellphone then says run it.
Project runs on secondary computer. (Reduces load on primary computer, peripherals, monitor, etc - Primary computer free to function as normal without interference.)


Claude joint plan
Claude / Gemini API routing — infrastructure built, just needs the google-genai fix and end-to-end testing
Save window position — remember where user dragged the GUI, restore on next launch --- can't on wayland

Planned Features

GUI update — display thought process, show what Brain is doing in real time
Self-expanding keyword layer — Atlas learns frequent commands and routes them instantly next time, skipping Brain entirely
RAG over local files — Atlas reads your own documents and answers questions about them
Screen / vision support — LLaVA model lets Atlas see your screen
Android client over SSH — control Atlas from your phone
Persistent memory — remembers preferences and context across sessions
Push-to-talk mode — keyboard shortcut triggers listening, bypasses mic threshold

Known Issues Worth Fixing

Cancel mid-generation — can only cancel between steps, not while DeepSeek is writing
Double execution — occasional write_code + generate_code both firing
Whisper mishearing — upgrading to medium model would help significantly

Stretch Goals

React/Flutter GUI — full web-based interface with richer animations
Voice training — tune STT to your specific voice


[x] - GUI update
    [] - ui button to proceed - build it
    [x] - speaking face color
    [x] - display thought process

[] - Adapt
[] - Move files from / that aren't needed there
    [] - move actions to task folder
[] - Remove extra code/files/etc.
[] - More AI models for more tasks
    [] - General question escalation AI
    [] - Write email/paper/pdf AI
[] - fix broken tests
[x] - ask what can you do? get response
    [x] - say hello -> i'm Atlas...
[] - config linux/mac/windows versions
[] - top level debug? or keep per file?
[] - research {topic}
    [] - create md
    [x] - give audio summary
[x] - RULE: One sentence max for all responses. --- made it 'Keep responses as short as possible.'
[x] - test performance under load/multitasking - looks good - 1 core at 100% rest under 20%
[] - for 'bike week schedule' Atlas should say feb 27-march 7
[] - say what can you do -> i'm Atlas, here to...
[x] - try different stt model sizes - see time delay? +5 seconds large not much better - small for fast response
[] - max recording length?
[] - read file
[] - should mistral only return code? or just when asked to create code? who should handle mid level tasks?

Context
[] - Remember Conversations
    [] - understand when a follow-up question/action is asked/requested
[] - be able to understand context and apply as appropriate
    [] - create a new {project} - now it knows we are talking about {project} until i talk about a different subject
        [] - create a new file, spreadsheet etc.
            [] - example: create a new project called gsx-r. create a price estimate for the following parts in txt file
    [] - create a new project called lead gen. make a md file for plan. execute step 1. or execute all steps in order

GUI
[x] - more beams on all expressions - add till it's too much, then turn it down a little
[] - approve action button
    [] - type 'yes' to approve action

Models
[] - Critic Model
    [] - Reviews code before execution
    [] - Flags unsafe shell commands
[] - Memory Module (Vector DB + embeddings)
    [] - Remembers previous debugging sessions
    [] - Remembers architecture decisions
[] - Self-Evaluator
    [] - Scores confidence
    [] - Re-routes if low
[] - Tool Executor Sandbox
    [] - Controlled command execution
    [] - Isolated environment
[] - Output validator (anti-hallucination critic) - gpt-oss-safeguard
[] - decide what should determine what model is used. workflow, keywords, task, hybrid, or let AI figure it out
[] - why does mistral open regular browser? is this what i want? probably better to use play write for voice controls
    [] - should mistral handle that or another model?
        [] - how can it? prompt says it can only write code.
[] Other models
    [] - Claude and
    [] - coding another option - qwen3-coder-next
    [] - analyze images
        [] - screenshot
            [] - text - summarize
                [] - text problem - how to solve - how to fix this error (or should it integrate directly into coder)
            [] - image - describe
    [] - create image
    [] - analyze PDF/document - give synapsis
    [] - what else
    [] - eyes - qwen3-vl

Bug Fix
[x] - 1st word or 2 cut off - fix - debug see what is's hearing - probably filtered out. check spike duration
    [x] confirm action - maybe say - confirm, please, proceed
[] - YouTube {a song} command seems to not let music play more than a few seconds. why? playwrite not playing nice?
    [] - why does google YouTube {song} work fine and not YouTube {song}

Integrations
[] - n8n integration
[x] - calendar integration
    [x] - what do i have today?
    [x] - add event
    [] - timer --- remind me this 30 minutes to do something
    [] - speak calendar alerts
[] - send gmail
    [] - read email and respond
[] - send text
    [] - read text and respond
[] - slack alerts
[] - email alerts
[] - text alerts
[] - pycharm/vscode
    [] - line 43 create a function that does x,y,z.
    [] - go
    [] - review code
    [] - look for edge cases
    [] - try and break it

[] - multiple commands
    [] - create stack
        [] - YouTube some music, create a file, then take a break.


AI Prompt:
A.T.L.A.S. Project Summary
Autonomous Task and Local AI System
Project Overview
A fully local, voice-controlled AI assistant for Linux built in Python. Runs on Pop!_OS with COSMIC desktop. GTX 1060 6GB GPU. Primary developer: Sean Doyle.

Tech Stack
Language:        Python 3.11
GUI:             PyQt5 + vispy 3D particle orb
STT:             Whisper + Faster-Whisper hybrid
TTS:             Piper TTS (British male voice - alan)
LLM Routing:     Ollama (local) + Gemini API
Models:          phi3:mini (classifier), Mistral 7B (orchestrator),
                 qwen2.5-coder:7b (code gen), LLaVA (vision)
Browser:         Playwright
Memory:          MemPalace + ChromaDB
Calendar:        Google Calendar API (OAuth2)
Email:           Gmail API (OAuth2)
Vision:          OpenCV + LLaVA
Remote Access:   SSH + Tailscale + Termux (Android)
Android:         proot Ubuntu + Ollama phi3 on S10+

File Structure
A.T.L.A.S/
├── main.py                          # entry point, --no-gui flag for SSH
├── config.yaml                      # local config (gitignored)
├── config.example.yaml              # template
├── custom_exceptions.py             # PermissionRequired, ModelUnavailable, PlanExecutionError
├── requirements.txt                 # pinned deps (numpy==1.26.4 critical)
├── mempalace.yaml                   # MemPalace config
├── entities.json                    # MemPalace entities
│
├── modules/
│   ├── brain.py                     # LLM routing, plan generation, query()
│   ├── tool_executor.py             # executes plans (files, code, browser)
│   ├── app_launcher.py              # fast keyword app launching
│   ├── browser_controller.py        # Playwright automation
│   ├── ears.py                      # mic input, noise calibration, use_mock support
│   ├── mouth.py                     # Piper TTS
│   ├── face.py                      # PyQt5 GUI, vispy orb, config-driven colors
│   ├── window_controller.py         # window management
│   ├── eyes.py                      # webcam + screenshot vision, LLaVA
│   ├── memory.py                    # MemPalace wrapper, recall/remember
│   ├── calendar_module.py           # Google Calendar CRUD
│   ├── gmail_module.py              # Gmail read/draft/send
│   ├── email_drafter.py             # fast rule-based + Mistral drafting
│   ├── document_reader.py           # PDF text extraction
│   ├── utils.py                     # timer context manager
│   │
│   ├── observer/
│   │   ├── __init__.py              # exports Observer
│   │   ├── observer.py              # main loop, keyword layer, Brain routing
│   │   ├── vision_handler.py        # vision command routing
│   │   ├── calendar_handler.py      # calendar command routing
│   │   ├── gmail_handler.py         # gmail command routing
│   │   ├── document_handler.py      # PDF summarization routing
│   │   └── brain_handler.py         # Brain command execution (planned)
│   │
│   └── stt/
│       └── hybrid_stt.py            # Whisper + Faster-Whisper
│
├── config/
│   └── api_keys.py                  # keyring-based secure key storage
│
├── tests/
│   ├── conftest.py
│   ├── test_brain.py
│   ├── test_tool_executor.py
│   ├── test_observer.py
│   ├── test_ears.py
│   ├── test_stt.py
│   ├── test_calendar.py
│   ├── test_eyes.py
│   └── test_memory.py
│
├── workspace/                       # Atlas creates files here (gitignored)
├── screenshots/                     # README screenshots
└── voices/
    └── british_man_GB-alan-medium.onnx

Architecture: Command Flow
Voice/Text Input
    ↓
Ears (PyAudio) → STT (Whisper/FasterWhisper)
    ↓
Hallucination filters (repetition, echo, length)
    ↓
_text_command_queue (voice and text unified)
    ↓
Keyword Layer (observer.py):
  - wake/sleep commands
  - cancel/confirm
  - capabilities announcement
  - calendar commands → calendar_handler.py
  - gmail commands → gmail_handler.py
  - vision commands → vision_handler.py
  - document commands → document_handler.py
  - memory commands (remember/recall)
    ↓
AppLauncher.dispatch() (fast keyword matching)
    ↓
Brain.process():
  Layer 1: phi3:mini → quick answer or ESCALATE
  Layer 2: Mistral → JSON execution plan
  Layer 3: API (Gemini/Claude) → complex/realtime
    ↓
ToolExecutor → create_file, generate_code, browser, web_search
    ↓
TTS (Piper) → speak response
    ↓
Memory.remember_conversation() → store in MemPalace

Brain Model Routing
phi3:mini      → fast Q&A, classification, ESCALATE guard
Mistral 7B     → orchestration, JSON plans, email drafts
qwen2.5-coder  → ALL code generation (.py .js .html .css etc)
LLaVA          → webcam/screenshot vision analysis
Gemini API     → long context, document analysis (user confirms)
Claude API     → complex reasoning (disabled, needs payment)
FunctionGemma  → live data: weather, stocks, news (IN PROGRESS)

Completed Features
✅ Voice pipeline (Whisper STT + Piper TTS)
✅ PyQt5 GUI with vispy 3D particle orb (5 states)
✅ Config-driven GUI colors
✅ Multi-model AI orchestration
✅ Dynamic context window sizing
✅ phi3 ESCALATE guardrails + refusal detection
✅ Fast keyword layer
✅ App launching
✅ Browser automation (Playwright)
✅ Context-aware keyboard shortcuts
✅ File/folder/code creation
✅ Plan → Confirm → Execute flow
✅ File overwrite confirmation
✅ Cancel between steps
✅ Command queue (concurrent voice+brain)
✅ "One moment please" for slow responses
✅ Dynamic noise floor calibration
✅ ALSA stream recovery
✅ Echo cancellation
✅ Hallucination filters
✅ Gemini API routing (user confirms before sending)
✅ Secure API key storage (OS keyring)
✅ GUI API key entry
✅ Google Calendar (read/create events, smart NLP parsing)
✅ Gmail (read, draft, send, meeting detection)
✅ Fast rule-based email classification
✅ Webcam vision (LLaVA - describe, identify, read, count people)
✅ Screenshot analysis (latest screenshot)
✅ PDF summarization (Mistral local, Gemini with confirmation)
✅ MemPalace persistent memory (522 drawers indexed)
✅ Memory voice commands (remember/recall)
✅ Memory context injection into Brain calls
✅ SSH headless mode (--no-gui)
✅ Android SSH via Termux + Tailscale
✅ Android standalone (proot Ubuntu + phi3 on S10+)
✅ Configurable assistant name
✅ Wake word from config
✅ Capability announcement (dynamic)
✅ Mute mic + mute TTS (separate)
✅ Goodbye message
✅ State labels capitalized
✅ Sleeping particle size reduced
✅ Observer refactored (vision/calendar/gmail/document handlers)
✅ Text input routes through full keyword layer
✅ Timer context manager (utils.py)
✅ Comprehensive test suite

Current Config Structure
yamlpersonalize:
  ai_assistant_name: "A.T.L.A.S."

gui:
  border_color: "#1a6aff"
  border_width: 2
  background_color: "#0a0a0f"
  text_color: "#c8d8e8"
  caption_color: "#8a9ab8"
  heard_color: "#4499ff"
  button_bg: "#1a1a2e"
  button_border: "#2a2a4e"
  button_hover: "#2a2a4e"

llm:
  models:
    classifier:   phi3:mini
    orchestrator: mistral
    code:         qwen2.5-coder:7b-instruct-q4_K_M
  api_models:
    claude:  enabled: false
    gemini:  enabled: true, ask_permission: true

integrations:
  google_calendar:
    enabled: true
    credentials_path: ~/.config/atlas/google_calendar_credentials.json
    token_path: ~/.config/atlas/google_calendar_token.json
  gmail:
    enabled: true

vision:
  enabled: true
  camera_index: 0
  model: llava
  max_storage_mb: 500
  resolution_width: 1280
  resolution_height: 720

memory:
  enabled: true
  wing: atlas
  palace_path: ~/.mempalace/palace
  auto_store_conversations: true

audio:
  use_mock: false
  samplerate: 16000

system:
  use_gpu: true

Known Issues / In Progress
⚠️  FunctionGemma integration — IN PROGRESS
⚠️  Screen capture blocked on COSMIC Wayland (grim not supported)
⚠️  brain_handler.py not yet extracted from observer.py
⚠️  say_helper.py not yet extracted
⚠️  debug=True still in several files (should be False for prod)
⚠️  numpy==1.26.4 pinned due to opencv/gruut conflict
⚠️  pephubclient pandas conflict in requirements.txt (harmless warning)

Roadmap — Priority Order
THIS SPRINT:
  1. FunctionGemma — live weather/stocks/news (in progress)
  2. Email monitor — background inbox watcher + alerts
  3. Slack alerts via webhook
  4. SMS alerts via Twilio
  5. Follow-up commands — context between commands
  6. Self-expanding keyword layer
  7. WorldMonitor integration

NEXT SPRINT:
  8. FastAPI server — REST/WebSocket for mobile clients
  9. Simple web UI — browser access
  10. n8n integration (reconsider after FastAPI)

BIG PROJECTS (after job offer):
  11. Flutter mobile app (native mic/speakers/camera/orb)
  12. Phone node (phi3 on Android distributed)
  13. Persistent memory improvements (recency decay, access boost)
  14. RAG over local notes/files
  15. Mac/PC versions
  16. Docker deployment
  17. Monetization — BSL license, $2/month subscription
  18. Healthcare B2B angle

STRETCH:
  19. Camera — Gemini Vision fallback
  20. Phone camera via IP Webcam
  21. Push-to-talk mode
  22. Summarize screenshot (COSMIC Wayland blocked)

Android Setup
Phone: Samsung Galaxy S10+ (8GB RAM, Snapdragon 855, ARM64)
Access: Termux → SSH → Tailscale → Desktop
Standalone: proot Ubuntu → phi3:mini (5s responses on CPU)
Alias: 'atlas' in ~/.bashrc connects and auto-starts
Desktop ~/.bashrc: auto-starts Atlas on SSH connect
tmux: keeps Atlas running after Termux closes

Security Notes
API keys:     OS keyring (never plaintext)
Gmail/Cal:    OAuth2, token in ~/.config/atlas/
Gemini:       ask_permission: true (user confirms voice commands)
              bypass_permission: True for internal Atlas calls
SSH:          Tailscale encrypted tunnel (no open ports)
n8n:          Not yet installed (security review pending)
Monetization: BSL license planned (public repo stays visible for jobs)

Resume Bullet Points
"Built A.T.L.A.S. — a fully local voice AI assistant with multi-model
orchestration (phi3/Mistral/qwen/Gemini), Google Workspace integration,
computer vision, persistent memory, and Android remote access"

Key talking points:
- Multi-model routing with dynamic context sizing
- Privacy-first: all processing local, cloud opt-in with confirmation
- Google Calendar + Gmail with smart NLP and meeting detection
- Computer vision via LLaVA (webcam + screenshots + PDFs)
- Persistent memory via MemPalace/ChromaDB
- SSH/Android access via Tailscale
- Async Python architecture with PyQt5 GUI
- Secure credential management via OS keyring
- Comprehensive test suite