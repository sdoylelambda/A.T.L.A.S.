# GEMINI.md - Instructional Context for J.A.R.V.I.S Notes

This directory serves as the documentation, research, and project tracking hub for **J.A.R.V.I.S** (Just A Rather Very Intelligent System), a fully local, voice-controlled AI assistant for Linux.

## Directory Overview
This is a **Non-Code Project** directory containing strategic notes, setup instructions, feature roadmaps, and bug tracking for the J.A.R.V.I.S AI assistant. It provides the architectural context and development goals for the system's multi-layered intelligence (Whisper, Ollama, Mistral, DeepSeek).

## Key Files
- **README.md**: The primary documentation file. It details the project's layered architecture (Fast Keyword -> phi3:mini -> Mistral -> DeepSeek), system requirements, setup checklist, and voice command reference.
- **Jarvis-README.txt**: A secondary reference containing setup instructions for Wayland/X11, MVP goals, stretch goals, and a detailed list of planned features and integrations (e.g., n8n, calendar, Gmail).
- **errors.txt**: A simple log file for tracking current bugs or performance issues (e.g., UI latency during cancellations).
- **Jarvis_Build_AI_Checklist.docx**: A binary document containing a checklist for building or configuring the AI components.
- **GEMINI.md**: This file, providing instructional context for AI interactions within this directory.

## J.A.R.V.I.S Architecture Summary
While this folder contains documentation, it refers to a system with the following architecture:
- **Speech-to-Text (STT)**: Whisper + Faster-Whisper.
- **Classifier**: phi3:mini (fast local model for simple tasks).
- **Orchestrator**: Mistral 7B (complex task planning).
- **Code Generator**: DeepSeek Coder 6.7B.
- **TTS**: Piper.
- **Tooling**: Playwright (browser), xdg-open (apps).

## Usage
The contents of this directory should be used to:
1.  **Reference Project Goals**: Check `README.md` or `Jarvis-README.txt` for the current MVP status and planned features.
2.  **Troubleshoot Setup**: Refer to the "Setup Checklist" and "Troubleshooting" sections in `README.md`.
3.  **Refine AI Behavior**: Use the "Voice Commands" and "Architecture" sections to understand how the system is intended to route user requests.
4.  **Track Progress**: Update the checklist in `Jarvis-README.txt` as features are completed or new bugs are identified.

## Development Context
- **Operating System**: Primarily Linux (Pop!_OS/Wayland).
- **Tech Stack**: Python 3.11+, Ollama (local LLMs), Playwright.
- **Privacy Focus**: All processing is local by default; cloud APIs (Claude/Gemini) require explicit permission.
