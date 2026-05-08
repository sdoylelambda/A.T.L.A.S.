"""
fastapi_server.py

A.T.L.A.S. FastAPI Server
Exposes Atlas to Flutter mobile client over Tailscale.

Runs as a lightweight always-on process (started at boot via systemd).
Starts Atlas automatically if not already running when phone connects.

Endpoints:
    POST /command   — send a text command, get a response
    GET  /status    — check if Atlas is running and its current state

Auth:
    X-API-Key header — set api_key in config.yaml under api_server section

Run manually:
    uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload

Run via systemd:
    see atlas-api.service

Author: Sean Doyle / A.T.L.A.S. Project
"""

import asyncio
import os
import subprocess
import sys
import time
import yaml

from pathlib import Path
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


config     = load_config()
api_key_path = Path.home() / ".config/atlas/api_key"
API_KEY = api_key_path.read_text().strip() if api_key_path.exists() else ""
PORT       = config.get("api_server", {}).get("port", 8000)
PROJECT_DIR = Path(__file__).parent

if not API_KEY:
    print("[Atlas API] WARNING: No api_key set in config.yaml — server is unprotected!")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    if not API_KEY:
        return  # no key configured — open (dev mode)
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Atlas process manager
# ---------------------------------------------------------------------------

class AtlasProcess:
    """
    Manages the Atlas main process lifecycle.
    Starts Atlas headless (--no-gui) if not already running.
    Communicates via the Brain module directly once running.
    """

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._brain = None
        self._observer = None
        self._state = "stopped"
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None  # None = still running

    @property
    def state(self) -> str:
        return self._state

    async def ensure_running(self):
        """Start Atlas if not already running. Idempotent."""
        async with self._lock:
            if self.is_running and self._brain is not None:
                return

            if not self.is_running:
                print("[Atlas API] Atlas not running — starting headless...")
                self._start_process()
                await asyncio.sleep(5)  # give Atlas time to initialise

            if self._brain is None:
                self._load_brain()

    def _start_process(self):
        """Launch Atlas as a subprocess in --no-gui mode."""
        python = sys.executable
        main   = str(PROJECT_DIR / "main.py")
        self._process = subprocess.Popen(
            [python, main, "--no-gui"],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._state = "starting"
        print(f"[Atlas API] Atlas started (pid={self._process.pid})")

    def _load_brain(self):
        """Import and initialise Brain directly for API calls."""
        try:
            # Add project to path so modules resolve correctly
            if str(PROJECT_DIR) not in sys.path:
                sys.path.insert(0, str(PROJECT_DIR))

            from modules.brain import Brain
            cfg = load_config()
            self._brain = Brain(cfg)
            self._state = "listening"
            print("[Atlas API] Brain loaded and ready")
        except Exception as e:
            print(f"[Atlas API] Failed to load Brain: {e}")
            self._state = "error"
            raise

    async def process_command(self, text: str) -> str:
        """
        Send a text command through Brain and return the response.
        Mirrors what Observer does when it receives a voice command.
        """
        await self.ensure_running()

        if self._brain is None:
            raise RuntimeError("Brain not initialised")

        self._state = "thinking"
        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._brain.process,
                text
            )

            # brain.process() returns either a string response or a dict plan.
            # For the phone we always want a speakable string.
            if isinstance(result, dict):
                response = result.get("summary", "Done.")
            else:
                response = str(result) if result else "Done."

            return response
        finally:
            self._state = "listening"

    def stop(self):
        """Gracefully stop Atlas."""
        if self._process and self.is_running:
            self._process.terminate()
            self._process.wait(timeout=10)
            print("[Atlas API] Atlas stopped")
        self._state = "stopped"
        self._brain = None
        self._process = None


# Global process manager instance
atlas = AtlasProcess()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Atlas when the API server boots."""
    print("[Atlas API] Server starting — initialising Atlas...")
    try:
        await atlas.ensure_running()
    except Exception as e:
        print(f"[Atlas API] Warning: Atlas failed to start on boot: {e}")
    yield
    # Shutdown
    print("[Atlas API] Server shutting down...")
    atlas.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="A.T.L.A.S. API",
    description="Autonomous Task and Local AI System — Mobile API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your Tailscale IP range if desired
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CommandRequest(BaseModel):
    text: str


class CommandResponse(BaseModel):
    response: str
    state: str


class StatusResponse(BaseModel):
    atlas_running: bool
    state: str
    pid: int | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Check if Atlas is running and what state it's in.
    Flutter should call this on app launch to decide whether to show
    a loading screen while Atlas boots.
    """
    return StatusResponse(
        atlas_running=atlas.is_running,
        state=atlas.state,
        pid=atlas._process.pid if atlas._process else None,
    )


@app.post("/command", response_model=CommandResponse, dependencies=[Depends(verify_api_key)])
async def post_command(request: CommandRequest):
    """
    Send a text command to Atlas and get a spoken response back.

    Flutter flow:
        1. User speaks → Flutter STT → text
        2. Flutter POST /command {"text": "what's on my calendar"}
        3. Atlas processes → returns {"response": "You have 2 meetings..."}
        4. Flutter passes response to Flutter TTS → phone speaks

    This endpoint may take up to 30s for complex Mistral tasks.
    Set your Flutter http client timeout accordingly.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Command text cannot be empty")

    try:
        response = await atlas.process_command(request.text)
        return CommandResponse(
            response=response,
            state=atlas.state,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Atlas unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Atlas error: {e}")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
