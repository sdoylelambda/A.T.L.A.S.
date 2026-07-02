"""
api/fastapi_server.py
Started by main.py internally — not run standalone anymore.
Observer is injected via set_observer() after it starts.
"""

import asyncio
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# Config + Auth
# ---------------------------------------------------------------------------

API_KEY_PATH = Path.home() / ".config/atlas/api_key"

def load_api_key() -> str:
    if API_KEY_PATH.exists():
        return API_KEY_PATH.read_text().strip()
    return ""

API_KEY = load_api_key()

if not API_KEY:
    print("[Atlas API] WARNING: No api_key found — server is unprotected!")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if not API_KEY:
        return
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ---------------------------------------------------------------------------
# Observer injection — set by main.py after Observer starts
# ---------------------------------------------------------------------------

_observer = None

def set_observer(observer):
    global _observer
    _observer = observer
    print("[Atlas API] Observer connected")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="A.T.L.A.S. API",
    description="Autonomous Task and Local AI System — Mobile API",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CommandRequest(BaseModel):
    text: str

class CommandResponse(BaseModel):
    response: str
    state: str

class StatusResponse(BaseModel):
    atlas_running: bool
    state: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        atlas_running=_observer is not None,
        state=_observer._current_state or "listening" if _observer else "stopped",
    )

@app.post("/command", response_model=CommandResponse, dependencies=[Depends(verify_api_key)])
async def post_command(request: CommandRequest):
    if _observer is None:
        raise HTTPException(status_code=503, detail="Atlas not running")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Command text cannot be empty")

    _observer._api_request_pending = True
    while not _observer._api_response_queue.empty():
        _observer._api_response_queue.get_nowait()

    await _observer._text_command_queue.put(request.text)

    try:
        response = await asyncio.wait_for(
            _observer._api_response_queue.get(),
            timeout=60,
        )
        return CommandResponse(response=response, state="listening")
    except asyncio.TimeoutError:
        _observer._api_request_pending = False
        raise HTTPException(status_code=504, detail="Atlas took too long to respond")

@app.post("/cancel", dependencies=[Depends(verify_api_key)])
async def cancel_command():
    if _observer is None:
        raise HTTPException(status_code=503, detail="Atlas not running")
    _observer._cancel_all()
    return {"status": "cancelled"}



# """
# api/fastapi_server.py
#
# A.T.L.A.S. FastAPI Server
# Exposes Atlas to Flutter mobile client over Tailscale.
#
# Mirrors run_text_only() in main.py — starts Observer.listen_and_respond()
# as a background task so the queue is always being processed.
#
# Commands from the phone are injected into Observer's _text_command_queue
# so they run through the exact same pipeline as voice and desktop text:
# keyword layer → phi3 → Mistral → ToolExecutor → calendar/email/vision/etc.
#
# Observer's say() captures the response into _api_response_queue when an
# API request is pending, so FastAPI can return it to the phone.
#
# Endpoints:
#     POST /command   — send a text command, get a response
#     GET  /status    — check if Atlas is running and its current state
#
# Auth:
#     X-API-Key header — key stored in ~/.config/atlas/api_key
#
# Run manually:
#     cd ~/dev/A.T.L.A.S.
#     uvicorn api.fastapi_server:app --host 0.0.0.0 --port 8000
#
# Author: Sean Doyle / A.T.L.A.S. Project
# """
#
# import asyncio
# import sys
# import yaml
# import threading
#
# from pathlib import Path
# from fastapi import FastAPI, HTTPException, Security, Depends
# from fastapi.security import APIKeyHeader
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from contextlib import asynccontextmanager
#
# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
#
# CONFIG_PATH  = Path(__file__).parent.parent / "config.yaml"
# API_KEY_PATH = Path.home() / ".config/atlas/api_key"
# PROJECT_DIR  = Path(__file__).parent.parent
#
#
# def load_config() -> dict:
#     with open(CONFIG_PATH, "r") as f:
#         return yaml.safe_load(f)
#
#
# def load_api_key() -> str:
#     if API_KEY_PATH.exists():
#         return API_KEY_PATH.read_text().strip()
#     return ""
#
#
# API_KEY = load_api_key()
#
# if not API_KEY:
#     print("[Atlas API] WARNING: No api_key found at ~/.config/atlas/api_key — server is unprotected!")
#
# # ---------------------------------------------------------------------------
# # Auth
# # ---------------------------------------------------------------------------
#
# api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
#
#
# async def verify_api_key(key: str = Security(api_key_header)):
#     if not API_KEY:
#         return  # no key configured — open (dev mode only)
#     if key != API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid or missing API key")
#
#
# # ---------------------------------------------------------------------------
# # Atlas manager
# # ---------------------------------------------------------------------------
#
# # class AtlasManager:
# #     """
# #     Manages the Atlas Observer instance.
# #     Starts listen_and_respond() as a background task — same as main.py
# #     run_text_only() — so the command queue is always being processed.
# #     """
# #
# #     def __init__(self):
# #         self._observer       = None
# #         self._observer_task  = None
# #         self._state          = "stopped"
# #         self._lock           = asyncio.Lock()
# #
# #     @property
# #     def is_running(self) -> bool:
# #         return self._observer is not None
# #
# #     @property
# #     def state(self) -> str:
# #         return self._state
# #
# #     async def start(self):
# #         """
# #         Initialize Observer and start listen_and_respond() as a
# #         background task. Mirrors run_text_only() in main.py exactly.
# #         """
# #         async with self._lock:
# #             if self._observer is not None:
# #                 return
# #
# #             if str(PROJECT_DIR) not in sys.path:
# #                 sys.path.insert(0, str(PROJECT_DIR))
# #
# #             from modules.observer.observer import Observer
# #             from modules.window_controller import WindowController
# #             from config.api_keys import set_key_request_callback
# #             from modules.face import FaceController
# #             from main import run_async
# #
# #             config = load_config()
# #
# #             window_controller = WindowController()
# #             face = FaceController(config)
# #             face.show()
# #             self._observer = Observer(face, window_controller, config)
# #
# #             thread = threading.Thread(target=run_async, args=(face, config), daemon=True)
# #             thread.start()
# #
# #             set_key_request_callback(self._observer._request_key_via_gui)
# #
# #             # Start the observer loop as a background task — this is what
# #             # processes the _text_command_queue, exactly like main.py
# #             self._observer_task = asyncio.create_task(
# #                 self._observer.listen_and_respond()
# #             )
# #
# #             self._state = "listening"
# #             print("[Atlas API] Observer started — queue is live")
# #
# #     def stop(self):
# #         if self._observer_task:
# #             self._observer_task.cancel()
# #         self._observer = None
# #         self._state    = "stopped"
# #         print("[Atlas API] Observer stopped")
# #
# #     async def send_command(self, text: str, timeout: int = 60) -> str:
# #         """
# #         Inject a text command into Observer's queue and wait for response.
# #
# #         Flow:
# #             1. Set _api_request_pending = True so say() captures response
# #             2. Put command in _text_command_queue
# #             3. Observer loop picks it up and processes it normally
# #             4. say() fires → sees _api_request_pending → puts text in
# #                _api_response_queue
# #             5. We read and return it
# #         """
# #         if self._observer is None:
# #             raise HTTPException(status_code=503, detail="Atlas not initialised")
# #
# #         obs = self._observer
# #
# #         # Signal to say() that this is an API request
# #         obs._api_request_pending = True
# #
# #         # Drain any stale responses
# #         while not obs._api_response_queue.empty():
# #             obs._api_response_queue.get_nowait()
# #
# #         # Inject into the unified command queue
# #         await obs._text_command_queue.put(text)
# #
# #         self._state = "thinking"
# #         print(f"[Atlas API] Command queued: {text}")
# #
# #         try:
# #             response = await asyncio.wait_for(
# #                 obs._api_response_queue.get(),
# #                 timeout=timeout,
# #             )
# #             print(f"[Atlas API] Response: {response[:80]}")
# #             return response
# #         except asyncio.TimeoutError:
# #             obs._api_request_pending = False
# #             raise HTTPException(
# #                 status_code=504,
# #                 detail="Atlas took too long to respond. Try again."
# #             )
# #         finally:
# #             self._state = "listening"
# #
# #
# # # Global manager instance
# # fastapi_server.py — simplified, no AtlasManager needed
#
# from main import run_async
# from modules.face import FaceController
#
#
# with open("config.yaml") as f:
#     config = yaml.safe_load(f)
#
# face = FaceController(config)
# face.show()
#
# atlas = run_async(face, config)
#
# # ---------------------------------------------------------------------------
# # App lifecycle
# # ---------------------------------------------------------------------------
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("[Atlas API] Server starting...")
#     try:
#         await atlas.start()
#     except Exception as e:
#         print(f"[Atlas API] Warning: Observer failed to start: {e}")
#         import traceback
#         traceback.print_exc()
#     yield
#     print("[Atlas API] Server shutting down.")
#     atlas.stop()
#
#
# # ---------------------------------------------------------------------------
# # App
# # ---------------------------------------------------------------------------
#
# app = FastAPI(
#     title="A.T.L.A.S. API",
#     description="Autonomous Task and Local AI System — Mobile API",
#     version="1.0.0",
#     lifespan=lifespan,
# )
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # ---------------------------------------------------------------------------
# # Models
# # ---------------------------------------------------------------------------
#
# class CommandRequest(BaseModel):
#     text: str
#
#
# class CommandResponse(BaseModel):
#     response: str
#     state:    str
#
#
# class StatusResponse(BaseModel):
#     atlas_running: bool
#     state:         str
#
#
# _observer = None
#
# def set_observer(observer):
#     """Called by main.py to inject the running Observer instance."""
#     global _observer
#     _observer = observer
#     print("[Atlas API] Observer connected")
#
# @app.get("/status")
# async def get_status():
#     return StatusResponse(
#         atlas_running=_observer is not None,
#         state=_observer._current_state or "listening" if _observer else "stopped",
#     )
#
# @app.post("/command", dependencies=[Depends(verify_api_key)])
# async def post_command(request: CommandRequest):
#     if _observer is None:
#         raise HTTPException(status_code=503, detail="Atlas not running")
#
#     _observer._api_request_pending = True
#     while not _observer._api_response_queue.empty():
#         _observer._api_response_queue.get_nowait()
#
#     await _observer._text_command_queue.put(request.text)
#
#     try:
#         response = await asyncio.wait_for(
#             _observer._api_response_queue.get(),
#             timeout=60,
#         )
#         return CommandResponse(response=response, state="listening")
#     except asyncio.TimeoutError:
#         _observer._api_request_pending = False
#         raise HTTPException(status_code=504, detail="Atlas took too long")
#
# @app.post("/cancel", dependencies=[Depends(verify_api_key)])
# async def cancel_command():
#     if _observer is None:
#         raise HTTPException(status_code=503, detail="Atlas not running")
#     _observer._cancel_all()
#     return {"status": "cancelled"}
#
#
# # ---------------------------------------------------------------------------
# # Endpoints
# # ---------------------------------------------------------------------------
#
# @app.get("/status", response_model=StatusResponse)
# async def get_status():
#     """
#     Check if Atlas Observer is running.
#     Flutter calls this on launch to determine Atlas state.
#     """
#     return StatusResponse(
#         atlas_running=atlas.is_running,
#         state=atlas.state,
#     )
#
# @app.post("/cancel", dependencies=[Depends(verify_api_key)])
# async def cancel_command():
#     """Cancel current Atlas command instantly."""
#     if atlas._observer is None:
#         raise HTTPException(status_code=503, detail="Atlas not running")
#     try:
#         atlas._observer._cancel_all()
#         return {"status": "cancelled"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
# @app.post("/command", response_model=CommandResponse, dependencies=[Depends(verify_api_key)])
# async def post_command(request: CommandRequest):
#     """
#     Send a text command through the full Atlas Observer pipeline.
#
#     Everything works — calendar, email, vision, memory, code generation.
#     Same pipeline as voice input on desktop.
#
#     Flutter flow:
#         1. User speaks → Flutter STT → text
#         2. POST /command {"text": "what's on my calendar"}
#         3. Observer processes → say() captures response
#         4. Returns {"response": "You have 2 meetings..."} to Flutter
#         5. Flutter TTS speaks it
#
#     Timeout: 60s for complex Mistral tasks.
#     """
#     if not request.text.strip():
#         raise HTTPException(status_code=400, detail="Command text cannot be empty")
#
#     try:
#         response = await atlas.send_command(request.text)
#         return CommandResponse(
#             response=response,
#             state=atlas.state,
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Atlas error: {e}")
#
#
# # ---------------------------------------------------------------------------
# # Dev entry point
# # ---------------------------------------------------------------------------
#
# if __name__ == "__main__":
#     import uvicorn
#     port = load_config().get("api_server", {}).get("port", 8000)
#     uvicorn.run(
#         "api.fastapi_server:app",
#         host="0.0.0.0",
#         port=port,
#         reload=False,
#     )
