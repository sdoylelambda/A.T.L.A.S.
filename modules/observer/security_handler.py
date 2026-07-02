"""
modules/observer/security_handler.py
"""

import asyncio
import time
from modules.security_audit import run_audit, format_voice_summary

SECURITY_KEYWORDS = [
    "verify connection security",
    "verify security",
    "run security audit",
    "security audit",
    "check security",
    "security check",
    "check for vulnerabilities",
    "vulnerability scan",
    "check connection security",
]


class SecurityHandler:
    def __init__(self, config: dict, observer):
        self.config        = config
        self.observer       = observer
        self._running       = False
        self._last_run_time = 0
        self._cooldown       = 15  # seconds — ignore re-triggers while running or just finished

    def matches(self, text: str) -> bool:
        return any(kw in text.lower() for kw in SECURITY_KEYWORDS)

    async def handle(self, text: str) -> str:
        if not self.matches(text):
            return ""  # not a security command — let it fall through to Brain

        now = time.time()
        if self._running or (now - self._last_run_time) < self._cooldown:
            print(f"[Security] Ignoring duplicate trigger (already running or recent)")
            return ""

        self._running = True
        try:
            print(f"[Security] Running audit: {text}")
            self.observer.face.set_caption("Running security audit...")
            loop = asyncio.get_event_loop()
            audit = await loop.run_in_executor(None, run_audit)
            return format_voice_summary(audit)
        finally:
            self._running = False
            self._last_run_time = time.time()

def register(observer, config: dict):
    handler = SecurityHandler(config, observer)
    for keyword in SECURITY_KEYWORDS:
        observer.register_handler(keyword, handler.handle)
    print(f"[Security] Handler registered ({len(SECURITY_KEYWORDS)} keywords)")
    return handler