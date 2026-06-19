"""
modules/observer/security_handler.py

Observer handler for security audit voice commands.

Voice commands:
    "verify connection security"
    "run security audit"
    "check security"
    "security check"
    "check for vulnerabilities"

Author: Sean Doyle / A.T.L.A.S. Project
"""

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
    def __init__(self, config: dict):
        self.config = config

    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in SECURITY_KEYWORDS)

    async def handle(self, text: str, observer) -> str:
        print(f"[Security] Running audit: {text}")
        observer.face.set_caption("Running security audit...")

        # run_audit does blocking subprocess calls — run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        audit = await loop.run_in_executor(None, run_audit)

        return format_voice_summary(audit)


def register(observer, config: dict):
    handler = SecurityHandler(config)
    for keyword in SECURITY_KEYWORDS:
        observer.register_handler(keyword, handler.handle)
    print(f"[Security] Handler registered ({len(SECURITY_KEYWORDS)} keywords)")
    return handler
