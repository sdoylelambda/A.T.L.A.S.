from modules.light_controller import LightController

LIGHT_KEYWORDS = [
    "turn on the light",
    "turn off the light",
    "lights on",
    "lights off",
    "light on",
    "light off",
    "switch on the light",
    "switch off the light",
    "set the light",
    "set the lights",
    "change the light",
    "change color",
    "change colour",
    "dim the light",
    "dim the lights",
    "brightness",
    "flash the light",
    "flash the lights",
    "strobe the light",
    "strobe the lights",
    "strobe mode",
    "fade the light",
    "fade the lights",
    "pulse the light",
    "pulse the lights",
]


class LightHandler:
    def __init__(self, config):
        self.config  = config
        self._light  = LightController(config)
        self.enabled = config.get("light", {}).get("enabled", True)

    def matches(self, text):
        if not self.enabled:
            return False
        return any(kw in text.lower() for kw in LIGHT_KEYWORDS)

    async def handle(self, text, observer):
        if not self.enabled:
            return "Light control is disabled in config."
        print(f"[Light] Handling: {text}")
        return await self._light.handle_command(text)


def register(observer, config):
    handler = LightHandler(config)
    for keyword in LIGHT_KEYWORDS:
        observer.register_handler(keyword, handler.handle)
    print(f"[Light] Handler registered ({len(LIGHT_KEYWORDS)} keywords)")
    return handler
