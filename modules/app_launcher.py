import os
from modules.browser_controller import BrowserController


class AppLauncher:
    def __init__(self, window_controller):
        self.browser_controller = BrowserController()
        self.window_controller = window_controller
        self.current_app = None
        self.apps = {
            "pycharm": {
                "path": "/opt/pycharm/bin/pycharm.sh",
                "aliases": [
                    "pycharm",
                    "charm",
                    "idea",
                    "jetbrains",
                    "open pycharm",
                    "launch pycharm",
                    "python"
                ],
            },
            "vscode": {
                "path": "/usr/bin/code",
                "aliases": [
                    "vscode",
                    "visual studio code",
                    "code editor",
                    "launch vscode",
                    "code"
                ],
            },
            "browser": {
                "path": "/usr/bin/firefox",
                "aliases": [
                    "firefox",
                    "browser",
                    "open browser",
                    "launch firefox",
                    "bowser"
                ],
            },
            "terminal": {
                "path": "/usr/bin/cosmic-term",
                "aliases": [
                    "terminal",
                    "shell",
                    "open terminal",
                    "launch terminal",
                ],
            },
        }


    # ---------- Google Search ----------
    def google_search(self, command):
        google_triggers = ["search google for", "google", "look up", "search for"]
        for trigger in google_triggers:
            if trigger in command:
                # Extract query after the trigger phrase
                query = command.split(trigger, 1)[1].strip()
                if query:
                    # speak(f"Searching Google for {query}")
                    os.system(f"xdg-open 'https://www.google.com/search?q={query.replace(' ', '+')}'")
                    return True
        return False

    # ---------- Open App ----------
    def open_app(self, spoken_text: str):
        for app_name, app_info in self.apps.items():
            for alias in app_info["aliases"]:
                if alias in spoken_text:      #        Why is the word in the sentence, but doesn't execute?
                    print(f"[Launcher] Launching {app_name}...")
                    os.system(f'"{app_info["path"]}" &')
                    self.current_app = app_name
                    self.window_controller.update_active_window(app_name)
                    return f"Opening {app_name}"

        return False

    # ---------- Main Router ----------
    async def handle_command(self, spoken_text: str) -> bool:
        spoken_text = spoken_text.lower().strip()

        # Native app launch first
        if self.open_app(spoken_text):
            self.current_app = self.get_current_app()
            return True

        # Browser search
        if await self.browser_controller.handle_command(spoken_text):
            if self.current_app != "browser":
                self.window_controller.update_active_window("browser")
            self.current_app = "browser"
            return True

        # Hotkeys
        if self.current_app and self.current_app != "browser":
            if self.window_controller.send_command(spoken_text):
                return True

        return False

    def get_current_app(self):
        return self.current_app
