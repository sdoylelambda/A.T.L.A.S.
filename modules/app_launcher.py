import subprocess
import urllib
import urllib.parse


class AppLauncher:
    def __init__(self, window_controller, browser_controller=None):
        self.window_controller = window_controller
        self.browser_controller = browser_controller
        self.current_app = None
        self.apps = {
            "pycharm": {
                "path": "/opt/pycharm/bin/pycharm.sh",
                "aliases": [
                    "pycharm", "charm", "idea",
                    "jetbrains", "open pycharm",
                    "launch pycharm", "pie charm"
                ],
            },
            "vscode": {
                "path": "/usr/bin/code",
                "aliases": [
                    "vscode", "visual studio code",
                    "code editor", "launch vscode", "v.s."
                ],
            },
            "browser": {
                "path": "/usr/bin/firefox",
                "aliases": [
                    "firefox", "browser", "open browser",
                    "launch firefox", "bowser"
                ],
            },
            "terminal": {
                "path": "/usr/bin/cosmic-term",
                "aliases": [
                    "terminal", "shell",
                    "open terminal", "launch terminal",
                ],
            },
        }

    def google_search(self, command) -> bool:
        google_triggers = ["search google for", "google", "look up", "search for"]
        for trigger in google_triggers:
            if trigger in command:
                query = command.split(trigger, 1)[1].strip()
                if query:
                    url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
                    subprocess.Popen(["xdg-open", url])
                    return True
        return False

    def open_app(self, spoken_text: str) -> str | bool:
        for app_name, app_info in self.apps.items():
            for alias in app_info["aliases"]:
                if alias in spoken_text:
                    try:
                        subprocess.Popen([app_info["path"]])
                        print(f"[Launcher] Launching {app_name}...")
                    except Exception as e:
                        print(f"[Launcher] Failed to launch {app_name}: {e}")
                        return False
                    self.current_app = app_name
                    self.window_controller.update_active_window(app_name)
                    return f"Opening {app_name}"
        return False

    async def dispatch(self, spoken_text: str) -> bool:
        spoken_text = spoken_text.lower().strip()

        if self.open_app(spoken_text):
            self.current_app = self.get_current_app()
            return True

        if await self.browser_controller.handle_command(spoken_text):
            if self.current_app != "browser":
                self.window_controller.update_active_window("browser")
            self.current_app = "browser"
            return True

        if self.current_app and self.current_app != "browser":
            if self.window_controller.send_command(spoken_text):
                return True

        return False

    def get_current_app(self):
        return self.current_app
