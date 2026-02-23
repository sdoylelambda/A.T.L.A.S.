import os
import subprocess
import urllib.parse
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
        # Might be redundant
        # Define Google search triggers
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
    def open_app(self, spoken_text: str) -> str:
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

    # Moved to browser_controller.py

    #     self.google_triggers = [
    #         "search google for",
    #         "google",
    #         "look up",
    #         "search for"
    #     ]
    #
    # # ---------- Google Search ----------
    # def google_search(self, spoken_text: str) -> bool:
    #     words = spoken_text.split()
    #
    #     if not words:
    #         return False
    #
    #     if words[0] == "google":
    #         query = " ".join(words[1:])
    #     elif spoken_text.startswith("search for "):
    #         query = spoken_text.replace("search for ", "", 1)
    #     elif spoken_text.startswith("look up "):
    #         query = spoken_text.replace("look up ", "", 1)
    #     else:
    #         return False
    #
    #     if not query:
    #         print("[Launcher] No search query detected.")
    #         return False
    #
    #     print(f"[Launcher] Google searching: {query}")
    #     encoded_query = urllib.parse.quote_plus(query)
    #
    #     subprocess.Popen([
    #         "xdg-open",
    #         f"https://www.google.com/search?q={encoded_query}"
    #     ])
    #     return True
    # self.google_triggers = [
    #     "search google for",
    #     "google",
    #     "look up",
    #     "search for"
    # ]

    # def google_search(self, spoken_text: str) -> bool:
    #     words = spoken_text.split()
    #
    #     if not words:
    #         return False
    #
    #     if words[0] == "google":
    #         query = " ".join(words[1:])
    #     elif spoken_text.startswith("search for "):
    #         query = spoken_text.replace("search for ", "", 1)
    #     elif spoken_text.startswith("look up "):
    #         query = spoken_text.replace("look up ", "", 1)
    #     else:
    #         return False
    #
    #     if not query:
    #         print("[Launcher] No search query detected.")
    #         return False
    #
    #     print(f"[Launcher] Google searching: {query}")
    #     encoded_query = urllib.parse.quote_plus(query)
    #
    #     subprocess.Popen([
    #         "xdg-open",
    #         f"https://www.google.com/search?q={encoded_query}"
    #     ])
    #     return True

    # def handle_command(self, spoken_text: str) -> bool:
    #     spoken_text = spoken_text.lower()
    #
    #     # Priority 1: Google Search
    #     if self.browser_controller.handle_command(spoken_text):
    #         self.current_app = "browser"
    #         return True
    #
    #     # Priority 2: App Launch
    #     if self.open_app(spoken_text):
    #         self.current_app = spoken_text  # this needs to be just the app name --- UPDATE
    #         return True
    #
    #     # 3️⃣ Active app hotkeys
    #     if self.current_app and self.current_app != "browser":
    #         if self.window_controller.send_command(spoken_text):
    #             return True
    #
    #     print(f"[Launcher] Command not recognized: {spoken_text}")
    #     return False

    def get_current_app(self):
        return self.current_app















# import subprocess
# import shlex
# import os
#
#
# class AppLauncher:
#     def __init__(self):
#         self.apps = {
#             "pycharm": {
#                 "path": "/opt/pycharm/bin/pycharm.sh",
#                 "aliases": [
#                     "pycharm",
#                     "charm",
#                     "idea",
#                     "jetbrains",
#                     "open pycharm",
#                     "launch pycharm",
#                     "python"
#                 ],
#             },
#             "vscode": {
#                 "path": "/usr/bin/code",
#                 "aliases": [
#                     "vscode",
#                     "visual studio code",
#                     "code editor",
#                     "launch vscode",
#                     "code"
#                 ],
#             },
#             "browser": {
#                 "path": "/usr/bin/firefox",
#                 "aliases": [
#                     "firefox",
#                     "browser",
#                     "open browser",
#                     "launch firefox",
#                     "Bowser"
#                 ],
#             },
#             "terminal": {
#                 "path": "/usr/bin/cosmic-term",
#                 "aliases": [
#                     "terminal",
#                     "shell",
#                     "open terminal",
#                     "launch terminal",
#                 ],
#             },
#         }
#
#     def open_app(self, spoken_text: str) -> bool:
#         spoken_text = spoken_text.lower()
#
#         for app_name, app_info in self.apps.items():
#             for alias in app_info["aliases"]:
#                 if alias in spoken_text:
#                     print(f"[Launcher] Launching {app_name}...")
#                     os.system(f'"{app_info["path"]}" &')
#
#                     google_triggers = ["search google for", "google", "look up", "search for"]
#                     for trigger in google_triggers:
#                         if trigger in spoken_text:
#                             print(f'Google search for {spoken_text}')
#                             # Extract query after the trigger phrase
#                             query = spoken_text.split(trigger, 1)[1].strip()
#                             if query:
#                                 # speak(f"Searching Google for {query}")
#                                 os.system(f"xdg-open 'https://www.google.com/search?q={query.replace(' ', '+')}'")
#                                 return True
#                     return True
#
#         print(f"[Launcher] App not recognized in: {spoken_text}")
#         return False

    # def google_search(self, command):
    #     # Might be redundant
    #     # Define Google search triggers
    #     google_triggers = ["search google for", "google", "look up", "search for"]
    #     for trigger in google_triggers:
    #         if trigger in command:
    #             # Extract query after the trigger phrase
    #             query = command.split(trigger, 1)[1].strip()
    #             if query:
    #                 # speak(f"Searching Google for {query}")
    #                 os.system(f"xdg-open 'https://www.google.com/search?q={query.replace(' ', '+')}'")
    #                 return True
    #     return False




    # def open_app(self, spoken_name):
    #     spoken_name = spoken_name.lower()
    #     for key, path in self.apps.items():
    #         if key in spoken_name:
    #             print(f"[Launcher] Opening {key}...")
    #             try:
    #                 subprocess.Popen(shlex.split(path))
    #             except FileNotFoundError:
    #                 print(f"[Launcher] Executable not found: {path}")
    #             return True
    #     return False













# import subprocess
# import shlex
#
#
# class AppLauncher:
#     def __init__(self):
#         self.apps = {
#             "pycharm": "/opt/pycharm/bin/pycharm.sh",
#             "vscode": "/usr/bin/code",
#             "browser": "/usr/bin/firefox",
#         }
#
#     def open_app(self, spoken_name):
#         spoken_name = spoken_name.lower()
#         for key, path in self.apps.items():
#             if key in spoken_name:
#                 print(f"[Launcher] Opening {key}...")
#                 try:
#                     subprocess.Popen(shlex.split(path))
#                 except FileNotFoundError:
#                     print(f"[Launcher] Executable not found: {path}")
#                 return True
#         return False
























# import subprocess
# import shlex
#
#
# class AppLauncher:
#     def __init__(self):
#         self.apps = {
#             "pycharm": "/opt/pycharm/bin/pycharm.sh",
#             "vscode": "/opt/pycharm/bin/code",
#             "browser": "/usr/bin/firefox",
#         }
#
#     def open_app(self, spoken_name):
#         spoken_name = spoken_name.lower()
#         for key, path in self.apps.items():
#             if key in spoken_name:
#                 print(f"[Launcher] Opening {key}...")
#                 try:
#                     subprocess.Popen(shlex.split(path))
#                 except FileNotFoundError:
#                     print(f"[Launcher] Executable not found: {path}")
#                 return True
#         return False

































# class AppLauncher:
#     def __init__(self):
#         self.apps = {
#             "pycharm": "/opt/pycharm/bin/pycharm.sh",
#             "vscode": "/opt/pycharm/bin/code",
#             "browser": "/usr/bin/firefox",
#         }
#
#     def open_app(self, spoken_name):
#         spoken_name = spoken_name.lower()
#         for key, path in self.apps.items():
#             if key in spoken_name:
#                 print(f"[Launcher] Opening {key}...")
#                 try:
#                     subprocess.Popen(shlex.split(path))
#                 except FileNotFoundError:
#                     print(f"[Launcher] Executable not found: {path}")
#                 return True
#         return False




















# class AppLauncher:
#     def __init__(self):
#         # Map spoken names to executable paths
#         self.apps = {
#             "pycharm": "/opt/pycharm/bin/pycharm.sh",
#             "vscode": "/opt/pycharm/bin/code",  # adjust if different
#             "browser": "/usr/bin/firefox",
#         }
#
#     def open_app(self, spoken_name):
#         spoken_name = spoken_name.lower()
#         for key, path in self.apps.items():
#             if key in spoken_name:
#                 print(f"[Launcher] Opening {key}...")
#                 try:
#                     subprocess.Popen(shlex.split(path))
#                 except FileNotFoundError:
#                     print(f"[Launcher] Executable not found: {path}")
#                 return
#         print(f"[Launcher] App not recognized: {spoken_name}")



    # def open_app(self, spoken_name):
    #     # Normalize name
    #     app_name = spoken_name.lower().replace("open ", "").strip()
    #
    #     if app_name in self.apps:
    #         path = self.apps[app_name]
    #         print(f"[Launcher] Opening {app_name}...")
    #         try:
    #             # Use shlex.split in case path has spaces
    #             subprocess.Popen(shlex.split(path))
    #         except FileNotFoundError:
    #             print(f"[Launcher] Executable not found: {path}")
    #     else:
    #         print(f"[Launcher] App not recognized: {app_name}")
