import urllib.parse
import os
import asyncio

from playwright.async_api import async_playwright


class BrowserController:
    def __init__(self, config: dict):
        self.debug = False
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.use_chrome = config["browser"].get("use_chrome", False)
        self.chrome_profile_path = config["browser"].get("chrome_profile_path", None)
        self.use_firefox = config["browser"].get("use_firefox", True)
        self.firefox_profile_path = config["browser"].get("firefox_profile_path", None)
        self.firefox_executable_path = config["browser"].get("firefox_executable_path", None)
        print("[Browser] Lazy init ready")

    # ==========================================
    # Playwright browser control
    # ==========================================
    async def start(self):
        print("[Browser] Launching Playwright browser...")
        self.playwright = await async_playwright().start()

        try:
            if self.use_chrome:
                if not os.path.exists(self.chrome_profile_path):
                    await self._create_profile("chrome")
                    return
                self.context = await self.playwright.chromium.launch_persistent_context(
                    user_data_dir=self.chrome_profile_path,
                    headless=False,
                    channel="chrome",
                    args=["--autoplay-policy=no-user-gesture-required"]
                )

            elif self.use_firefox:
                if self.debug:
                    print(f"[Browser] Profile path: {self.firefox_profile_path}")
                    print(f"[Browser] Profile exists: {os.path.exists(self.firefox_profile_path)}")
                    print(
                        f"[Browser] prefs.js exists: {os.path.exists(os.path.join(self.firefox_profile_path, 'prefs.js'))}")

                if not os.path.exists(os.path.join(self.firefox_profile_path, "prefs.js")):
                    await self._create_profile("firefox")
                    return

                self.context = await self.playwright.firefox.launch_persistent_context(
                    user_data_dir=self.firefox_profile_path,
                    executable_path=self.firefox_executable_path,
                    headless=False,
                    firefox_user_prefs={
                        "browser.downgrade.ignore": True,
                        "browser.sessionstore.resume_from_crash": False,
                        "browser.startup.page": 0,  # blank page on startup
                        "browser.startup.homepage_override.mstone": "ignore",
                        "toolkit.startup.max_resumed_crashes": -1,
                        "browser.shell.checkDefaultBrowser": False,
                        "browser.tabs.warnOnClose": False,
                        "datareporting.policy.dataSubmissionEnabled": False,
                        "datareporting.healthreport.uploadEnabled": False,
                    }
                )

            self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
            if self.debug:
                print(f"[Browser] Ready — page url: {self.page.url}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Browser] Failed to launch: {e}")

    async def _create_profile(self, browser_type: str):
        """First run only — creates browser profile for persistent logins."""
        import subprocess
        import psutil

        if browser_type == "chrome":
            profile_ready = os.path.exists(os.path.join(self.chrome_profile_path, "Default"))
            running_name = "chrome"
            profile_path = self.chrome_profile_path
        else:
            profile_ready = os.path.exists(os.path.join(self.firefox_profile_path, "prefs.js"))
            running_name = "firefox"
            profile_path = self.firefox_profile_path

        if profile_ready:
            print("[Browser] Profile already exists, skipping setup.")
            return

        # check if browser already running
        if self.debug:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and running_name in proc.info['name'].lower():
                    print(f"[Browser] ⚠️  {browser_type} is running — please close it, then restart Atlas.")
                    input("Press Enter once closed...")
                    break

        os.makedirs(profile_path, exist_ok=True)
        print(f"[Browser] First run — setting up {browser_type} profile.")

        if browser_type == "firefox":
            self.playwright = await async_playwright().start()
            self.context = await self.playwright.firefox.launch_persistent_context(
                user_data_dir=profile_path,
                headless=False,
                firefox_user_prefs={
                    "browser.downgrade.ignore": True,
                    "browser.sessionstore.resume_from_crash": False,
                    "browser.startup.page": 0,
                    "browser.startup.homepage_override.mstone": "ignore",
                    "toolkit.startup.max_resumed_crashes": -1,
                    "browser.shell.checkDefaultBrowser": False,
                    "browser.tabs.warnOnClose": False,
                    "datareporting.policy.dataSubmissionEnabled": False,
                    "datareporting.healthreport.uploadEnabled": False,
                }
            )
            self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
            print("[Browser] Profile created. Ready.")
            return

        elif browser_type == "chrome":
            proc = subprocess.Popen([
                "google-chrome",
                "--profile-directory=Jarvis"
            ])
            proc.wait()
            print("[Browser] Profile setup complete. Continuing startup...")

    async def stop(self):
        """Shutdown browser cleanly."""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()

    async def _ensure_browser(self):
        if await self._ensure_page_alive():
            return

        if self.playwright is None:
            if self.debug:
                print("[Browser] First browser command — launching...")
            try:
                await asyncio.wait_for(self.start(), timeout=30)
            except asyncio.TimeoutError:
                print("[Browser] Launch timed out after 30s")
            return

        print("[Browser] Recovering lost browser session...")
        await self.start()
        print("[Browser] Recovered")

    async def _ensure_page_alive(self) -> bool:
        """Check if page is still usable, reset if not."""
        try:
            if self.page is None:
                return False
            await self.page.title()  # lightweight check
            return True
        except Exception:
            print("[Browser] Page was closed, resetting.")
            self.page = None
            self.context = None
            self.browser = None
            self.playwright = None
            return False

    # ==========================================
    # MAIN ROUTER
    # ==========================================
    async def handle_command(self, spoken_text: str):
        text = spoken_text.lower().strip()
        if self.debug:
            print(f"[Browser] handle_command called: '{text}'")

        # check if this is actually a browser command first
        is_browser_command = (
                "youtube" in text or
                "google" in text or
                "search for" in text or
                "look up" in text or
                "navigate to" in text or
                "scroll" in text or
                "zoom" in text or
                "go back" in text or
                "go forward" in text or
                "new tab" in text or
                "close tab" in text or
                "refresh" in text or
                "reload" in text or
                "full screen" in text or
                "fullscreen" in text or
                (self.page and "youtube.com" in self.page.url)  # context aware
        )

        if not is_browser_command:
            return False  # don't touch browser, let brain handle it

        await self._ensure_browser()  # only launch if actually needed
        text = spoken_text.lower().strip()

        # --- GOOGLE SEARCH ---
        if text.startswith("google ") or text.startswith("search for ") or text.startswith("look up "):
            return await self.google_search(text)

        # --- YOUTUBE ---
        if "youtube" in text and not text.startswith("google"):
            await self._ensure_browser()
            query = text.replace("youtube", "").replace("open", "").replace("search", "").strip()
            if self.debug:
                print(f"[Browser] YouTube query: '{query}'")
            if query:
                encoded = urllib.parse.quote_plus(query)
                url = f"https://www.youtube.com/results?search_query={encoded}"
                print(f"[Browser] Navigating to: {url}")
                await self.page.goto(url)
                await self.page.wait_for_load_state("domcontentloaded")
                if self.debug:
                    print(f"[Browser] Current url: {self.page.url}")
            else:
                await self.page.goto("https://www.youtube.com")
            return True

        # Only continue if page is alive
        if not await self._ensure_page_alive():
            return False

        # --- NAVIGATION ---
        if text.startswith("next page") or text == "next":
            return await self.next_page()

        if "click" in text or "select" in text:  # fix this
            return await self.click_result(text)

        if "scroll down" in text or "go down" in text:
            await self.page.mouse.wheel(0, 900)
            return True

        if "scroll up" in text or "go up" in text:
            await self.page.mouse.wheel(0, -900)
            return True

        if "zoom in" in text:
            await self.page.evaluate("document.body.style.zoom = '130%'")
            return True

        if "zoom out" in text:
            await self.page.evaluate("document.body.style.zoom = '70%'")
            return True

        if "zoom reset" in text or "zoom normal" in text:
            await self.page.evaluate("document.body.style.zoom = '100%'")
            return True

        if "go back" in text:
            await self.page.go_back()
            return True

        # after "go back"
        if "go forward" in text:
            await self.page.go_forward()
            return True

        if "refresh" in text or "reload" in text:
            await self.page.reload()
            return True

        if "close tab" in text:
            await self.page.close()
            self.page = await self.context.new_page()
            return True

        if "full screen" in text or "fullscreen" in text:  # fix this
            await self.page.keyboard.press("F11")
            return True

        if "find" in text or "search on page" in text:
            await self.page.keyboard.press("Control+f")
            return True

        if "copy" in text:
            await self.page.keyboard.press("Control+c")
            return True

        if "new window" in text:
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()
            return True

        # navigate directly to URL
        TLDS = (".com", ".org", ".net", ".io", ".dev", ".ai", ".edu", ".gov", ".co", ".tv")

        # navigate to gets forcing ESCALATE mistral gets proper command but does not execute
        if text.startswith("navigate to ") or text.startswith("open "):  # open? conflict with open program?
            for trigger in ["navigate to ", "open "]:
                if text.startswith(trigger):
                    url = text.replace(trigger, "", 1).strip()
                    if "." in url and any(url.endswith(tld) or f"{tld}/" in url for tld in TLDS):
                        if not url.startswith("http"):
                            url = f"https://{url}"
                        await self._ensure_browser()
                        await self.page.goto(url)
                        return True

        if "new tab" in text:
            self.page = await self.context.new_page()
            return True

        if "press enter" in text or "enter" in text:
            await self.page.keyboard.press("Enter")
            return True

        if text.startswith("type "):
            query = text.replace("type ", "", 1)
            await self.page.keyboard.type(query)
            return True

        if "escape" in text or "deselect" in text:
            await self.page.keyboard.press("Escape")
            return True

        return False

    # ==========================================
    # GOOGLE SEARCH (stateful)
    # ==========================================
    async def google_search(self, spoken_text: str):
        await self._ensure_browser()

        words = spoken_text.split()

        if not words:
            return False

        if words[0] == "google":
            query = " ".join(words[1:])
        elif spoken_text.startswith("search for "):
            query = spoken_text.replace("search for ", "", 1)
        elif spoken_text.startswith("look up "):
            query = spoken_text.replace("look up ", "", 1)
        else:
            return False

        if not query:
            print("[Browser] No search query detected.")
            return False

        print(f"[Browser] Google searching: {query}")
        encoded = urllib.parse.quote_plus(query)
        await self.page.goto(f"https://www.google.com/search?q={encoded}")
        return True

    # ==========================================
    # NEXT PAGE
    # ==========================================
    async def next_page(self):
        url = self.page.url
        if "google.com/search" in url:
            try:
                await self.page.click("a#pnnext")  # missing await
                return True
            except Exception as e:
                print(f"[Browser] Next button not found: {e}")
                return False
        await self.page.mouse.wheel(0, 1200)  # missing await
        return True

    # ==========================================
    # CLICK RESULT
    # ==========================================

    # add these to allow commands - currently skipped to short
    async def click_result(self, text: str):
        words = text.split()
        if "first" in words:
            idx = 0
        elif "second" in words:
            idx = 1
        elif "third" in words:
            idx = 2
        elif "fourth" in words:
            idx = 3
        elif "fifth" in words:
            idx = 4
        else:
            idx = 0

        url = self.page.url
        if "google.com/search" in url:
            # try multiple selectors — Google changes these often
            for selector in ["h3", "div.yuRUbf a", "#search a"]:
                results = await self.page.query_selector_all(selector)
                # filter out non-result links
                results = [r for r in results if r is not None]
                if len(results) > idx:
                    await results[idx].click()
                    print(f"[Browser] Clicked result {idx + 1} via {selector}")
                    return True

        # generic fallback
        links = await self.page.query_selector_all("a[href]")
        visible = [l for l in links if await l.is_visible()]
        if len(visible) > idx:
            await visible[idx].click()
            return True

        print("[Browser] No clickable result found")
        return False
