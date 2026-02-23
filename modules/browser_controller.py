from playwright.async_api import async_playwright
import urllib.parse


class BrowserController:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        print("[Browser] Lazy init ready")

    # ==========================================
    # Ensure browser exists
    # ==========================================
    async def _ensure_browser(self):
        if await self._ensure_page_alive():
            return

        print("[Browser] Launching Playwright browser...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        print("[Browser] Ready")

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

        # --- GOOGLE SEARCH ---
        if text.startswith("google ") or text.startswith("search for ") or text.startswith("look up "):
            return await self.google_search(text)

        # Only continue if page is alive
        if not await self._ensure_page_alive():
            return False

        # --- NAVIGATION ---
        if text.startswith("next page") or text == "next":
            return await self.next_page()

        if text.startswith("click") or text.startswith("select"):
            return await self.click_result(text)

        if "scroll down" in text:
            await self.page.mouse.wheel(0, 900)
            return True

        if "scroll up" in text:
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

        if "cancel" in text or "deselect" in text:
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
                self.page.click("a#pnnext")
                print("[Browser] Next page")
                return True
            except:  #                                      ADD EXCEPTION TYPE
                print("[Browser] Next button not found")
                return False

        # fallback scroll
        self.page.mouse.wheel(0, 1200)
        return True

    # ==========================================
    # CLICK RESULT
    # ==========================================


    #  THESE ARE NOT WORKING ----------> FIX THIS


    async def click_result(self, text: str):
        words = text.split()

        if "first" in words:
            idx = 0
        elif "second" in words:
            idx = 1
        elif "third" in words:
            idx = 2
        else:
            idx = 0

        url = self.page.url

        # Google results
        if "google.com/search" in url:
            results = self.page.query_selector_all("h3")
            if len(results) > idx:
                results[idx].click()
                print(f"[Browser] Clicked result {idx+1}")
                return True

        # generic fallback
        links = self.page.query_selector_all("a")
        if len(links) > idx:
            links[idx].click()
            return True

        return False
