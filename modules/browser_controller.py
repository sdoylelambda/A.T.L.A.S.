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

        # --- YOUTUBE ---
        if "youtube" in text and not text.startswith("google"):
            await self._ensure_browser()
            query = text.replace("youtube", "").replace("open", "").replace("search", "").strip()
            if query:
                encoded = urllib.parse.quote_plus(query)
                await self.page.goto(f"https://www.youtube.com/results?search_query={encoded}")
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
