import os
import subprocess
import json
from pathlib import Path
from custom_exceptions import PlanExecutionError


class ToolExecutor:
    def __init__(self, app_launcher, browser_controller, brain):
        self.launcher = app_launcher  # reuse existing launcher, don't duplicate
        self.browser = browser_controller
        self.brain = brain
        self.workspace = Path("workspace")  # location projects created by Jarvis are stored.
        self.workspace.mkdir(exist_ok=True)  # create on first run, safe if already exists

        self.tools = {
            "open_app":         self._open_app,
            "create_file":      self._create_file,
            "create_dir":       self._create_dir,
            "write_code":       self._write_code,
            "generate_code":    self._generate_code,
            "read_file":        self._read_file,
            "run_script":       self._run_script,
            "list_dir":         self._list_dir,
            "delete_file":      self._delete_file,
            "web_search":       self._web_search,
            "browser_navigate": self._browser_navigate,
            "browser_search": self._browser_search,
            "browser_navigate": self._browser_navigate,
            "browser_click": self._browser_click,
            "browser_scroll": self._browser_scroll,
        }

    # ─── main entry point ─────────────────────────────────────────────────

    async def execute_plan(self, plan: dict, cancelled=None, on_step=None) -> list[str]:
        """
        Execute all steps in a plan.
        Returns list of result strings for Jarvis to speak.
        """
        results = []
        steps = plan.get("steps", [])
        total = len(steps)
        for i, step in enumerate(steps):
            if cancelled and cancelled():
                print("[ToolExecutor] Cancelled between steps.")
                break

            action = step.get("action")
            params = step.get("params", {})
            tool = self.tools.get(action)

            # report progress before executing
            if on_step:
                on_step(i + 1, total, action)  # ← add

            if not tool:
                print(f"[ToolExecutor] Unknown action: {action}")
                continue

            try:
                result = await tool(**params)
                print(f"[ToolExecutor] ✅ {action}: {result}")
                results.append(result)
            except PlanExecutionError as e:
                print(f"[ToolExecutor] ❌ {action}: {e}")
                results.append(str(e))
            except Exception as e:
                print(f"[ToolExecutor] ❌ {action} unexpected error: {e}")
                results.append(f"Something went wrong with {action}, sir.")

        return results

    # ─── tools ───────────────────────────────────────────────────────────

    async def _open_app(self, app: str, **kwargs) -> str:
        result = self.launcher.open_app(app)
        if result:
            return result
        raise PlanExecutionError({"action": "open_app"}, f"Could not find app: {app}")

    async def _create_file(self, path: str, content: str = "", overwrite: bool = False, **kwargs) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        if p.exists() and not overwrite:
            raise PlanExecutionError(
                {"action": "create_file"},
                f"{p} already exists. Say 'overwrite' to replace it."
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Done, sir."

    async def _create_dir(self, path: str, **kwargs) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        if p.exists():
            raise PlanExecutionError(
                {"action": "create_dir"},
                f"{p} already exists."
            )
        p.mkdir(parents=True)
        return f"Created directory: {p}"

    async def _write_code(self, path: str, content: str, **kwargs) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written code to: {p}"

    async def _generate_code(self, path: str, description: str = "", overwrite: bool = False, **kwargs) -> str:
        """Delegate code generation to code model, then write to file."""
        print(f"[ToolExecutor] Generating code for: {description}")

        # if description looks like code, Mistral passed wrong content
        if "<" in description or "def " in description or len(description) > 500:
            print("[ToolExecutor] Bad description received, using path as context")
            description = f"typical {Path(path).suffix.lower()} file for {Path(path).stem}"

        # determine language from file extension
        ext = Path(path).suffix.lower()
        lang_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".jsx": "React JSX",
            ".ts": "TypeScript",
            ".tsx": "React TSX",
            ".html": "HTML",
            ".css": "CSS",
            ".dart": "Dart",
        }
        lang = lang_map.get(ext, "code")

        code = self.brain.query(
            f"Write complete working {lang} code for: {description}.\n"
            f"RULES:\n"
            f"- Return ONLY raw code, nothing else\n"
            f"- No explanations, no apologies, no comments about requirements\n"
            f"- No markdown, no code blocks, no backticks\n"
            f"- If you don't know exactly what to write, make your best attempt\n"
            f"- Start with the very first line of code immediately\n"
            f"- Never say 'I'm sorry' or 'I cannot' or 'However'\n"
            f"- Just write the best code you can",
            model_key="code"
        )

        # strip markdown code blocks if model adds them anyway
        if "```" in code:
            lines = code.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            code = "\n".join(lines)

        # separate code from explanation text
        lines = code.split("\n")
        code_lines = []
        code_started = False

        for line in lines:
            if not code_started:
                if (line.startswith("def ") or
                        line.startswith("class ") or
                        line.startswith("import ") or
                        line.startswith("from ") or
                        line.startswith("<!") or
                        line.startswith("<html") or
                        line.startswith("const ") or
                        line.startswith("function ") or
                        line.startswith("var ") or
                        line.startswith("let ") or
                        line.startswith("#!") or
                        line.startswith("<?") or
                        line.strip().startswith("@")):
                    code_started = True
                    code_lines.append(line)
            else:
                if line.startswith("This ") or line.startswith("Note:") or line.startswith("Please"):
                    break
                code_lines.append(line)

        if code_lines:
            stripped_lines = [l for l in lines if l not in code_lines]
            if stripped_lines:
                print(f"[ToolExecutor] ⚠️ DeepSeek added {len(stripped_lines)} lines of explanation, commenting out")

                ext = Path(path).suffix.lower()
                if ext in (".html", ".css"):
                    commented = [f"<!-- {l} -->" for l in stripped_lines if l.strip()]
                else:
                    commented = [f"# {l}" for l in stripped_lines if l.strip()]

                code = "\n".join(code_lines) + "\n\n" + "\n".join(commented)
            else:
                code = "\n".join(code_lines)

        # write to file
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p

        if p.exists() and not overwrite:  # ← now resolved
            raise PlanExecutionError(
                {"action": "generate_code"},
                f"{p.name} already exists, sir. Say overwrite to replace it."
            )

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        return f"Done, sir. Code written to: {p}"

    async def _read_file(self, path: str, **kwargs) -> str:
        p = Path(path)
        if not p.exists():
            raise PlanExecutionError({"action": "read_file"}, f"File not found: {path}")
        return p.read_text()

    async def _run_script(self, path: str, **kwargs) -> str:
        result = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            raise PlanExecutionError({"action": "run_script"}, result.stderr)
        return result.stdout or f"Script ran successfully: {path}"

    async def _list_dir(self, path: str = ".", **kwargs) -> str:
        p = Path(path)
        if not p.exists():
            raise PlanExecutionError({"action": "list_dir"}, f"Directory not found: {path}")
        items = [f.name for f in p.iterdir()]
        return f"Contents of {path}: {', '.join(items)}"

    async def _delete_file(self, path: str, **kwargs) -> str:
        p = Path(path)
        if not p.exists():
            raise PlanExecutionError({"action": "delete_file"}, f"Not found: {path}")
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
        else:
            p.unlink()
        return f"Deleted: {path}"

    async def _web_search(self, query: str, **kwargs) -> str:
        self.launcher.google_search(f"search google for {query}")
        return f"Searched for: {query}"

    async def _browser_navigate(self, url: str, **kwargs) -> str:
        subprocess.Popen(["xdg-open", url])
        return f"Navigating to: {url}"

    async def _browser_search(self, query: str, **kwargs) -> str:
        await self.browser.google_search(f"google {query}")
        return f"Searched for: {query}"

    async def _browser_click(self, target: str = "first", **kwargs) -> str:
        await self.browser.click_result(f"click {target}")
        return f"Clicked {target} result"

    async def _browser_scroll(self, direction: str = "down", **kwargs) -> str:
        await self.browser.handle_command(f"scroll {direction}")
        return f"Scrolled {direction}"
