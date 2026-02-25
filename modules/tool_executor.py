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

    async def execute_plan(self, plan: dict) -> list[str]:
        """
        Execute all steps in a plan.
        Returns list of result strings for Jarvis to speak.
        """
        results = []
        steps = plan.get("steps", [])

        if not steps:
            return [plan.get("summary", "Done.")]

        for step in steps:
            action = step.get("action")
            params = step.get("params", {})

            if action not in self.tools:
                raise PlanExecutionError(step, f"Unknown tool: {action}")

            try:
                result = await self.tools[action](**params)
                results.append(result)
                print(f"[ToolExecutor] ✅ {action}: {result}")
            except PlanExecutionError:
                raise
            except Exception as e:
                raise PlanExecutionError(step, str(e))

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

    async def _generate_code(self, path: str, description: str, **kwargs) -> str:
        """Delegate code generation to code model, then write to file."""
        print(f"[ToolExecutor] Generating code for: {description}")

        code = self.brain.query(
            f"Write complete Python code for: {description}. "
            f"Return only working code. No explanation. No markdown.",
            model_key="code"
        )

        # strip markdown code blocks if model adds them anyway
        if "```" in code:
            lines = code.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            code = "\n".join(lines).strip()

        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        if p.exists():
            raise PlanExecutionError(
                {"action": "generate_code"},
                f"{p} already exists."
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
