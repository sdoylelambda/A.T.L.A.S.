import faiss
import json
import ollama
import anthropic
import textwrap

from config.api_keys import get_api_key
from anthropic.types import MessageParam
from google import genai
from custom_exceptions import PermissionRequired, ModelUnavailable
from modules.utils import timer
from config.api_keys import get_api_key


class Brain:
    def __init__(self, config: dict):
        self.config = config["llm"]
        self.models = self.config["models"]
        self.api_models = self.config["api_models"]
        self._gemini_client = None

        # FAISS memory (step 11 - RAG, stubbed for now)
        self.vector_db = faiss.IndexFlatL2(384)
        self.memory_texts = []
        self._encoder = None  # lazy load
        self.debug = True

    # ─── core query method ───────────────────────────────────────────────

    def query(self, prompt: str, model_key: str = "orchestrator",
              system: str = None, num_ctx_override: int = None) -> str:
        """Single entry point for all LLM calls."""

        # local ollama models
        if model_key in self.models:
            cfg = self.models[model_key]
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(
                model=cfg["name"],
                messages=messages,
                options={
                    "num_ctx": int(num_ctx_override or cfg.get("num_ctx", 512)),
                    "temperature": float(cfg.get("temperature", 0.1)),
                    "num_predict": int(cfg.get("max_tokens", 500)),
                }
            )
            return response["message"]["content"]

        # claude api
        elif model_key == "claude":
            cfg = self.api_models["claude"]
            if not cfg.get("enabled", False):
                raise ModelUnavailable("claude")
            if cfg.get("ask_permission", True):
                raise PermissionRequired("claude", prompt)
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=cfg["model"],
                max_tokens=int(cfg.get("max_tokens", 1000)),
                system=system or "You are Jarvis, a helpful AI assistant.",
                messages=[MessageParam(role="user", content=prompt)]
            )
            return message.content[0].text

        # gemini api
        elif model_key == "gemini":
            cfg = self.api_models["gemini"]
            if not cfg.get("enabled", False):
                raise ModelUnavailable("gemini")
            if cfg.get("ask_permission", True):
                raise PermissionRequired("gemini", prompt)

            gemini_api_key = get_api_key("gemini")

            if not self._gemini_client:
                self._gemini_client = genai.Client(api_key=gemini_api_key)

            try:
                with timer("Gemini", self.debug):
                    response = self._gemini_client.models.generate_content(
                        model=cfg["model"],
                        contents=prompt
                    )
            except Exception as e:
                if "API key" in str(e):
                    # key was just stored — rebuild client and retry once
                    print("[Brain] Rebuilding Gemini client with new key")
                    self._gemini_client = genai.Client(api_key=get_api_key("gemini"))
                    response = self._gemini_client.models.generate_content(
                        model=cfg["model"],
                        contents=prompt
                    )
                else:
                    raise
            # strip markdown before returning
            import re
            text = response.text
            text = re.sub(r'\*+', '', text)  # remove asterisks
            text = re.sub(r'#{1,6}\s?', '', text)  # remove headers
            text = re.sub(r'`+', '', text)  # remove code ticks
            text = re.sub(r'\n+', ' ', text)  # flatten newlines
            return text.strip()

        else:
            raise ValueError(f"Unknown model key: {model_key}")

    def quick_answer(self, command: str) -> dict | None:
        """
        phi3 attempts to handle the command directly.
        Returns a result dict if it can handle it, None if it needs Mistral.
        """
        system = textwrap.dedent("""
            You are Jarvis, a witty British AI assistant.
            Always say 'sir'. 
            Keep responses as short as possible.

            ESCALATE for anything code or creation related: files, languages, apps, web, scripts, system.
            ESCALATE if unsure. Never write code. Never pretend to do computer tasks.

            # Examples:
            User: create a file
            Jarvis: ESCALATE
            User: write a python class
            Jarvis: ESCALATE
            User: open browser
            Jarvis: ESCALATE
            User: what is the capital of France
            Jarvis: Paris, sir.
            User: what's the address for daytona international speedway
            Jarvis: the address for daytona international speedway 1801 W International Speedway Blvd, Daytona Beach, FL 32114
            User: tell me a joke
            Jarvis: Why don't eggs tell jokes? They'd crack each other up, sir.
            User: how are you
            Jarvis: Fully operational, sir.
            User: what is 2 plus 2
            Jarvis: 4, sir.""").strip()

        result = self.query(command, model_key="classifier", system=system)
        result = result.strip()
        if self.debug:
            print(f"[Brain] phi3 raw: {result[:200]}")

        if "ESCALATE" in result:
            return None

        # Excavate rather than fail
        escalate_phrases = [
            "i'm sorry",
            "i am sorry",
            "i can't",
            "i cannot",
            "i'm unable",
            "i am unable",
            "i don't have",
            "i do not have",
            "as an ai",
            "i'm just an ai",
        ]

        if any(phrase in result.lower() for phrase in escalate_phrases):
            print(f"[Brain] phi3 apologized or refused, forcing ESCALATE")
            return None

        # If trying to return code
        if ("```" in result or
                "folder " in result or
                "project " in result or
                "() " in result or
                "class " in result or
                "import " in result or
                ".txt " in result or
                "file " in result or
                ".py " in result or
                ".js " in result or
                ".ts " in result or
                ".md " in result or
                "html " in result or
                "css " in result or
                "yaml " in result):
            print("[Brain] phi3 returned code, forcing ESCALATE")
            return None

        # Response got cut off mid-sentence
        if result and result[-1] not in ".!?":
            print("[Brain] phi3 response incomplete, forcing ESCALATE")
            return None

        # Adjust if getting to long of a response or unnecessarily escalating
        if len(result) > 300:
            print("[Brain] phi3 response too long, forcing ESCALATE")
            return None

        return {"summary": result, "steps": []}

    # ─── plan creation ────────────────────────────────────────────────────

    def create_plan(self, command: str) -> dict:
        system = textwrap.dedent("""
            You are Jarvis, an AI computer assistant.
            Your only job is to return a valid JSON execution plan. Nothing else.
            Never respond with text, explanations, or ESCALATE. Only JSON.
            For write_code actions, only include a brief skeleton in content.
            Keep all JSON responses under 500 tokens.

            Output format:
            {
              "summary": "one sentence plain english description of what you will do",
              "route": "local|claude|gemini",
              "steps": [
                {"action": "tool_name", "params": {...}}
              ]
            }

            Available tools and their params:
            - create_file: {"path": "filename.txt", "content": "optional short content"}
            - create_dir: {"path": "dirname"}
            - write_code: {"path": "file.py", "content": "code"}  ← simple code only, max 10 lines
            - generate_code: {"path": "file.py", "description": "what the code should do"}  ← use for ANY real code
            - read_file: {"path": "filename.txt"}
            - run_script: {"path": "script.py"}
            - list_dir: {"path": "."}
            - delete_file: {"path": "filename.txt"}
            - web_search: {"query": "search terms"}
            - browser_navigate: {"url": "https://..."}
            - browser_search: {"query": "search terms"}
            
            IMPORTANT RULES:
            - Always use generate_code for ANY code files including .py .js .html .css .ts .jsx
            - Never use write_code and generate_code together for the same file
            - Never put more than 10 lines of code in write_code content
            - Keep JSON responses concise — descriptions only, never actual code content
            - "route" must always be "local", "claude", or "gemini" — never a tool name
    
            Set route to "claude" for complex reasoning or long document analysis.
            Set route to "gemini" for real-time or current information.
            Otherwise route is "local".
    
            Examples:
            User: create a file called hello.txt
            {"summary": "Creating hello.txt in workspace.", "route": "local", "steps": [{"action": "create_file", "params": {"path": "hello.txt", "content": ""}}]}
    
            User: create a folder called projects then add a file called main.py
            {"summary": "Creating projects folder with main.py inside.", "route": "local", "steps": [{"action": "create_dir", "params": {"path": "projects"}}, {"action": "generate_code", "params": {"path": "projects/main.py", "description": "empty python file with main guard"}}]}
    
            User: create a file called backend.py with flask basic methods
            {"summary": "Generating Flask backend in backend.py.", "route": "local", "steps": [{"action": "generate_code", "params": {"path": "backend.py", "description": "Flask app with index route, about route, and REST API endpoint returning JSON"}}]}
    
            User: create a homepage.html with about and contact sections
            {"summary": "Generating homepage.html with sections.", "route": "local", "steps": [{"action": "generate_code", "params": {"path": "homepage.html", "description": "HTML page with inline CSS, home, about us, contact sections"}}]}
    
            User: what is star wars about
            {"summary": "Star Wars is a space opera franchise created by George Lucas.", "route": "local", "steps": []}
    
            Only return JSON. No explanation. No markdown. No code blocks.""").strip()

        command = command.replace("ESCALATE", "").strip()
        num_ctx = self._get_num_ctx(command)

        # log why this ctx was chosen
        command_lower = command.lower()
        code_keywords = ["class", "function", "flask", "react", "html", "css", "javascript", "api"]
        reason = "code" if any(kw in command_lower for kw in code_keywords) else f"{len(command.split())} words"
        print(f"[Brain] num_ctx: {num_ctx} ({reason})")

        result = self.query(
            command,
            model_key="orchestrator",
            system=system,
            num_ctx_override=num_ctx
        )
        print(f"[Brain] Mistral raw response: {result[:200]}")

        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end > start:
                json_str = result[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[Brain] JSON parse failed: {e}")

        return {"summary": result, "steps": [], "route": "local"}

    # ─── main entry point ─────────────────────────────────────────────────

    def process(self, command: str) -> dict:
        """
        Layered routing:
        1. phi3 tries to handle it directly
        2. Mistral handles complex/multistep
        3. API models as last resort
        """
        # layer 1 — phi3 quick answer
        with timer("phi3", self.debug):
            result = self.quick_answer(command)
        if result:
            print("[Brain] Handled by phi3")
            return result

        # layer 2 — mistral full plan
        print("[Brain] Escalating to Mistral")
        with timer("Mistral", self.debug):
            plan = self.create_plan(command)

        # layer 3 — api escalation if mistral flags it
        route = plan.get("route")
        if route in ("claude", "gemini"):
            print(f"[Brain] Escalating to {route}")
            with timer("Gemini/Claude", self.debug):
                response = self.query(command, model_key=route)
            return {"summary": response, "steps": []}

        return plan

    # ─── memory / RAG (step 11) ───────────────────────────────────────────

    @property
    def encoder(self):
        """Lazy load encoder only when memory is used."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._encoder

    def add_memory(self, text: str):
        vec = self.encoder.encode([text])
        self.vector_db.add(vec)
        self.memory_texts.append(text)

    def query_memory(self, query: str, k: int = 5) -> list:
        if not self.memory_texts:
            return []
        q_vec = self.encoder.encode([query])
        _, indices = self.vector_db.search(q_vec, k=k)
        return [self.memory_texts[i] for i in indices[0]]

    # ─── determine tokens needed  ───────────────────────────────────────────

    def _get_num_ctx(self, command: str) -> int:
        words = len(command.split())
        command_lower = command.lower()

        code_keywords = [
            "class", "function", "method", "flask", "django", "react",
            "api", "database", "auth", "authentication", "script",
            "html", "css", "javascript", "typescript", "component",
            "module", "library", "framework", "backend", "frontend",
            "py", "dart", "js", "jsx"
        ]

        # also check for file extensions in command
        code_extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".dart"]
        has_code_extension = any(ext in command_lower for ext in code_extensions)

        is_code = any(kw in command_lower for kw in code_keywords) or has_code_extension

        # multistep commands need more context
        multi_keywords = ["and", "then", "also", "with", "plus", "add"]
        is_multi = sum(1 for kw in multi_keywords if kw in command_lower) >= 2

        if is_code and words > 15:
            return 8192  # complex code generation
        elif is_code:
            return 4096  # simple code generation
        elif is_multi and words > 20:
            return 4096  # complex multi-step
        elif words <= 10:
            return 1024  # simple single commands
        elif words <= 20:
            return 2048  # medium commands
        else:
            return 4096  # long commands
