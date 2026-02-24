import yaml
import ollama
import anthropic
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import json
from custom_exceptions import PermissionRequired, ModelUnavailable, PlanExecutionError


class Brain:
    def __init__(self, config: dict):
        self.config = config["llm"]
        self.models = self.config["models"]
        self.api_models = self.config["api_models"]

        # FAISS memory (step 11 - RAG, stubbed for now)
        self.vector_db = faiss.IndexFlatL2(384)
        self.memory_texts = []
        self._encoder = None  # lazy load

    # ─── core query method ───────────────────────────────────────────────

    def query(self, prompt: str, model_key: str = "orchestrator", system: str = None) -> str:
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
                    "num_ctx": cfg.get("num_ctx", 512),
                    "temperature": cfg.get("temperature", 0.1),
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
                max_tokens=cfg.get("max_tokens", 1000),
                system=system or "You are Jarvis, a helpful AI assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        # gemini api
        elif model_key == "gemini":
            cfg = self.api_models["gemini"]
            if not cfg.get("enabled", False):
                raise ModelUnavailable("gemini")
            if cfg.get("ask_permission", True):
                raise PermissionRequired("gemini", prompt)
            model = genai.GenerativeModel(cfg["model"])
            response = model.generate_content(prompt)
            return response.text

        else:
            raise ValueError(f"Unknown model key: {model_key}")

    # ─── permission bypass ────────────────────────────────────────────────

    def process_with_permission(self, command: str, model_key: str) -> dict:
        """Called after user grants permission — skips ask_permission check."""
        cfg = self.api_models.get(model_key, {})
        if not cfg.get("enabled", False):
            return {"summary": f"{model_key} API is disabled in config.", "steps": []}

        # temporarily bypass permission check by calling API directly
        if model_key == "claude":
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=cfg["model"],
                max_tokens=cfg.get("max_tokens", 1000),
                system="You are Jarvis, a helpful AI assistant.",
                messages=[{"role": "user", "content": command}]
            )
            return {"summary": message.content[0].text, "steps": []}

        elif model_key == "gemini":
            model = genai.GenerativeModel(cfg["model"])
            response = model.generate_content(command)
            return {"summary": response.text, "steps": []}

        return {"summary": "Unknown API model.", "steps": []}

    # ─── intent classification ────────────────────────────────────────────

    def quick_answer(self, command: str) -> dict | None:
        """
        phi3 attempts to handle the command directly.
        Returns a result dict if it can handle it, None if it needs Mistral.
        """
        system = """You are Jarvis, a witty and personable AI assistant with a dry British wit, 
        inspired by Iron Man's Jarvis. Speak naturally and conversationally, never robotic.
        If you can answer or handle this command confidently and completely, do so.
        If it requires multi-step computer tasks, file operations, or code generation, respond with exactly: ESCALATE
        For simple questions, facts, or conversation respond in one sentence max.
        Do not explain yourself. Do not say ESCALATE unless you truly cannot handle it. 
        Reply as short as possible without losing important information."""

        result = self.query(command, model_key="classifier", system=system)
        result = result.strip()

        if "ESCALATE" in result:
            return None  # pass to Mistral

        return {"summary": result, "steps": []}

    # ─── plan creation ────────────────────────────────────────────────────

    def create_plan(self, command: str) -> dict:
        system = """You are Jarvis, an AI assistant that controls a computer.
            Given a command, return a JSON execution plan.
            Output format:
            {
              "summary": "plain english summary of what you will do",
              "route": "local|claude|gemini",
              "steps": [
                {"action": "tool_name", "params": {...}}
              ]
            }
            Set route to "claude" for long document analysis or complex reasoning.
            Set route to "gemini" for anything needing real-time or current information.
            Otherwise route is "local".
            Available tools: create_file, create_dir, write_code, open_app, read_file,
            run_script, web_search, browser_navigate, list_dir, delete_file
            Only return JSON. No explanation."""

        result = self.query(command, model_key="orchestrator", system=system)
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"summary": result, "steps": [], "route": "local"}
            return {"summary": result, "steps": []}

    # ─── main entry point ─────────────────────────────────────────────────

    def process(self, command: str) -> dict:
        """
        Layered routing:
        1. phi3 tries to handle it directly
        2. Mistral handles complex/multi-step
        3. API models as last resort
        """
        # layer 1 — phi3 quick answer
        result = self.quick_answer(command)
        if result:
            print("[Brain] Handled by phi3")
            return result

        # layer 2 — mistral full plan
        print("[Brain] Escalating to Mistral")
        plan = self.create_plan(command)

        # layer 3 — api escalation if mistral flags it
        route = plan.get("route")
        if route in ("claude", "gemini"):
            print(f"[Brain] Escalating to {route}")
            response = self.query(command, model_key=route)
            return {"summary": response, "steps": []}

        return plan

    # ─── memory / RAG (step 11) ───────────────────────────────────────────

    @property
    def encoder(self):
        """Lazy load encoder only when memory is used."""
        if self._encoder is None:
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



# import yaml
# import ollama
# import anthropic
# import google.generativeai as genai
# import faiss
# import json
# from custom_exceptions import PermissionRequired, ModelUnavailable, PlanExecutionError
# from sentence_transformers import SentenceTransformer
#
#
# class Brain:
#     def __init__(self, config_path="./config.yaml"):
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)["llm"]
#
#         self.models = self.config["models"]
#         self.api_models = self.config["api_models"]
#
#         # FAISS memory (step 11 - RAG, stubbed for now)
#         self.vector_db = faiss.IndexFlatL2(384)  # 384 = all-MiniLM-L6-v2 dimension
#         self.memory_texts = []
#         self._encoder = None  # lazy load, only when needed
#
#     # ─── core query method ───────────────────────────────────────────────
#     def _ask_permission(self, model_key: str, prompt: str):
#         """Ask user before sending data to external API."""
#         print(f"\n⚠️  This command would be sent to {model_key.upper()}'s servers.")
#         print(f"Command: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")
#         if cfg.get("ask_permission", True):
#             raise PermissionRequired(model_key, prompt)
#
#     def process_with_permission(self, command: str, model_key: str) -> dict:
#         """Called after user grants permission — skips ask_permission check."""
#         cfg = self.api_models.get(model_key, {})
#         if not cfg.get("enabled", False):
#             return {"summary": f"{model_key} API is disabled in config.", "steps": []}
#
#         response = self.query(command, model_key=model_key)
#         return {"summary": response, "steps": []}
#
#     def query(self, prompt: str, model_key: str = "orchestrator", system: str = None) -> str:
#         """Single entry point for all LLM calls."""
#         def _ask_permission(self, model_key: str, prompt: str):
#             """Ask user before sending data to external API."""
#             print(f"\n⚠️  This command would be sent to {model_key.upper()}'s servers.")
#             print(f"Command: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")
#             if cfg.get("ask_permission", True):
#                 raise PermissionRequired(model_key, prompt)
#
#         # local ollama models
#         if model_key in self.models:
#             cfg = self.models[model_key]
#             messages = []
#             if system:
#                 messages.append({"role": "system", "content": system})
#             messages.append({"role": "user", "content": prompt})
#
#             response = ollama.chat(
#                 model=cfg["name"],
#                 messages=messages,
#                 options={
#                     "num_ctx": cfg.get("num_ctx", 512),
#                     "temperature": cfg.get("temperature", 0.1),
#                 }
#             )
#             return response["message"]["content"]
#
#         # claude api
#         elif model_key == "claude":
#             cfg = self.api_models["claude"]
#             if not cfg.get("enabled", False):
#                 return "Claude API is disabled in config."
#             if cfg.get("ask_permission", True):
#                 if not self._ask_permission("claude", prompt):
#                     return "Cancelled. Handling locally instead."
#             client = anthropic.Anthropic()
#             message = client.messages.create(
#                 model=cfg["model"],
#                 max_tokens=cfg.get("max_tokens", 1000),
#                 system=system or "You are Jarvis, a helpful AI assistant.",
#                 messages=[{"role": "user", "content": prompt}]
#             )
#             return message.content[0].text
#
#         # gemini api
#         elif model_key == "gemini":
#             cfg = self.api_models["gemini"]
#             model = genai.GenerativeModel(cfg["model"])
#             response = model.generate_content(prompt)
#             return response.text
#
#         else:
#             raise ValueError(f"Unknown model key: {model_key}")
#
#     # ─── intent classification ────────────────────────────────────────────
#
#     def classify(self, command: str) -> dict:
#         """Fast intent classification via phi3:mini."""
#         system = """You are an intent classifier. Return JSON only, no explanation.
#             Output format: {"intent": "simple|complex|code|api", "route": "orchestrator|code|claude|gemini"}
#             simple = basic file/app/system task
#             complex = multi-step planning needed
#             code = code generation/editing
#             api = needs real-time info or long context"""
#
#         result = self.query(command, model_key="classifier", system=system)
#         try:
#             return json.loads(result)
#         except json.JSONDecodeError:
#             return {"intent": "simple", "route": "orchestrator"}  # safe fallback
#
#     # ─── plan creation ────────────────────────────────────────────────────
#
#     def create_plan(self, command: str) -> dict:
#         """Orchestrator builds a structured execution plan."""
#         system = """You are Jarvis, an AI assistant that controls a computer.
#             Given a command, return a JSON execution plan.
#             Output format:
#             {
#               "summary": "plain english summary of what you will do",
#               "steps": [
#                 {"action": "tool_name", "params": {...}},
#                 ...
#               ]
#             }
#             Available tools: create_file, create_dir, write_code, open_app, read_file,
#             run_script, web_search, browser_navigate, list_dir, delete_file
#             Only return JSON. No explanation."""
#
#         result = self.query(command, model_key="orchestrator", system=system)
#         try:
#             return json.loads(result)
#         except json.JSONDecodeError:
#             return {"summary": result, "steps": []}
#
#     # ─── main entry point ─────────────────────────────────────────────────
#
#     def process(self, command: str) -> dict:
#         """Classify then route to the right model."""
#         classification = self.classify(command)
#         route = classification.get("route", "orchestrator")
#
#         if route in ("claude", "gemini"):
#             # api escalation - just get a response for now
#             response = self.query(command, model_key=route)
#             return {"summary": response, "steps": []}
#         else:
#             return self.create_plan(command)
#
#     # ─── memory / RAG (step 11) ───────────────────────────────────────────
#
#     @property
#     def encoder(self):
#         """Lazy load encoder only when memory is used."""
#         if self._encoder is None:
#             self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
#         return self._encoder
#
#     def add_memory(self, text: str):
#         vec = self.encoder.encode([text])
#         self.vector_db.add(vec)
#         self.memory_texts.append(text)
#
#     def query_memory(self, query: str, k: int = 5) -> list:
#         if not self.memory_texts:
#             return []
#         q_vec = self.encoder.encode([query])
#         _, indices = self.vector_db.search(q_vec, k=k)
#         return [self.memory_texts[i] for i in indices[0]]






# from sentence_transformers import SentenceTransformer
# import faiss
# import os
# import yaml
# # Example with HuggingFace local LLM
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import whisper
#
#
# class Brain:
#     def __init__(self, config_path="./config.yaml"):
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)
#         # self.model_path = self.config['llm']['model_path']
#         model_name = 'base'  # 'tiny.en' change llm model here
#         self.model = whisper.load_model(model_name, device="cpu")
#
#         print(f"Loading local LLM from {model_name}...")
#         # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#         # self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
#         # self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
#         #
#         # # FAISS memory
#         # self.vector_db = faiss.IndexFlatL2(768) # Placeholder dimension
#         # self.memory_texts = []
#
#     def add_memory(self, text):
#         # Convert text to vector and add
#         from sentence_transformers import SentenceTransformer
#         encoder = SentenceTransformer('all-MiniLM-L6-v2')
#         vec = encoder.encode([text])
#         self.vector_db.add(vec)
#         self.memory_texts.append(text)
#
#     def query_memory(self, query):
#         if len(self.memory_texts) == 0:
#             return []
#         encoder = SentenceTransformer('all-MiniLM-L6-v2')
#         q_vec = encoder.encode([query])
#         D, I = self.vector_db.search(q_vec, k=5)
#         return [self.memory_texts[i] for i in I[0]]
#
#     def create_plan(self, intent: str):
#         print(f"Brain received intent: {intent}")
#         return {
#             "summary": f"I plan to handle: {intent}",
#             "tasks": [{"action": "create_file", "filename": "main.py", "content": "print('Hello World')"}]
#         }



# from sentence_transformers import SentenceTransformer
# Example with HuggingFace local LLM
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# import faiss
# import os
# import yaml


# class Brain:
#     def __init__(self):
#         print("Brain (mock) initialized.")
#
#     def create_plan(self, intent: str):
#         print(f"Brain received intent: {intent}")
#         return {
#             "summary": f"I plan to handle: {intent}",
#             "tasks": [{"action": "create_file", "filename": "main.py", "content": "print('Hello World')"}]
#         }


# class Brain:
#     def __init__(self):
        # with open(config_path, "r") as f:
        #     self.config = yaml.safe_load(f)
        # self.model_path = self.config['llm']['model_path']
        #
        # print(f"Loading local LLM from {self.model_path}...")
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        # self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        #
        # # FAISS memory
        # self.vector_db = faiss.IndexFlatL2(768)  # Placeholder dimension
        # self.memory_texts = []
        # print("Brain (mock) initialized.")

    # def add_memory(self, text):
        # Convert text to vector
    #     encoder = SentenceTransformer('all-MiniLM-L6-v2')
    #     vec = encoder.encode([text])
    #     self.vector_db.add(vec)
    #     self.memory_texts.append(text)
    #     print("Memory added.")
    #
    # def query_memory(self, query):
    #     if len(self.memory_texts) == 0:
    #         return []
    #     encoder = SentenceTransformer('all-MiniLM-L6-v2')
    #     q_vec = encoder.encode([query])
    #     D, I = self.vector_db.search(q_vec, k=5)
    #     return [self.memory_texts[i] for i in I[0]]

    # def create_plan(self, intent: str):
    #     # Query memory for context
    #     # context = self.query_memory(intent)
    #     # prompt = f"Context: {context}\nUser intent: {intent}\nPlan steps:"
    #     # result = self.generator(prompt, max_length=200)[0]['generated_text']
    #     # self.add_memory(f"Plan for '{intent}': {result}")
    #     print(f"Brain received intent: {intent}")
    #     return {
    #         "summary": result,
    #         "tasks": [
    #             {"action": "create_file", "filename": "main.py", "content": "print('Hello World')"}
    #         ]
    #     }


    # MOCK
    # def create_plan(self, intent: str):
    #     print(f"Brain received intent: {intent}")
    #     return {
    #         "summary": f"I plan to: {intent}",
    #         "tasks": [{"action": "create_file", "filename": "main.py", "content": "print('Hello World')"}]
    #     }
