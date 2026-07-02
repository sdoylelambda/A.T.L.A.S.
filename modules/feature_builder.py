import os
import json
from datetime import datetime
import re


class FeatureBuilder:
    """
    Atlas Feature Builder
    Converts natural language requests into modular Atlas features.
    """

    def __init__(self, brain, tool_executor, observer, config=None):
        self.brain = brain
        self.tools = tool_executor
        self.observer = observer
        self.config = config or {}
        self.response_name = config["personalize"].get("response_name", "")
        self.feature_root = "modules/features"
        self.pending_feature = None

    def _safe_json_extract(self, text: str):
        # find first JSON object
        # match = re.search(r"\{.*\}", text, re.DOTALL)
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None

        json_str = match.group(0)

        # clean common LLM mistakes
        json_str = json_str.replace("'", '"')  # fix single quotes

        # remove control chars
        json_str = re.sub(r"[\x00-\x1f\x7f]", "", json_str)

        try:
            return json.loads(json_str)
        except Exception as e:
            print("[FeatureBuilder] JSON repair failed:", e)
            print(json_str[:300])
            return None

    # ─────────────────────────────────────────────
    # ENTRY POINT
    # ─────────────────────────────────────────────
    async def handle_request(self, text: str):
        """
        Main entry point for feature creation requests.
        """

        feature_name = self._extract_feature_name(text)

        print(f"[FeatureBuilder] Requested: {feature_name}")

        plan = self._generate_feature_plan(text, feature_name)

        self.pending_feature = plan

        return self._ask_confirmation(plan)

    # ─────────────────────────────────────────────
    # STEP 1 — FEATURE NAME EXTRACTION
    # ─────────────────────────────────────────────
    def _extract_feature_name(self, text: str) -> str:
        """
        Extract feature name from command.
        """
        try:
            lower = text.lower()
            if "called" in lower:
                return lower.split("called")[1].split("that allows")[0].strip().replace(" ", "_")
            return "custom_feature_" + datetime.now().strftime("%H%M%S")
        except:
            return "custom_feature"

    # ─────────────────────────────────────────────
    # STEP 2 — MISTRAL FEATURE DESIGN
    # ─────────────────────────────────────────────
    def _generate_feature_plan(self, text: str, feature_name: str):
        """
        Ask Mistral to design full feature architecture.
        """

        prompt = f"""
    You are Atlas Feature Builder.

    Return ONLY valid JSON. No markdown. No explanation.

    User request:
    {text}

    Feature name:
    {feature_name}

    Schema:
    {{
      "feature_name": "{feature_name}",
      "summary": "string",
      "files": [
        {{
          "path": "modules/features/{feature_name}/controller.py",
          "content": "python code as single string"
        }},
        {{
          "path": "modules/features/{feature_name}/handler.py",
          "content": "python code as single string"
        }}
      ],
      "register_code": "python function as string",
      "dependencies": []
    }}

    RULES:
    - NO backticks
    - NO markdown
    - NO single quotes
    - ALL strings must use double quotes
    - NO multiline literal JSON breaks (use \\n)
    """

        response = self.brain.query(prompt, model_key="orchestrator")

        print(f"[FeatureBuilder] Raw response: {response[:200]}")

        return self._safe_json_extract(response)


#     def _generate_feature_plan(self, text: str, feature_name: str):
#         """
#         Ask Mistral to design full feature architecture.
#         """
#
#         prompt = f"""
# You are Atlas Feature Builder.
#
# Design a fully working modular feature for A.T.L.A.S.
#
# User request:
# {text}
#
# Feature name:
# {feature_name}
#
# Return STRICT JSON ONLY:
#
# {{
#   "feature_name": "{feature_name}",
#   "summary": "...",
#   "files": [
#     {{
#       "path": "modules/features/{feature_name}/controller.py",
#       "content": "..."
#     }},
#     {{
#       "path": "modules/features/{feature_name}/handler.py",
#       "content": "ONE-LINE STRING ONLY (no raw newlines, use \\n instead)"
#     }}
#   ],
#   "register_code": "...python register(observer, brain, tools) function...",
#   "dependencies": []
# }}
#
# Rules:
# - Must integrate with Atlas observer system
# - Must be safe and modular
# - Must NOT modify unrelated files
# - Must include a register() function
# - ALL strings must escape newlines as \\n
# - NO literal line breaks allowed inside JSON strings
# """
#
#         response = self.brain.query(
#             prompt,
#             model_key="orchestrator"
#         )
#
#         try:
#             start = response.find("{")
#             end = response.rfind("}") + 1
#             response = response.replace("\n", "\\n").replace("\r", "\\r")
#             return self._safe_json_extract(response)
#         except Exception as e:
#             print("[FeatureBuilder] JSON parse failed:", e)
#             return None

    # ─────────────────────────────────────────────
    # STEP 3 — CONFIRMATION GATE
    # ─────────────────────────────────────────────
    def _ask_confirmation(self, plan: dict):
        if not plan:
            return "Failed to generate feature plan."

        summary = plan.get("summary", "No summary available")

        return f"""
[Atlas Feature Builder]

I will create a new feature: {plan['feature_name']}

Summary:
{summary}

This will create {len(plan.get('files', []))} file(s).

Shall I proceed, {self.response_name}?
"""

    # ─────────────────────────────────────────────
    # STEP 4 — EXECUTION (FILE CREATION)
    # ─────────────────────────────────────────────
    async def execute(self, approved: bool):
        """
        Execute feature creation after confirmation.
        """

        if not approved:
            self.pending_feature = None
            return "Feature creation cancelled."

        plan = self.pending_feature
        if not plan:
            return "No pending feature."

        feature_name = plan["feature_name"]
        base_path = os.path.join(self.feature_root, feature_name)

        os.makedirs(base_path, exist_ok=True)

        created_files = []

        # ── Create files ─────────────────────────────
        for file in plan.get("files", []):
            path = file["path"]
            content = file["content"]

            try:
                self.tools.create_file(path, content)
                created_files.append(path)
                print(f"[FeatureBuilder] Created: {path}")

            except Exception as e:
                print(f"[FeatureBuilder] Failed file {path}: {e}")

        # ── Write register hook ──────────────────────
        register_path = os.path.join(base_path, "register.py")

        register_code = plan.get("register_code", "")

        if register_code:
            self.tools.create_file(register_path, register_code)
            created_files.append(register_path)

        self.pending_feature = None

        return f"""
[Atlas Feature Builder]

Feature '{feature_name}' created successfully.

Files created:
{chr(10).join(created_files)}

Run 'enable feature {feature_name}' to activate it.
"""

    # ─────────────────────────────────────────────
    # STEP 5 — OBSERVER HOOK
    # ─────────────────────────────────────────────
    def attach(self):
        """
        Attach handler into observer routing system.
        """

        def handler(text):
            if text.lower().startswith("add a new feature"):
                import asyncio
                return asyncio.create_task(self.handle_request(text))
            return None

        # async def handler(text):
        #     if text.lower().startswith("add a new feature"):
        #         return await self.handle_request(text)
        #     return None

        self.observer.register_handler("feature_builder", handler)
