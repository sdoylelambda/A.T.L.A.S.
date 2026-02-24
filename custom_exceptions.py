class PermissionRequired(Exception):
    """Raised when an API call needs user permission before sending data externally."""
    def __init__(self, model_key: str, prompt: str):
        self.model_key = model_key
        self.prompt = prompt
        super().__init__(f"Permission required to send to {model_key}")


class ModelUnavailable(Exception):
    """Raised when a requested model is disabled or unreachable."""
    def __init__(self, model_key: str):
        self.model_key = model_key
        super().__init__(f"Model '{model_key}' is unavailable or disabled in config.")


class PlanExecutionError(Exception):
    """Raised when a step in an execution plan fails."""
    def __init__(self, step: dict, reason: str):
        self.step = step
        self.reason = reason
        super().__init__(f"Plan step failed: {reason}")
