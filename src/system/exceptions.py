class LLMPermissionDeniedError(Exception):
    def __init__(self, message="LLM model is not supported. Permission Denied."):
        super().__init__(message)