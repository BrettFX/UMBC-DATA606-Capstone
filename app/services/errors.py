class AppError(Exception):
    """An expected, user-facing problem (bad file, missing model, ...). The message is shown as-is."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint
