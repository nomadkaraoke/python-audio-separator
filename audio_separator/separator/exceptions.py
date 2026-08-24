"""Public exceptions raised by audio validation and separation output handling."""


class InvalidAudioDataError(ValueError):
    """Raised when decoded or generated audio data cannot be processed safely."""


class AudioExportError(RuntimeError):
    """Raised when an audio backend or filesystem cannot publish an output file."""

    def __init__(self, message, *, path, backend):
        super().__init__(message)
        self.path = path
        self.backend = backend


class BatchSeparationError(RuntimeError):
    """Raised after a batch finishes with failed inputs.

    ``successful_files`` contains files from inputs that returned normally. A
    fully published stem from an input that later failed is left on disk but is
    intentionally excluded because that input did not complete its full stem set.
    """

    def __init__(self, successful_files, failures):
        self.successful_files = list(successful_files)
        self.failures = list(failures.items()) if isinstance(failures, dict) else list(failures)
        failure_details = "; ".join(f"{path}: {error}" for path, error in self.failures)
        super().__init__(f"Separation failed for {len(self.failures)} input(s): {failure_details}")
