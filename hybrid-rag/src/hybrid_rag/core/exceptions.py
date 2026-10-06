


class PipelineNotInitializedError(KeyError):
    """Raised when pipeline status is accessed before initialization."""

    def __init__(self, namespace: str = ""):
        msg = (
            f"Pipeline namespace '{namespace}' not found.\n"
            f"\n"
            f"initialize_storages() creates it; call `await rag.initialize_storages()` "
            f"before ingesting into this workspace."
        )
        super().__init__(msg)


class PipelineCancelledException(Exception):
    """Raised when pipeline processing is cancelled by user request."""

    def __init__(self, message: str = "User cancelled"):
        super().__init__(message)
        self.message = message


class IndexFlushError(Exception):
    """Raised when a storage backend fails to flush buffered index ops.

    Carries the storage driver name and namespace so the pipeline can abort
    the batch with an actionable reason. The underlying error is preserved as
    the exception ``__cause__`` (set via ``raise ... from cause``).
    """

    def __init__(self, storage_name: str, namespace: str, cause: BaseException):
        self.storage_name = storage_name
        self.namespace = namespace
        super().__init__(f"{storage_name}[{namespace}] index flush failed: {cause}")


