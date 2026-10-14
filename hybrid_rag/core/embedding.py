"""``EmbeddingFunc``: the embedding contract every vector storage relies on."""

import inspect
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingFunc:
    """Embedding function wrapper with dimension validation

    This class wraps an embedding function to ensure that the output embeddings have the correct dimension.
    If wrapped multiple times, the inner wrappers will be automatically unwrapped to prevent
    configuration conflicts where inner wrapper settings would override outer wrapper settings.

    Building one:
        ``hybrid_rag.providers.build_embedding_func`` is the only place this is
        constructed — it wraps the LangChain model the host installs through
        ``providers.configure(get_embeddings=...)``. ``func`` must be an async
        callable taking ``list[str]`` and returning an ndarray; if it is itself
        an EmbeddingFunc it gets unwrapped (see __post_init__).

    Context-aware embedding:
        The wrapper supports passing a 'context' parameter to distinguish between query and document
        embeddings. This allows wrapped functions to apply different processing (e.g., prefixes,
        different models) based on the context:

        Example:
            embeddings = await embed_func(texts, context="document")  # For indexing
            embeddings = await embed_func([query], context="query")   # For search

    Args:
        embedding_dim: Expected dimension of the embeddings(For dimension checking and workspace data isolation in vector DB)
        func: The actual embedding function to wrap
        max_token_size: Enable embedding token limit checking for description summarization
        send_dimensions: Whether to inject embedding_dim argument to underlying function
        model_name: Model name for implementing workspace data isolation in vector DB
        supports_asymmetric: Whether the underlying function supports context parameter so it can be injected
    """

    embedding_dim: int
    func: callable
    max_token_size: int | None = None
    send_dimensions: bool = False
    model_name: str | None = (
        None  # Model name for implementing workspace data isolation in vector DB
    )
    supports_asymmetric: bool = (
        False  # Whether underlying function accepts context parameter
    )

    def __post_init__(self):
        """Unwrap nested EmbeddingFunc to prevent double wrapping issues.

        When an EmbeddingFunc wraps another EmbeddingFunc, the inner wrapper's
        __call__ preprocessing would override the outer wrapper's settings.
        This method detects and unwraps nested EmbeddingFunc instances to ensure
        that only the outermost wrapper's configuration is applied.
        """
        # Check if func is already an EmbeddingFunc instance and unwrap it
        max_unwrap_depth = 3  # Safety limit to prevent infinite loops
        unwrap_count = 0
        while isinstance(self.func, EmbeddingFunc):
            unwrap_count += 1
            if unwrap_count > max_unwrap_depth:
                raise ValueError(
                    f"EmbeddingFunc unwrap depth exceeded {max_unwrap_depth}. "
                    "Possible circular reference detected."
                )
            # Unwrap to get the original function
            self.func = self.func.func

        if unwrap_count > 0:
            logger.warning(
                f"Detected nested EmbeddingFunc wrapping (depth: {unwrap_count}), "
                "auto-unwrapped to prevent configuration conflicts. "
                "Consider using .func to access the unwrapped function directly."
            )

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        # Only inject embedding_dim when send_dimensions is True
        if self.send_dimensions:
            # Check if user provided embedding_dim parameter
            if "embedding_dim" in kwargs:
                user_provided_dim = kwargs["embedding_dim"]
                # If user's value differs from class attribute, output warning
                if (
                    user_provided_dim is not None
                    and user_provided_dim != self.embedding_dim
                ):
                    logger.warning(
                        f"Ignoring user-provided embedding_dim={user_provided_dim}, "
                        f"using declared embedding_dim={self.embedding_dim} from decorator"
                    )

            # Inject embedding_dim from decorator
            kwargs["embedding_dim"] = self.embedding_dim

        # Remove context parameter if underlying function does not support asymmetric embedding
        if "context" in kwargs and not self.supports_asymmetric:
            # Log when a user-provided context is ignored due to lack of support
            logger.debug(
                "Context parameter was provided but supports_asymmetric=False. The context value has been ignored."
            )
            kwargs.pop("context")

        # Check if underlying function supports max_token_size and inject if not provided
        if self.max_token_size is not None and "max_token_size" not in kwargs:
            sig = inspect.signature(self.func)
            if "max_token_size" in sig.parameters:
                kwargs["max_token_size"] = self.max_token_size

        # Call the actual embedding function
        result = await self.func(*args, **kwargs)

        # Validate embedding dimensions using total element count
        total_elements = result.size  # Total number of elements in the numpy array
        expected_dim = self.embedding_dim

        # Check if total elements can be evenly divided by embedding_dim
        if total_elements % expected_dim != 0:
            raise ValueError(
                f"Embedding dimension mismatch detected: "
                f"total elements ({total_elements}) cannot be evenly divided by "
                f"expected dimension ({expected_dim}). "
            )

        # Optional: Verify vector count matches input text count
        actual_vectors = total_elements // expected_dim
        if args and isinstance(args[0], (list, tuple)):
            expected_vectors = len(args[0])
            if actual_vectors != expected_vectors:
                raise ValueError(
                    f"Vector count mismatch: "
                    f"expected {expected_vectors} vectors but got {actual_vectors} vectors (from embedding result)."
                )

        return result
