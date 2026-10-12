import asyncio
import bisect
import html
import inspect
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from hashlib import md5
from typing import (
    Annotated,
    Any,
    Protocol,
    Callable,
    TYPE_CHECKING,
    List,
    Optional,
    Iterable,
    Collection,
    NoReturn,
)
import numpy as np
from pydantic import AfterValidator
from hybrid_rag.core.constants import (
    GRAPH_FIELD_SEP,
    VALID_SOURCE_IDS_LIMIT_METHODS,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
)

logger = logging.getLogger(__name__)

# Precompile regex pattern for JSON sanitization (module-level, compiled once)
_SURROGATE_PATTERN = re.compile(r"[\uD800-\uDFFF\uFFFE\uFFFF]")
_CONTROL_CHAR_PATTERN_ALL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)



async def safe_vdb_operation_with_exception(
    operation: Callable,
    operation_name: str,
    entity_name: str = "",
    max_retries: int = 3,
    retry_delay: float = 0.2,
    logger_func: Optional[Callable] = None,
    timeout_seconds: float | None = None,
    log_start: bool = False,
    success_log_threshold_seconds: float = 10.0,
) -> None:
    """
    Safely execute vector database operations with retry mechanism and exception handling.

    This function ensures that VDB operations are executed with proper error handling
    and retry logic. If all retries fail, it raises an exception to maintain data consistency.

    Args:
        operation: The async operation to execute
        operation_name: Operation name for logging purposes
        entity_name: Entity name for logging purposes
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        logger_func: Logger function to use for error messages
        timeout_seconds: Optional timeout for a single operation attempt
        log_start: Whether to emit start/success logs for each attempt
        success_log_threshold_seconds: Log successful attempts when duration exceeds this threshold

    Raises:
        Exception: When operation fails after all retry attempts
    """
    log_func = logger_func or logger.warning

    for attempt in range(max_retries):
        start_ts = time.perf_counter()
        attempt_label = f"{attempt + 1}/{max_retries}"
        try:
            if log_start:
                logger.info(
                    "VDB %s start for %s (attempt %s, timeout=%s)",
                    operation_name,
                    entity_name or "<unknown>",
                    attempt_label,
                    f"{timeout_seconds:.1f}s"
                    if timeout_seconds is not None
                    else "none",
                )

            if timeout_seconds is not None and timeout_seconds > 0:
                await asyncio.wait_for(operation(), timeout=timeout_seconds)
            else:
                await operation()

            elapsed = time.perf_counter() - start_ts
            if log_start or elapsed >= success_log_threshold_seconds:
                logger.info(
                    "VDB %s success for %s in %.2fs (attempt %s)",
                    operation_name,
                    entity_name or "<unknown>",
                    elapsed,
                    attempt_label,
                )
            return  # Success, return immediately
        except asyncio.TimeoutError as e:
            elapsed = time.perf_counter() - start_ts
            timeout_msg = (
                f"VDB {operation_name} timeout for {entity_name or '<unknown>'} "
                f"after {elapsed:.2f}s (attempt {attempt_label}, timeout={timeout_seconds}s)"
            )
            if attempt >= max_retries - 1:
                log_func(timeout_msg)
                raise TimeoutError(timeout_msg) from e
            log_func(f"{timeout_msg}, retrying...")
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)
        except Exception as e:
            elapsed = time.perf_counter() - start_ts
            if attempt >= max_retries - 1:
                error_msg = (
                    f"VDB {operation_name} failed for {entity_name or '<unknown>'} "
                    f"after {max_retries} attempts in {elapsed:.2f}s: {e}"
                )
                log_func(error_msg)
                raise Exception(error_msg) from e
            else:
                log_func(
                    f"VDB {operation_name} attempt {attempt + 1} failed for "
                    f"{entity_name or '<unknown>'} after {elapsed:.2f}s: {e}, retrying..."
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)


# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    pass


@dataclass


@dataclass
class EmbeddingFunc:
    """Embedding function wrapper with dimension validation

    This class wraps an embedding function to ensure that the output embeddings have the correct dimension.
    If wrapped multiple times, the inner wrappers will be automatically unwrapped to prevent
    configuration conflicts where inner wrapper settings would override outer wrapper settings.

    Building one:
        ``pipeline/factory.build_embedding_func`` is the only place this is
        constructed — it wraps the shared LangChain model from
        ``services/embedding_service.get_embeddings``. ``func`` must be an async
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


def compute_args_hash(*args: Any) -> str:
    """Compute a hash for the given arguments with safe Unicode handling.

    Args:
        *args: Arguments to hash
    Returns:
        str: Hash string
    """
    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])

    # Use 'replace' error handling to safely encode problematic Unicode characters
    # This replaces invalid characters with Unicode replacement character (U+FFFD)
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        # Handle surrogate characters and other encoding issues
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + compute_args_hash(content)


def generate_cache_key(mode: str, cache_type: str, hash_value: str) -> str:
    """Generate a flattened cache key in the format {mode}:{cache_type}:{hash}

    Args:
        mode: Cache mode (e.g., 'default', 'local', 'global')
        cache_type: Type of cache (e.g., 'extract', 'query', 'keywords')
        hash_value: Hash value from compute_args_hash

    Returns:
        str: Flattened cache key
    """
    return f"{mode}:{cache_type}:{hash_value}"


# Custom exception classes


class TokenizerInterface(Protocol):
    """
    Defines the interface for a tokenizer, requiring encode and decode methods.
    """

    def encode(self, content: str) -> List[int]:
        """Encodes a string into a list of tokens."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of tokens into a string."""
        ...


class Tokenizer:
    """
    A wrapper around a tokenizer to provide a consistent interface for encoding and decoding.
    """

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        """
        Initializes the Tokenizer with a tokenizer model name and a tokenizer instance.

        Args:
            model_name: The associated model name for the tokenizer.
            tokenizer: An instance of a class implementing the TokenizerInterface.
        """
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> List[int]:
        """
        Encodes a string into a list of tokens using the underlying tokenizer.

        Args:
            content: The string to encode.

        Returns:
            A list of integer tokens.
        """
        try:
            return self.tokenizer.encode(content)
        except ValueError as e:
            # tiktoken (and some other tokenizers) raise ValueError when the
            # content contains literal special-token strings such as
            # "<|endoftext|>", because by default disallowed_special is the
            # full set of special tokens. This crashes document indexing on
            # any user content that happens to contain those strings — common
            # in documentation, notes, or model output captured in source
            # corpora. Retry with disallowed_special=() so the tokens are
            # encoded as ordinary text. Tokenizers that don't accept the
            # kwarg fall through and re-raise the original error.
            if "special token" not in str(e):
                raise
            try:
                return self.tokenizer.encode(content, disallowed_special=())
            except TypeError:
                raise e

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens into a string using the underlying tokenizer.

        Args:
            tokens: A list of integer tokens to decode.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens)


class TiktokenTokenizer(Tokenizer):
    """
    A Tokenizer implementation using the tiktoken library.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the TiktokenTokenizer with a specified model name.

        Args:
            model_name: The model name for the tiktoken tokenizer to use.  Defaults to "gpt-4o-mini".

        Raises:
            ImportError: If tiktoken is not installed.
            ValueError: If the model_name is invalid.
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is not installed. Please install it with `pip install tiktoken` or define custom `tokenizer_func`."
            )

        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
            super().__init__(model_name=model_name, tokenizer=tokenizer)
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}.")


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[int]:
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def normalize_string_list(raw_values: Any, context: str = "") -> list[str]:
    """Return a list of non-empty strings from raw_values.

    Non-string elements are dropped and logged as warnings. If raw_values is
    not a list, an empty list is returned.
    """
    if not isinstance(raw_values, list):
        return []
    result = []
    for i, value in enumerate(raw_values):
        if isinstance(value, str) and value:
            result.append(value)
        else:
            logger.warning(
                "Non-string element dropped from list%s at index %d: %r",
                f" ({context})" if context else "",
                i,
                value,
            )
    return result


def split_text_units_for_hard_fallback(text: str) -> list[str]:
    """Split text into sentence/paragraph-like units for fallback chunking."""
    if not text:
        return []
    units: list[str] = []
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        for sentence in re.split(r"(?<=[。！？；.!?])", p):
            s = sentence.strip()
            if s:
                units.append(s)
    return units if units else [text]


def split_text_by_token_limit(
    text: str, tokenizer: Tokenizer, max_tokens: int
) -> list[str]:
    """Split text by token limit with sentence-first, token-window fallback."""
    if not text:
        return []

    try:
        total_tokens = len(tokenizer.encode(text))
    except Exception:
        total_tokens = 0

    if total_tokens > 0 and total_tokens <= max_tokens:
        return [text]

    units = split_text_units_for_hard_fallback(text)
    out: list[str] = []
    cur_parts: list[str] = []
    cur_tokens = 0

    for unit in units:
        try:
            unit_tokens = len(tokenizer.encode(unit))
        except Exception:
            unit_tokens = 0

        # Sentence itself is oversize: token-window split directly.
        if unit_tokens > max_tokens:
            if cur_parts:
                out.append("\n\n".join(cur_parts))
                cur_parts = []
                cur_tokens = 0

            token_ids = tokenizer.encode(unit)
            for start in range(0, len(token_ids), max_tokens):
                piece = tokenizer.decode(token_ids[start : start + max_tokens]).strip()
                if piece:
                    out.append(piece)
            continue

        if cur_parts and cur_tokens + unit_tokens > max_tokens:
            out.append("\n\n".join(cur_parts))
            cur_parts = [unit]
            cur_tokens = unit_tokens
        else:
            cur_parts.append(unit)
            cur_tokens += unit_tokens

    if cur_parts:
        out.append("\n\n".join(cur_parts))

    return [x for x in out if x.strip()]


def _normalized_child_offsets(
    parent_content: str,
    piece: str,
    search_from: int,
) -> tuple[int, int] | None:
    """Locate ``piece`` in ``parent_content`` ignoring all whitespace.

    Returns ``(start, end)`` char offsets into ``parent_content`` for the first
    whitespace-stripped occurrence at/after ``search_from``, or ``None`` if absent.
    Removing every whitespace char (not collapsing runs) keeps the match exact even
    when the two sides space the same characters differently — the same monotonic
    projection sidecar span backfill relies on.
    """
    norm_piece = "".join(piece.split())
    if not norm_piece:
        return None
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    for idx, ch in enumerate(parent_content):
        if ch.isspace():
            continue
        norm_chars.append(ch)
        norm_to_orig.append(idx)
    norm_parent = "".join(norm_chars)
    # First normalized index whose source offset is >= search_from (norm_to_orig is
    # strictly increasing), so repeated pieces resolve forward in order.
    norm_start = bisect.bisect_left(norm_to_orig, search_from)
    pos = norm_parent.find(norm_piece, norm_start)
    if pos < 0:
        return None
    o_start = norm_to_orig[pos]
    o_end = norm_to_orig[pos + len(norm_piece) - 1] + 1
    return o_start, o_end


def _child_source_span(
    parent_content: str,
    parent_span: Any,
    piece: str,
    search_from: int,
) -> tuple[dict[str, int] | None, int]:
    """Locate a hard-split child ``piece`` inside its parent's source span.

    Pieces are usually verbatim substrings of ``parent_content`` (token-window
    slices), so an exact forward ``find`` resolves them precisely. But
    :func:`split_text_by_token_limit` rejoins multiple sentence units with
    ``"\\n\\n"``, so a multi-unit piece is *not* byte-verbatim when the source
    separated those sentences with a single space/newline. In that case we fall
    back to a whitespace-stripped match (the same projection sidecar backfill uses),
    which stays exact because whitespace removal is monotonic. Without this fallback
    the child would lose its span and sidecar backfill would wrongly FAIL the
    document.

    Returns ``(span | None, next_search_from)`` where ``next_search_from`` is a
    ``parent_content`` offset threaded forward by the caller so repeated pieces
    resolve in order.
    """
    if not isinstance(parent_span, dict):
        return None, search_from
    try:
        parent_start = int(parent_span["start"])
        parent_end = int(parent_span["end"])
    except (KeyError, TypeError, ValueError):
        return None, search_from
    if parent_start < 0 or parent_end < parent_start:
        return None, search_from

    search_from = max(0, search_from)

    # Exact: verbatim token-window pieces.
    local_start = parent_content.find(piece, search_from)
    if local_start >= 0:
        local_end = local_start + len(piece)
    else:
        # Whitespace-normalized fallback: multi-unit pieces rejoined with "\n\n".
        offsets = _normalized_child_offsets(parent_content, piece, search_from)
        if offsets is None:
            return None, search_from
        local_start, local_end = offsets

    if parent_start + local_end > parent_end:
        return None, search_from
    return (
        {"start": parent_start + local_start, "end": parent_start + local_end},
        local_end,
    )


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


async def handle_cache(
    hashing_kv,
    args_hash,
    prompt,
    mode="default",
    cache_type="unknown",
) -> tuple[str, int] | None:
    """Generic cache handling function with flattened cache keys

    Returns:
        tuple[str, int] | None: (content, create_time) if cache hit, None if cache miss
    """
    if hashing_kv is None:
        return None

    if mode != "default":  # handle cache for all type of query
        if not hashing_kv.global_config.get("enable_llm_cache"):
            return None
    else:  # handle cache for entity extraction
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract"):
            return None

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(mode, cache_type, args_hash)
    cache_entry = await hashing_kv.get_by_id(flattened_key)
    if cache_entry:
        logger.debug(f"Flattened cache hit(key:{flattened_key})")
        content = cache_entry["return"]
        timestamp = cache_entry.get("create_time", 0)
        return content, timestamp

    logger.debug(f"Cache missed(mode:{mode} type:{cache_type})")
    return None


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    mode: str = "default"
    cache_type: str = "query"
    chunk_id: str | None = None
    queryparam: dict | None = None


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Save data to cache using flattened key structure.

    Args:
        hashing_kv: The key-value storage for caching
        cache_data: The cache data to save
    """
    # Skip if storage is None or content is a streaming response
    if hashing_kv is None or not cache_data.content:
        return

    # If content is a streaming response, don't cache it
    if hasattr(cache_data.content, "__aiter__"):
        logger.debug("Streaming response detected, skipping cache")
        return

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(
        cache_data.mode, cache_data.cache_type, cache_data.args_hash
    )

    # Check if we already have identical content cached
    existing_cache = await hashing_kv.get_by_id(flattened_key)
    if existing_cache:
        existing_content = existing_cache.get("return")
        if existing_content == cache_data.content:
            logger.warning(
                f"Cache duplication detected for {flattened_key}, skipping update"
            )
            return

    # Create cache entry with flattened structure
    cache_entry = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "chunk_id": cache_data.chunk_id if cache_data.chunk_id is not None else None,
        "original_prompt": cache_data.prompt,
        "queryparam": cache_data.queryparam
        if cache_data.queryparam is not None
        else None,
    }

    logger.info(f" == LLM cache == saving: {flattened_key}")

    # Save using flattened key
    await hashing_kv.upsert({flattened_key: cache_entry})


def get_content_summary(content: str, max_length: int = 250) -> str:
    """Get summary of document content

    Args:
        content: Original document content
        max_length: Maximum length of summary

    Returns:
        Truncated content with ellipsis if needed
    """
    content = content.strip()
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def normalize_extracted_info(name: str, remove_inner_quotes=False) -> str:
    """Normalize entity/relation names and description with the following rules:
    - Clean HTML tags (paragraph and line break tags)
    - Convert Chinese symbols to English symbols
    - Remove spaces between Chinese characters
    - Remove spaces between Chinese characters and English letters/numbers
    - Preserve spaces within English text and numbers
    - Replace Chinese parentheses with English parentheses
    - Replace Chinese dash with English dash
    - Remove English quotation marks from the beginning and end of the text
    - Remove English quotation marks in and around chinese
    - Remove Chinese quotation marks
    - Filter out short numeric-only text (length < 3 and only digits/dots)
    - remove_inner_quotes = True
        remove Chinese quotes
        remove English quotes in and around chinese
        Convert non-breaking spaces to regular spaces
        Convert narrow non-breaking spaces after non-digits to regular spaces

    Args:
        name: Entity name to normalize
        is_entity: Whether this is an entity name (affects quote handling)

    Returns:
        Normalized entity name
    """
    # Clean HTML tags - remove paragraph and line break tags
    name = re.sub(r"</p\s*>|<p\s*>|<p/>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*>|<br/>", "", name, flags=re.IGNORECASE)

    # Chinese full-width letters to half-width (A-Z, a-z)
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )

    # Chinese full-width numbers to half-width
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    # Chinese full-width symbols to half-width
    name = name.replace("－", "-")  # Chinese minus
    name = name.replace("＋", "+")  # Chinese plus
    name = name.replace("／", "/")  # Chinese slash
    name = name.replace("＊", "*")  # Chinese asterisk

    # Replace Chinese parentheses with English parentheses
    name = name.replace("（", "(").replace("）", ")")

    # Replace Chinese dash with English dash (additional patterns)
    name = name.replace("—", "-").replace("－", "-")

    # Chinese full-width space to regular space (after other replacements)
    name = name.replace("　", " ")

    # Use regex to remove spaces between Chinese characters
    # Regex explanation:
    # (?<=[\u4e00-\u9fa5]): Positive lookbehind for Chinese character
    # \s+: One or more whitespace characters
    # (?=[\u4e00-\u9fa5]): Positive lookahead for Chinese character
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)

    # Remove spaces between Chinese and English/numbers/symbols
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    # Remove outer quotes
    if len(name) >= 2:
        # Handle double quotes
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:  # No double quotes inside
                name = inner_content

        # Handle single quotes
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:  # No single quotes inside
                name = inner_content

        # Handle Chinese-style double quotes
        if name.startswith("“") and name.endswith("”"):
            inner_content = name[1:-1]
            if "“" not in inner_content and "”" not in inner_content:
                name = inner_content
        if name.startswith("‘") and name.endswith("’"):
            inner_content = name[1:-1]
            if "‘" not in inner_content and "’" not in inner_content:
                name = inner_content

        # Handle Chinese-style book title mark
        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    if remove_inner_quotes:
        # Remove Chinese quotes
        name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        # Remove English queotes in and around chinese
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
        # Convert non-breaking space to regular space
        name = name.replace("\u00a0", " ")
        # Convert narrow non-breaking space to regular space when after non-digits
        name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    # Remove spaces from the beginning and end of the text
    name = name.strip()

    # Filter out pure numeric content with length < 3
    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""

    def should_filter_by_dots(text):
        """
        Check if the string consists only of dots and digits, with at least one dot
        Filter cases include: 1.2.3, 12.3, .123, 123., 12.3., .1.23 etc.
        """
        return all(c.isdigit() or c == "." for c in text) and "." in text

    if len(name) < 6 and should_filter_by_dots(name):
        # Filter out mixed numeric and dot content with length < 6, requiring at least one dot
        return ""

    return name


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    """Sanitize text to ensure safe UTF-8 encoding by removing or replacing problematic characters.

    This function handles:
    - Surrogate characters (the main cause of encoding errors)
    - Other invalid Unicode sequences
    - Control characters that might cause issues
    - Unescape HTML escapes
    - Remove control characters
    - Whitespace trimming

    Args:
        text: Input text to sanitize
        replacement_char: Character to use for replacing invalid sequences

    Returns:
        Sanitized text that can be safely encoded as UTF-8
    """
    if not text:
        return text

    # First, strip whitespace
    text = text.strip()

    # Early return if text is empty after basic cleaning
    if not text:
        return text

    # 1. html.unescape first to catch entities that might become surrogates or control chars
    text = html.unescape(text)

    # 2. Use pre-compiled regex to clean surrogates and non-characters in one pass
    # This replaces the slow manual loop and initial .encode() check
    text = _SURROGATE_PATTERN.sub(replacement_char, text)

    # 3. Remove control characters but preserve common whitespace (\t, \n, \r)
    text = _CONTROL_CHAR_PATTERN_ALL.sub(replacement_char, text)

    return text.strip()


# LLMs emitting LaTeX inside JSON strings routinely under-escape backslashes:
# "\frac" is *valid* JSON meaning form feed + "rac", so JSON parsers
# (including json_repair) silently decode it and the LaTeX command is
# destroyed. Form feed (\x0c) and backspace (\x08) followed by a letter have
# no legitimate use in LLM-generated prose, so restoring the backslash is
# unconditionally safe. The other three decodable escapes (\t, \n, \r) map to
# legitimate whitespace and cannot be restored without guessing; they are only
# *detected* (see _WS_LATEX_SUSPECT_PATTERN) so real-world frequency can be
# observed before deciding on heuristic restoration.
_FORMFEED_LATEX_PATTERN = re.compile(r"\x0c(?=[A-Za-z])")
_BACKSPACE_LATEX_PATTERN = re.compile(r"\x08(?=[A-Za-z])")
# Whitespace + residue spelling that completes a common LaTeX command whose
# remainder collides with no English word ("eq"/"o"/"exists" are deliberately
# absent: "eq." abbreviations, the word "o"/"exists" would false-positive).
_WS_LATEX_SUSPECT_PATTERN = re.compile(
    r"\t(?=(?:au|heta|imes|ext|ilde|herefore|riangle)\b)"
    r"|\r(?=(?:ho|ight|angle|ceil)\b)"
    r"|\n(?=(?:abla|otin)\b)"
)


def repair_vlm_json_escape_damage(text: str, *, context: str = "") -> str:
    """Restore LaTeX backslashes destroyed by JSON escape decoding.

    Applied to string values parsed out of VLM/LLM JSON responses, where an
    un-doubled LaTeX command like ``"\\frac"`` arrives as ``\\x0c`` + ``rac``.
    Only the two zero-risk cases are repaired:

    - form feed + letter  -> ``\\f`` + letter (``\\frac``, ``\\forall``, ...)
    - backspace + letter  -> ``\\b`` + letter (``\\beta``, ``\\bar``, ...)

    Isolated control characters (not followed by a letter) are left alone for
    downstream sanitization to drop. Whitespace-class damage (``\\tau`` ->
    tab + ``au`` etc.) is ambiguous with legitimate whitespace and is only
    logged at WARNING level, never rewritten.

    Args:
        text: Parsed string value to repair.
        context: Optional label (e.g. ``"table/t1.description"``) included in
            the detection log line.
    """
    if not text:
        return text

    repaired = _FORMFEED_LATEX_PATTERN.sub(r"\\f", text)
    repaired = _BACKSPACE_LATEX_PATTERN.sub(r"\\b", repaired)
    if repaired != text:
        logger.warning(
            "Repaired LaTeX escape damage (\\f/\\b decoded by JSON parser)%s",
            f" in {context}" if context else "",
        )

    suspect = _WS_LATEX_SUSPECT_PATTERN.search(repaired)
    if suspect:
        snippet = repaired[max(0, suspect.start() - 30) : suspect.start() + 30]
        logger.warning(
            "Suspected whitespace-class LaTeX escape damage%s (not auto-repaired): %r",
            f" in {context}" if context else "",
            snippet,
        )

    return repaired


#: A model-written prose field, with LaTeX backslashes the JSON decoding ate put back.
Prose = Annotated[str, AfterValidator(repair_vlm_json_escape_damage)]


def normalize_source_ids_limit_method(method: str | None) -> str:
    """Normalize the source ID limiting strategy and fall back to default when invalid."""

    if not method:
        return SOURCE_IDS_LIMIT_METHOD_KEEP

    normalized = method.upper()
    if normalized not in VALID_SOURCE_IDS_LIMIT_METHODS:
        logger.warning(
            "Unknown SOURCE_IDS_LIMIT_METHOD '%s', falling back to %s",
            method,
            SOURCE_IDS_LIMIT_METHOD_KEEP,
        )
        return SOURCE_IDS_LIMIT_METHOD_KEEP

    return normalized


def subtract_source_ids(
    source_ids: Iterable[str],
    ids_to_remove: Collection[str],
) -> list[str]:
    """Remove a collection of IDs from an ordered iterable while preserving order."""

    removal_set = set(ids_to_remove)
    if not removal_set:
        return [source_id for source_id in source_ids if source_id]

    return [
        source_id
        for source_id in source_ids
        if source_id and source_id not in removal_set
    ]


def raise_first_failure(failures: BaseExceptionGroup) -> NoReturn:
    """Re-raise the first failure of an ``asyncio.TaskGroup`` as itself.

    Callers match on the concrete error (cancellation, flush errors) and show
    its message as the document's ``error_msg``; the group wrapper would hide
    both. The failure keeps its own ``__cause__``.
    """
    failure = failures.exceptions[0]
    raise failure from failure.__cause__


def make_relation_chunk_key(src: str, tgt: str) -> str:
    """Create a deterministic storage key for relation chunk tracking."""

    return GRAPH_FIELD_SEP.join(sorted((src, tgt)))


def generate_track_id(prefix: str = "upload") -> str:
    """Generate a unique tracking ID with timestamp and UUID

    Args:
        prefix: Prefix for the track ID (e.g., 'upload', 'insert')

    Returns:
        str: Unique tracking ID in format: {prefix}_{timestamp}_{uuid}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    return f"{prefix}_{timestamp}_{unique_id}"


def validate_workspace(workspace: str) -> str:
    """Validate a workspace name before it is used to key stored data.

    Every storage in this project (PostgreSQL and Neo4j) partitions its rows,
    nodes and labels by workspace, so the name ends up embedded in queries and
    identifiers. It must therefore be a single, plain component: no path
    separator, no relative path reference.

    Unlike a sanitizing approach, this validator does not rewrite the name.
    Legitimate names containing dots (e.g. ``"v1.0"``) are accepted unchanged,
    while unsafe names are rejected so the caller fails fast instead of
    silently reading or writing another workspace's data.

    Args:
        workspace: Workspace name from configuration or environment variables.

    Returns:
        The workspace name unchanged when it is valid.

    Raises:
        ValueError: If the workspace contains ``/`` or ``\\``, or is ``"."`` or
            ``".."``.

    Examples:
        >>> validate_workspace("my_workspace")
        'my_workspace'
        >>> validate_workspace("v1.0")
        'v1.0'
        >>> validate_workspace("../../../etc")
        Traceback (most recent call last):
            ...
        ValueError: Invalid workspace name '../../../etc': must not contain path separators ('/', '\\') or be a relative path reference ('.', '..')
    """
    if "/" in workspace or "\\" in workspace or workspace in (".", ".."):
        raise ValueError(
            f"Invalid workspace name {workspace!r}: must not contain path "
            "separators ('/', '\\') or be a relative path reference ('.', '..')"
        )
    return workspace
