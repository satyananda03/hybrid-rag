"""Text cleaning for ingestion: encoding-safe sanitizing, entity-name normalization,
and repair of LaTeX backslashes eaten by JSON decoding."""

import html
import logging
import re
from typing import Annotated

from pydantic import AfterValidator

logger = logging.getLogger(__name__)


# Precompile regex pattern for JSON sanitization (module-level, compiled once)
_SURROGATE_PATTERN = re.compile(r"[\uD800-\uDFFF\uFFFE\uFFFF]")
_CONTROL_CHAR_PATTERN_ALL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


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


def strip_control_characters(text: str, replacement_char: str = "") -> str:
    """Remove control/separator chars and surrogates while preserving text.

    Strips the same character classes as :func:`sanitize_text_for_encoding`
    (surrogates via ``_SURROGATE_PATTERN`` and control chars via
    ``_CONTROL_CHAR_PATTERN_ALL`` — including the C0 separators ``\\x1c``-``\\x1f``
    FS/GS/RS/US, while keeping ``\\t``/``\\n``/``\\r``) but deliberately does
    *not* ``html.unescape`` or ``.strip()`` the result.

    This makes it safe for text that carries intentional markup (e.g. sidecar
    block content with ``<table>``/``<drawing>``/``<equation>`` tags, where
    unescaping ``&lt;`` would corrupt the markup) or significant leading/
    trailing whitespace. For control-char-free input it returns the string
    unchanged, so it does not perturb existing content hashes or snapshots.
    """
    if not text:
        return text
    text = _SURROGATE_PATTERN.sub(replacement_char, text)
    return _CONTROL_CHAR_PATTERN_ALL.sub(replacement_char, text)


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


def sanitize_and_normalize_extracted_text(
    input_text: str, remove_inner_quotes=False
) -> str:
    """Santitize and normalize extracted text
    Args:
        input_text: text string to be processed
        is_name: whether the input text is a entity or relation name

    Returns:
        Santitized and normalized text string
    """
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        normalized_text = normalize_extracted_info(
            safe_input_text, remove_inner_quotes=remove_inner_quotes
        )
        return normalized_text
    return ""


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
