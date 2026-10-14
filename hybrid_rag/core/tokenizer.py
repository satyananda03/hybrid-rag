"""Token counting: one o200k_base tokenizer shared by ingestion, retrieval and multimodal."""
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Protocol

class TokenizerInterface(Protocol):
    def encode(self, content: str) -> List[int]:
        """Encodes a string into a list of tokens."""
        ...
    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of tokens into a string."""
        ...

class Tokenizer:
    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        """
        Initializes the Tokenizer with a tokenizer model name and a tokenizer instance.
        """
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> List[int]:
        try:
            return self.tokenizer.encode(content)
        except ValueError as e:
            if "special token" not in str(e):
                raise
            try:
                return self.tokenizer.encode(content, disallowed_special=())
            except TypeError:
                raise e

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

_TIKTOKEN_CACHE = Path(__file__).with_name("tiktoken_cache")

class TiktokenTokenizer(Tokenizer):
    """Penghitung token o200k_base untuk semua anggaran token; BPE-nya dibaca dari paket."""

    def __init__(self):
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_TIKTOKEN_CACHE))
        import tiktoken
        super().__init__(model_name="gpt-4o-mini", tokenizer=tiktoken.get_encoding("o200k_base"))

@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    """Satu tokenizer per proses: ingestion, retrieval, dan multimodal menghitung dengan encoder yang sama."""
    return TiktokenTokenizer()