"""Back-compat shim.

``FastTokenizerWrapper`` is a data-encoding utility (DCT+BPE action
tokenization) consumed by the FAST transform; it now lives in
``transforms.fast_tokenizer``. Import-only re-export; the class is defined
exactly once in its new home.
"""
from src.transforms.fast_tokenizer import FastTokenizerWrapper  # noqa: F401

__all__ = ["FastTokenizerWrapper"]
