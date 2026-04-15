from __future__ import annotations

from typing import List

import torch
from transformers import PreTrainedTokenizer


def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    """Wrap a tokenizer so multi-char Chinese tokens split per character."""

    multichar_tokens = {
        token
        for token in tokenizer.vocab.keys()
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed: list[str] = []
            for token in tokens:
                clean_token = token.replace("▁", "")
                if clean_token in self.multichar_tokens:
                    processed.extend(list(clean_token))
                else:
                    processed.append(token)
            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            tokens = self.tokenize(text, **kwargs)
            return self.tokenizer.convert_tokens_to_ids(tokens)

    return CharTokenizerWrapper(tokenizer)


def get_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")
