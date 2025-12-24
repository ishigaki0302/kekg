"""Wrapper to make SROTokenizer compatible with HuggingFace tokenizer interface."""

import torch
from typing import List, Union, Optional


class TokenizerWrapper:
    """Wraps SROTokenizer to be compatible with HuggingFace tokenizer interface."""

    def __init__(self, sro_tokenizer):
        """
        Args:
            sro_tokenizer: SROTokenizer instance
        """
        self.sro_tokenizer = sro_tokenizer
        self.vocab_size = sro_tokenizer.vocab_size

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            truncation: Whether to truncate to max_length (ignored if max_length is None)
            max_length: Maximum length to truncate to

        Returns:
            List of token IDs
        """
        token_ids = self.sro_tokenizer.encode(text)

        # Apply truncation if requested and max_length is specified
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.sro_tokenizer.decode(token_ids)

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: bool = False
    ):
        """
        Tokenize text(s) and return dict-like object with input_ids and attention_mask.

        Args:
            text: String or list of strings to tokenize
            return_tensors: If "pt", return PyTorch tensors
            padding: Whether to pad sequences to same length

        Returns:
            Dict-like object with "input_ids" and "attention_mask" and .to() method
        """
        if isinstance(text, str):
            text = [text]

        # Encode all texts
        token_ids_list = [self.sro_tokenizer.encode(t) for t in text]

        if padding:
            # Pad to max length
            max_len = max(len(ids) for ids in token_ids_list)
            padded_ids = []
            attention_masks = []

            for ids in token_ids_list:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [0] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
        else:
            padded_ids = token_ids_list
            attention_masks = [[1] * len(ids) for ids in token_ids_list]

        if return_tensors == "pt":
            input_ids = torch.tensor(padded_ids)
            attention_mask = torch.tensor(attention_masks)
        else:
            input_ids = padded_ids
            attention_mask = attention_masks

        # Return dict-like object with .to() method and list-like slicing
        class TokenizerOutput(dict):
            def to(self, device):
                """Move tensors to device."""
                result = TokenizerOutput()
                for key, value in self.items():
                    if isinstance(value, torch.Tensor):
                        result[key] = value.to(device)
                    else:
                        result[key] = value
                return result

            def __getitem__(self, key):
                """Support both dict access and list slicing."""
                if isinstance(key, str):
                    # Dict-like access: result["input_ids"]
                    return super().__getitem__(key)
                elif isinstance(key, (int, slice)):
                    # List-like access: result[0] or result[0:2]
                    # Return the input_ids at that index/slice
                    input_ids = super().__getitem__("input_ids")
                    if isinstance(input_ids, torch.Tensor):
                        return input_ids[key].tolist()
                    else:
                        return input_ids[key]
                else:
                    raise TypeError(f"Invalid key type: {type(key)}")

        result = TokenizerOutput()
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

        return result
