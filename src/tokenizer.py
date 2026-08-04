# src/tokenizer.py

import sentencepiece as spm

from src.config import TOKENIZER_MODEL_PATH


class Tokenizer:
    """
    Wrapper class for the trained SentencePiece tokenizer.
    """

    def __init__(self):
        self.sp = spm.SentencePieceProcessor(
            model_file=TOKENIZER_MODEL_PATH
        )

    def encode(self, text: str) -> list[int]:
        """
        Convert input text to token IDs.
        """
        return self.sp.encode(text, out_type=int)

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back to text.
        """
        return self.sp.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()