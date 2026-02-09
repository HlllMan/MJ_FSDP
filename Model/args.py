from dataclasses import dataclass

@dataclass
class TransformerModelArgs:
    dim: int = 4096
    n_head: int = 32
    max_seq_len: int = 4096