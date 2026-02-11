from dataclasses import dataclass

@dataclass
class TransformerModelArgs:
    dim: int = 128
    n_heads: int = 4
    max_seq_len: int = 64
    output_dim: int = 5