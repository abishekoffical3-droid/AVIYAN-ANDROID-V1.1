from dataclasses import dataclass
from typing import Any
import yaml


@dataclass
class AviyanConfig:
    name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int
    activation: str = "silu"
    normalization: str = "rmsnorm"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_bias: bool = False
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    use_flash_attention: bool = True


def load_raw_config(path: str = "configs/aviyan_5b.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str = "configs/aviyan_5b.yaml") -> AviyanConfig:
    data = load_raw_config(path)
    model = data["model"]
    return AviyanConfig(**{
        k: model[k] for k in AviyanConfig.__dataclass_fields__ if k in model
    })


if __name__ == "__main__":
    config = load_config()
    print("AVIYAN AI Configuration")
    print("========================")
    for key, value in config.__dict__.items():
        print(f"{key:24}: {value}")
