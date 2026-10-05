from model.config import load_config


def count_parameters(config):
    # Tied input/output embeddings are counted once.
    embedding = config.vocab_size * config.hidden_size
    head_dim = config.hidden_size // config.num_attention_heads
    kv_dim = config.num_key_value_heads * head_dim
    attention = (
        config.hidden_size * config.hidden_size  # Q
        + config.hidden_size * kv_dim             # K
        + config.hidden_size * kv_dim             # V
        + config.hidden_size * config.hidden_size # O
    )
    ffn = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    blocks = config.num_layers * (attention + ffn + norms)
    final_norm = config.hidden_size
    return embedding + blocks + final_norm


if __name__ == "__main__":
    config = load_config("configs/aviyan_5b.yaml")
    total = count_parameters(config)
    print("AVIYAN-5B Parameter Count")
    print("=========================")
    print(f"Parameters: {total:,}")
    print(f"Parameters (B): {total / 1_000_000_000:.3f} B")
