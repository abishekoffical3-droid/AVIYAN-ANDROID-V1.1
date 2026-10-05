import torch
import torch.nn.functional as F

from model.aviyan_model import AviyanModel
from model.config import load_config
from tokenizer.load_tokenizer import load_tokenizer


def load_model():
    config = load_config("configs/aviyan_test.yaml")

    model = AviyanModel(config)

    checkpoint = torch.load(
        "checkpoints/aviyan_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def generate(
    model,
    tokenizer,
    text,
    max_new_tokens=20,
    temperature=1.0,
):
    encoded = tokenizer.encode(text)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
    )

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature

        probabilities = F.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    return tokenizer.decode(
        input_ids[0].tolist()
    )


if __name__ == "__main__":
    model, config = load_model()
    tokenizer = load_tokenizer()

    prompt = "Hello"

    output = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=20,
        temperature=0.8,
    )

    print("AVIYAN Generation Test")
    print("======================")
    print("Prompt:", prompt)
    print("Output:", output)
