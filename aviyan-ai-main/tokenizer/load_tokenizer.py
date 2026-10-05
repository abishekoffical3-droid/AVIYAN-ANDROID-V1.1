from tokenizers import Tokenizer


def load_tokenizer(path="tokenizer/tokenizer.json"):
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    tokenizer = load_tokenizer()

    text = "नमस्ते AVIYAN AI! Hello world."

    encoded = tokenizer.encode(text)

    print("Tokenizer Test")
    print("==============")
    print("Text:", text)
    print("Tokens:", encoded.tokens)
    print("IDs:", encoded.ids)
