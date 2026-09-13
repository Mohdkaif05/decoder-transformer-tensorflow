from app.inference import GPTInference

# Global inference object
_generator: GPTInference | None = None


def load_model_once():
    """
    Load the model only once when FastAPI starts.
    """
    global _generator

    if _generator is None:
        print("Loading GPT model...")
        _generator = GPTInference()
        print("GPT model loaded successfully.")


def get_generator() -> GPTInference:
    """
    Returns the loaded inference object.
    """
    if _generator is None:
        raise RuntimeError(
            "Model is not loaded. Call load_model_once() first."
        )

    return _generator


def generate_text(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """
    Generate text using the loaded model.
    """
    generator = get_generator()

    return generator.generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )