import tensorflow as tf

from src.config import (
    VOCAB_SIZE,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    D_MODEL,
    NUM_HEADS,
    DFF,
    DROPOUT_RATE,
    MODEL_WEIGHTS_PATH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_K
)

from src.model import GPT
from src.tokenizer import Tokenizer


class GPTInference:
    """
    Handles model loading and text generation.
    """

    def __init__(self):

        # Load tokenizer
        self.tokenizer = Tokenizer()

        # Build model
        self.model = GPT(
            vocab_size=VOCAB_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            dropout_rate=DROPOUT_RATE,
        )

        # Build model before loading weights
        dummy_input = tf.ones((1, MAX_SEQ_LEN), dtype=tf.int32)
        self.model(dummy_input)

        # Load trained weights
        self.model.load_weights(MODEL_WEIGHTS_PATH)

        print("✅ Model loaded successfully.")

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_k: int = TOP_K,
    ) -> str:

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)

        eos_id = self.tokenizer.eos_id

        # Generate tokens
        for _ in range(max_new_tokens):

            # Keep only last context window
            input_ids = input_ids[-MAX_SEQ_LEN:]

            # Convert to tensor
            x = tf.constant([input_ids], dtype=tf.int32)

            # Forward pass
            logits, _ = self.model(
                x,
                training=False,
            )

            # Get logits for last token
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            values, indices = tf.math.top_k(
                logits,
                k=top_k,
            )

            sampled = tf.random.categorical(
                values,
                num_samples=1,
            )[0, 0]

            predicted_id = int(
                indices[0, sampled].numpy()
            )

            # Stop if EOS token generated
            if predicted_id == eos_id:
                break

            input_ids.append(predicted_id)

        # Decode generated sequence
        return self.tokenizer.decode(input_ids)