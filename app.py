import os
import tensorflow as tf
import sentencepiece as spm

from fastapi import FastAPI
from pydantic import BaseModel, Field

from fastapi.responses import StreamingResponse


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gpt_epoch_6.weights.h5"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "tokenizer",
    "tok.model"
)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Story Generator API",
    version="1.0.0"
)


# --------------------------------------------------
# Transformer Components
# --------------------------------------------------

def scaled_dot_product_attention(q, k, v, mask=None):

    matmul_qk = tf.linalg.matmul(
        q,
        k,
        transpose_b=True
    )

    dk = tf.cast(
        tf.shape(k)[-1],
        q.dtype
    )

    scaled_logits = (
        matmul_qk /
        tf.math.sqrt(dk)
    )

    if mask is not None:
        mask = tf.cast(mask, q.dtype)
        scaled_logits += mask * -1e4

    attention_weights = tf.nn.softmax(
        scaled_logits,
        axis=-1
    )

    output = tf.linalg.matmul(
        attention_weights,
        v
    )

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):

        x = tf.reshape(
            x,
            (
                batch_size,
                -1,
                self.num_heads,
                self.depth
            )
        )

        return tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

    def call(self, x, mask=None):

        batch_size = tf.shape(x)[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = (
            scaled_dot_product_attention(
                q, k, v, mask
            )
        )

        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3]
        )

        concat_attention = tf.reshape(
            scaled_attention,
            (
                batch_size,
                -1,
                self.d_model
            )
        )

        output = self.dense(
            concat_attention
        )

        return output, attention_weights


def positional_encoding(seq_len, d_model):

    position = tf.range(
        seq_len,
        dtype=tf.float32
    )[:, tf.newaxis]

    i = tf.range(
        d_model,
        dtype=tf.float32
    )[tf.newaxis, :]

    angle_rates = 1 / tf.pow(
        10000.0,
        (
            2 * tf.floor(i / 2)
        ) / tf.cast(
            d_model,
            tf.float32
        )
    )

    angle_rads = position * angle_rates

    sines = tf.sin(
        angle_rads[:, 0::2]
    )

    cosines = tf.cos(
        angle_rads[:, 1::2]
    )

    pos_encoding = tf.reshape(
        tf.stack(
            [sines, cosines],
            axis=-1
        ),
        (seq_len, d_model)
    )

    return pos_encoding[tf.newaxis, ...]


class PositionalEncoding(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        max_seq_len,
        d_model
    ):
        super().__init__()

        self.pos_encoding = positional_encoding(
            max_seq_len,
            d_model
        )

    def call(self, x):

        seq_len = tf.shape(x)[1]

        return (
            x +
            tf.cast(
                self.pos_encoding[:, :seq_len, :],
                x.dtype
            )
        )


class FeedForwardNetwork(
    tf.keras.layers.Layer
):

    def __init__(self, d_model, dff):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(
            dff,
            activation="gelu"
        )

        self.dense2 = tf.keras.layers.Dense(
            d_model
        )

    def call(self, x):

        x = self.dense1(x)
        x = self.dense2(x)

        return x


def create_look_ahead_mask(seq_len):

    return 1 - tf.linalg.band_part(
        tf.ones(
            (seq_len, seq_len),
            dtype=tf.float32
        ),
        -1,
        0
    )


class DecoderBlock(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        d_model,
        num_heads,
        dff,
        dropout_rate=0.1
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ffn = FeedForwardNetwork(
            d_model,
            dff
        )

        self.layernorm1 = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )

        self.layernorm2 = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )

        self.dropout1 = tf.keras.layers.Dropout(
            dropout_rate
        )

        self.dropout2 = tf.keras.layers.Dropout(
            dropout_rate
        )

    def call(
        self,
        x,
        training=False,
        mask=None
    ):

        attn_output, attn_weights = self.mha(
            x,
            mask
        )

        attn_output = self.dropout1(
            attn_output,
            training=training
        )

        out1 = self.layernorm1(
            x + attn_output
        )

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        out2 = self.layernorm2(
            out1 + ffn_output
        )

        return out2, attn_weights


class GPT(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        num_layers,
        d_model,
        num_heads,
        dff,
        dropout_rate=0.1
    ):
        super().__init__()

        self.d_model = d_model

        self.embedding = (
            tf.keras.layers.Embedding(
                vocab_size,
                d_model
            )
        )

        self.pos_encoding = (
            PositionalEncoding(
                max_seq_len,
                d_model
            )
        )

        self.decoder_blocks = []

        for _ in range(num_layers):

            self.decoder_blocks.append(
                DecoderBlock(
                    d_model,
                    num_heads,
                    dff,
                    dropout_rate
                )
            )

        self.dropout = tf.keras.layers.Dropout(
            dropout_rate
        )

        self.final_layer = (
            tf.keras.layers.Dense(
                vocab_size
            )
        )

    def call(self, x, training=False):

        seq_len = tf.shape(x)[1]

        mask = create_look_ahead_mask(
            seq_len
        )

        mask = mask[
            tf.newaxis,
            tf.newaxis,
            :,
            :
        ]

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(
                self.d_model,
                x.dtype
            )
        )

        x = self.pos_encoding(x)

        x = self.dropout(
            x,
            training=training
        )

        attention_weights = {}

        for i, block in enumerate(
            self.decoder_blocks
        ):

            x, attn = block(
                x,
                training=training,
                mask=mask
            )

            attention_weights[
                f"block_{i+1}"
            ] = attn

        logits = self.final_layer(x)

        return logits, attention_weights


# --------------------------------------------------
# Load Tokenizer + Model
# --------------------------------------------------

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

eos_id = sp.eos_id()

model = GPT(
    vocab_size=8000,
    max_seq_len=128,
    num_layers=4,
    d_model=256,
    num_heads=4,
    dff=1024,
    dropout_rate=0.1
)

dummy_input = tf.ones(
    (1, 128),
    dtype=tf.int32
)

model(dummy_input)

model.load_weights(MODEL_PATH)


# --------------------------------------------------
# Generation
# --------------------------------------------------

def generate_text_stream(
    prompt,
    max_new_tokens=120,
    temperature=0.9,
    top_k=40
):
    input_ids = sp.encode(
        prompt,
        out_type=int
    )

    previous_text = sp.decode(input_ids)

    yield previous_text

    for _ in range(max_new_tokens):

        context_ids = input_ids[-128:]

        x = tf.constant(
            [context_ids],
            dtype=tf.int32
        )

        logits, _ = model(
            x,
            training=False
        )

        logits = logits[:, -1, :]
        logits = logits / temperature

        values, indices = tf.math.top_k(
            logits,
            k=top_k
        )

        sampled = tf.random.categorical(
            values,
            num_samples=1
        )[0, 0]

        predicted_id = int(
            indices[0, sampled].numpy()
        )

        if predicted_id == eos_id:
            break

        input_ids.append(predicted_id)

        current_text = sp.decode(input_ids)

        new_text = current_text[len(previous_text):]

        if new_text:
            yield new_text

        previous_text = current_text
# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class StoryRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=30
    )


# --------------------------------------------------
# API Endpoint
# --------------------------------------------------

@app.post("/generate")
def generate_story(request: StoryRequest):

    return StreamingResponse(
        generate_text_stream(request.prompt),
        media_type="text/plain"
    )