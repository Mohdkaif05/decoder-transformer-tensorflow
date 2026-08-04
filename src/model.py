import tensorflow as tf

from src.layers import (
    MultiHeadAttention,
    FeedForwardNetwork,
    PositionalEncoding,
)

from src.utils import create_look_ahead_mask

#Decoder Block
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        # Multi-head self-attention
        self.mha = MultiHeadAttention(d_model, num_heads)

        # Feed Forward Network
        self.ffn = FeedForwardNetwork(d_model, dff)

        # Layer Normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None, mask=None):
        
        #Masked Multi-Head Attention
        attn_output, attn_weights = self.mha(x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        # Add + LayerNorm
        out1 = self.layernorm1(x + attn_output)

       
        #Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Add + LayerNorm
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn_weights

# final complete architecture
class GPT(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len,
                 num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model

        # Token Embedding
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)

        # Decoder Blocks
        self.decoder_blocks = []
        for _ in range(num_layers):
            self.decoder_blocks.append(
                DecoderBlock(d_model, num_heads, dff, dropout_rate)
            )

        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Final Output Layer (logits)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training=None):
        """
        Args:
            x: (B, S) → token IDs
        Returns:
            logits: (B, S, vocab_size)
        """

        seq_len = tf.shape(x)[1]

       
        #Create Look-Ahead Mask (FIXED SHAPE)
       
        mask = create_look_ahead_mask(seq_len)           # (S, S)
        mask = mask[tf.newaxis, tf.newaxis, :, :]        # (1, 1, S, S)

        #Embedding
        
        x = self.embedding(x)                            # (B, S, d_model)

        # Scale embeddings (important)
        x *= tf.math.sqrt(tf.cast(self.d_model, x.dtype))

        # 3. Positional Encoding
        x = self.pos_encoding(x)

        x = self.dropout(x, training=training)

        # 4. Decoder Blocks
        attention_weights = {}

        for i, block in enumerate(self.decoder_blocks):
            x, attn = block(x, training=training, mask=mask)
            attention_weights[f"block_{i+1}"] = attn

        # 5. Final Linear Layer
        logits = self.final_layer(x)                     # (B, S, vocab_size)

        return logits, attention_weights