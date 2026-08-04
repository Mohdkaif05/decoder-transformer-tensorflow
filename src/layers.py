import tensorflow as tf

# Attention Mechanism

def scaled_dot_product_attention(q, k, v, mask=None):
   
    #QK^T
    matmul_qk = tf.linalg.matmul(q, k, transpose_b=True)

    #Scaling
    dk = tf.cast(tf.shape(k)[-1], q.dtype)
    scaled_logits = matmul_qk / tf.math.sqrt(dk)

    #Masking
    if mask is not None:
        mask = tf.cast(mask, q.dtype)
        scaled_logits += (mask * -1e4)

    # 4.Softmax Normalization
    attention_weights = tf.nn.softmax(scaled_logits, axis=-1)

    # 5. Multiply with V
    output = tf.linalg.matmul(attention_weights, v)

    return output, attention_weights


#Multihead Attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.depth = d_model // num_heads

        # Linear projections
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # Final output projection
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Input shape:  (batch, seq_len, d_model)
        Output shape: (batch, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]

        # 1. Linear projections
        q = self.wq(x)  # (B, S, d_model)
        k = self.wk(x)
        v = self.wv(x)

        # 2. Split into heads
        q = self.split_heads(q, batch_size)  # (B, H, S, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 3. Scaled dot-product attention
        scaled_attention,attention_weights =scaled_dot_product_attention(q,k,v,mask)

        # 4. Transpose back
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (B, S, H, depth)

        # 5. Concatenate heads
        concat_attention=tf.reshape(scaled_attention,(batch_size,-1,self.d_model))
        # (B, S, d_model)

        # 6. Final linear layer
        output = self.dense(concat_attention)

        return output, attention_weights
    

# positional Encoding

def positional_encoding(seq_len, d_model):

    position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis] # Position indices (S, 1)

    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]  # Dimension indices (1, d_model)

    angle_rates = 1 / tf.pow(10000.0,(2 * (tf.floor(i / 2))) / tf.cast(d_model, tf.float32))

    angle_rads = position * angle_rates  # (S, d_model)

    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])

    # Interleave sin and cos 
    pos_encoding = tf.reshape(tf.stack([sines, cosines], axis=-1),(seq_len, d_model))

    # Add batch dimension
    pos_encoding = pos_encoding[tf.newaxis, ...]

    return pos_encoding

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)
    
#Feed Forward Neural Network
class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(dff, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.dense1(x)   # (B, S, hidden layer neuron)
        x = self.dense2(x)   # (B, S, d_model)
        return x
    
  