import tensorflow as tf

#Masking
def create_look_ahead_mask(seq_len):
    #Shape: (S, S)
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
    return mask  