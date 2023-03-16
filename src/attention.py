import tensorflow as tf


class MoChAAttention(tf.keras.layers.Layer):
    def __init__(self, chunk_size, **kwargs):
        super(MoChAAttention, self).__init__(**kwargs)
        self.chunk_size = chunk_size

    def build(self, input_shape):
        super(MoChAAttention, self).build(input_shape)

    def call(self, emit_probs, softmax_logits):
        return self.stable_chunkwise_attention(self.chunk_size, emit_probs, softmax_logits)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

    def moving_sum(self, x, back, forward):
        x_padded = tf.pad(x, [[0, 0], [back, forward]])
        x_padded = tf.expand_dims(x_padded, -1)
        filters = tf.ones((back + forward + 1, 1, 1))
        x_sum = tf.nn.conv1d(x_padded, filters, 1, padding='VALID')
        return x_sum[..., 0]

    def stable_chunkwise_attention(self, chunk_size, emit_probs, softmax_logits):
        logits_max = self.moving_max(softmax_logits, chunk_size)
        padded_logits = tf.pad(softmax_logits, [[0, 0], [chunk_size - 1, 0]], constant_values=-tf.float32.max)
        framed_logits = tf.signal.frame(padded_logits, chunk_size, 1)
        framed_logits = framed_logits - tf.expand_dims(logits_max, -1)
        softmax_denominators = tf.reduce_sum(tf.exp(framed_logits), 2)
        framed_denominators = tf.signal.frame(softmax_denominators, chunk_size, 1, pad_end=True,
                                              pad_value=tf.float32.max)
        batch_size, seq_len = tf.unstack(tf.shape(softmax_logits))
        copied_shape = (batch_size, seq_len, chunk_size)
        copied_logits = (tf.expand_dims(softmax_logits, -1) * tf.ones(copied_shape, softmax_logits.dtype))
        framed_max = tf.signal.frame(logits_max, chunk_size, 1, pad_end=True, pad_value=tf.float32.max)
        copied_logits = copied_logits - framed_max
        softmax_numerators = tf.exp(copied_logits)
        framed_probs = tf.signal.frame(emit_probs, chunk_size, 1, pad_end=True)
        return tf.reduce_sum(framed_probs * softmax_numerators / framed_denominators, 2)

    def moving_max(self, x, w):
        x = tf.pad(x, [[0, 0], [w - 1, 0]], mode='CONSTANT', constant_values=-tf.float32.max)
        x = tf.reshape(x, [tf.shape(x)[0], 1, tf.shape(x)[1], 1])
        x = tf.nn.max_pool2d(x, ksize=[1, 1, w, 1], strides=[1, 1, 1, 1], padding='VALID')
        return x[:, 0, :, 0]
