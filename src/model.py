import tensorflow as tf
from .attention import MoChA


class Encoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = tf.keras.layers.RNN(
            [tf.keras.layers.GRUCell(hidden_dim) for _ in range(num_layers)],
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate,
        )
        self.input_projection = tf.keras.layers.Dense(hidden_dim, activation='tanh')

    def call(self, inputs):
        projected_inputs = self.input_projection(inputs)
        outputs, *hidden_states = self.rnn(projected_inputs)
        return outputs, hidden_states


class Decoder(tf.keras.Model):
    def __init__(self, output_dim, hidden_dim, num_layers, chunk_size, dropout_rate):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = tf.keras.layers.RNN(
            [tf.keras.layers.GRUCell(hidden_dim) for _ in range(num_layers)],
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate,
        )
        self.mocha_attention = MoChA(chunk_size)
        self.output_projection = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, encoder_outputs, initial_states=None):
        rnn_outputs, *hidden_states = self.rnn(inputs, initial_state=initial_states)
        attention_weights = self.mocha_attention(encoder_outputs, rnn_outputs)
        context_vectors = tf.matmul(attention_weights, encoder_outputs)
        combined_outputs = tf.concat([rnn_outputs, context_vectors], axis=-1)
        output_logits = self.output_projection(combined_outputs)
        return output_logits, hidden_states


class MoChAASR(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, chunk_size, dropout_rate):
        super(MoChAASR, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout_rate)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers, chunk_size, dropout_rate)

    def call(self, encoder_inputs, decoder_inputs):
        encoder_outputs, hidden_states = self.encoder(encoder_inputs)
        decoder_outputs, _ = self.decoder(decoder_inputs, encoder_outputs, initial_states=hidden_states)
        return decoder_outputs
