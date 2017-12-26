
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Dense, GRU, \
    TimeDistributed, Flatten, Lambda, CuDNNGRU
from keras import backend as K
from keras.engine.topology import Layer

def simple_birnn(model_input, units=50, dropout=0.1):
    bi_rnn = Bidirectional(LSTM(units, return_sequences=True))(model_input)
    max_pool = GlobalMaxPool1D()(bi_rnn)
    dropout_one = Dropout(dropout)(max_pool)
    dense = Dense(units, activation="relu")(dropout_one)
    dropout_dense = Dropout(dropout)(dense)

    return dropout_dense

class Attention(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        self.mlp_weight_matrix = None
        self.mlp_bias = None
        self.context_vector = None

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.mlp_weight_matrix = self.add_weight(name='mlp_weights',
                                      shape=(input_shape[-1], 25),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.mlp_bias = self.add_weight(name='mlp_bias', shape=(25, ),
                                        initializer='zeros',
                                        trainable=True)
        self.context_vector = self.add_weight(name='context_vector', shape=(25, 1),
                                              initializer='glorot_uniform',
                                              trainable=True)
        self.repeat_n = input_shape[len(input_shape) - 1]
        self.shape = input_shape

        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        h = K.tanh(K.dot(x, self.mlp_weight_matrix) + self.mlp_bias)
        h_context = K.dot(h, self.context_vector)
        alpha = K.repeat_elements(K.softmax(h_context), self.repeat_n, -1)
        return K.sum(alpha * x, axis=len(self.shape) - 2)


    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])

def hierarchical_attention_network(model_input, units=50):
    """
    http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
    :return:
    """
    word_encoder = TimeDistributed(Bidirectional(GRU(units, return_sequences=True)))(model_input)
    word_attention = TimeDistributed(Attention())(word_encoder)
    sentence_encoder = Bidirectional(GRU(units, return_sequences=True))(word_attention)
    sentence_attention = Attention()(sentence_encoder)

    return sentence_attention