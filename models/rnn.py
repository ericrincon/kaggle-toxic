from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Dense, GRU, \
    TimeDistributed, Flatten, Lambda, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras import backend as K
from keras.engine.topology import Layer

from keras.layers import Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout, \
    Activation, Add, BatchNormalization, Reshape


def simple_birnn(model_input, units=50, dropout=0.1):
    bi_rnn = Bidirectional(LSTM(units, return_sequences=True))(model_input)
    attention = Attention()(bi_rnn)
    max_pool = GlobalMaxPool1D()(attention)
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
        self.mlp_bias = self.add_weight(name='mlp_bias', shape=(25,),
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


def hierarchical_attention_network(model_input, units=50, rnn_type='gru'):

    """
    http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
    :return:
    """

    if rnn_type == 'gru':
        rnn_type = CuDNNGRU
    elif rnn_type == 'lstm':
        rnn_type = CuDNNLSTM

    word_encoder = TimeDistributed(Bidirectional(rnn_type(units, return_sequences=True)))(model_input)
    word_attention = TimeDistributed(Attention())(word_encoder)
    sentence_encoder = Bidirectional(rnn_type(units, return_sequences=True))(word_attention)
    sentence_attention = Attention(name='sentence_attention')(sentence_encoder)

    return sentence_attention


def clstm(model_input, nb_filters=50, filter_sizes=None):
    """
    https://arxiv.org/pdf/1511.08630.pdf
    :param model_input:
    :return:
    """

    dropout_emb = Dropout(0.25)(model_input)

    if not filter_sizes:
        filter_sizes = [2, 3, 4]

    features = []

    for filter_size in filter_sizes:
        feature_map = Conv1D(nb_filters, filter_size, activation='relu',
                             padding='same')(dropout_emb)
        # pooling = MaxPooling1D(pool_size=filter_size)(feature_map)
        flattened_features = Flatten()(feature_map)
        features.append(flattened_features)


    concatenated_features = Concatenate(axis=-1)(features)
    reshaped = Reshape((len(filter_sizes), 100 * nb_filters))(concatenated_features)


    gru = CuDNNGRU(50, return_sequences=False)(reshaped)
    dropout = Dropout(.5)(gru)

    return dropout



