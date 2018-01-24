from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dense, GRU, \
    TimeDistributed, CuDNNGRU, CuDNNLSTM, GlobalAveragePooling1D, PReLU
from keras import backend as K
from keras.engine.topology import Layer

from keras.layers import Conv1D, Concatenate, Flatten, Dropout, \
    Reshape

from keras import regularizers

from toxic_text.models.cnn import dpcnn_convolution_block

"""
Dict mapping RNN names to the corresponding Keras layer
"""
_RNNS = {
    'CGRU': CuDNNGRU,
    'CLSTM': CuDNNLSTM,
    'LSTM': LSTM,
    'GRU': GRU
}


def get_rnn_name(rnn_name):
    """
    Simple function for getting different RNN variants

    :param rnn_name: the name of the RNN layer
    :return: the corresponding Keras RNN layer
    """
    if rnn_name.upper() in _RNNS:
        return _RNNS[rnn_name]
    else:
        raise ValueError('The RNN type {} is not defined!'.format(rnn_name))



def simple_birnn(model_input, units=50, dropout=0.5, embedding_dropout=0.1,
                 prelu=True):
    if embedding_dropout:
        model_input = Dropout(embedding_dropout)(model_input)

    bi_rnn = Bidirectional(CuDNNLSTM(units, return_sequences=True))(model_input)
    max_pool = GlobalMaxPool1D()(bi_rnn)
    average_pool = GlobalAveragePooling1D()(bi_rnn)
    concat = Concatenate()([max_pool, average_pool])
    dropout_one = Dropout(dropout)(concat)

    activation = None if prelu else 'relu'
    dense = Dense(units, activation=activation)(dropout_one)

    if prelu:
        dense = PReLU()(dense)

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


def clstm(model_input, rnn_units=50, nb_filters=50, embedding_dropout=0.1, output_dropout=0.5,
          dpcnn=False, filter_sizes=None, prelu=True):
    """
    https://arxiv.org/pdf/1511.08630.pdf

    A Bi-Directional LSTM/GRU with n-gram convolutional features as the input

    :param model_input:
    :param rnn_units: The number of units in the RNN cell note the output vector size of
    each unit will be twice this number since the RNN is wrapper in a Bi-Directional layer
    :param nb_filters: the number of filters in the convolution layers
    :param dpcnn: if to use a deep pyramid style convolution block see the function
    dpcnn_convolution_block or the deep_pyramid model for more details
    :param filter_sizes: the size of the n-gram window sizes for the convolution layers
    :return: the output tensor of the model
    """

    if embedding_dropout:
        model_input = Dropout(embedding_dropout)(model_input)

    if dpcnn:
        # the the number of filters for this convolution block must be the same as the
        # size of the input embeddings as to avoid a dimension matching transformation in
        # the shortcut connection.
        # The embedding shape must be converted into an int as it of type Dimension
        nb_filters = int(model_input.shape[-1])

        conv_features = dpcnn_convolution_block(model_input, nb_filters=nb_filters,
                                                filter_size=3, pool=False)
    else:
        # If no filter sizes are passed than only a filter size of 3 will be used
        if not filter_sizes:
            filter_sizes = [3, 4]

        features = []

        for filter_size in filter_sizes:
            feature_map = Conv1D(nb_filters, filter_size, activation='relu', padding='same',
                                 kernel_regularizer=regularizers.l2(0.0001))(model_input)
            flattened_features = Flatten()(feature_map)
            features.append(flattened_features)

        if len(features) > 1:
            conv_features = Concatenate(axis=-1)(features)
        else:
            conv_features = features[0]

        conv_features = Reshape((len(filter_sizes), int(model_input.shape[1]) * nb_filters))(conv_features)

    bi_rnn = Bidirectional(CuDNNLSTM(rnn_units, return_sequences=True))(conv_features)
    max_pool = GlobalMaxPool1D()(bi_rnn)
    average_pool = GlobalAveragePooling1D()(bi_rnn)
    concat = Concatenate()([max_pool, average_pool])

    activation = None if prelu else 'relu'

    output = Dense(50, activation=activation)(concat)

    if prelu:
        output = PReLU()(output)

    if output_dropout:
        output = Dropout(output_dropout)(output)

    return output



