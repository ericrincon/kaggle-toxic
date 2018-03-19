import numpy as np

from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dense, GRU, \
    TimeDistributed, CuDNNGRU, CuDNNLSTM, GlobalAveragePooling1D, PReLU, \
    Embedding
from keras import backend as K
from keras.engine.topology import Layer

from keras.layers import Conv1D, Concatenate, Flatten, Dropout, \
    Reshape

from toxic_text.models.keras.cnn import dpcnn_convolution_block
from keras.optimizers import Adam
from keras import regularizers

from keras.models import Input, Model

"""
Dict mapping RNN names to the corresponding Keras layer
"""
_RNNS = {
    'CGRU': CuDNNGRU,
    'CLSTM': CuDNNLSTM,
    'LSTM': LSTM,
    'GRU': GRU
}


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


def simple_birnn(model_input, pos_tag_embeddings=None, units=50, dropout=0.5, embedding_dropout=0.1,
                 prelu=True, rnn_cell_name='CLSTM'):
    """

    :param model_input: embedding tensor of the input sequence
    :param pos_tag_embeddings:
    :param units:
    :param dropout:
    :param embedding_dropout:
    :param prelu:
    :param rnn_cell_name:
    :return:
    """

    if pos_tag_embeddings:
        model_input = Concatenate()([model_input, pos_tag_embeddings])

    if embedding_dropout:
        model_input = Dropout(embedding_dropout)(model_input)

    rnn_cell = get_rnn_name(rnn_cell_name)

    bi_rnn = Bidirectional(rnn_cell(units, return_sequences=True))(model_input)
    attention = Attention()(bi_rnn)

    max_pool = GlobalMaxPool1D()(bi_rnn)
    average_pool = GlobalAveragePooling1D()(bi_rnn)
    concat = Concatenate()([max_pool, average_pool, attention])
    dropout_one = Dropout(dropout)(concat)

    activation = None if prelu else 'relu'

    dense = Dense(units, activation=activation)(dropout_one)

    if prelu:
        dense = PReLU()(dense)

    dropout_dense = Dropout(dropout)(dense)

    return dropout_dense


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
            filter_sizes = [2, 3]

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



class UnifiedAbuseRNN:
    """

    A Unified Deep Learning Architecture for Abuse Detection
    :return:
    """

    def __init__(self, input_length, vocab_size, name='MultiChannelRNN', l2=0.0, rnn_units=50,
                 embedding_dropout=None, rnn_type='gru', attention=None, concat_dropout=None,
                 prelu=False, final_units=50, final_dropout=None, nb_metadata_features=10,
                 embedding_dim=100, embedding_trainable=True, embedding_weights=None):

        if rnn_type == 'gru':
            self.RNN = CuDNNGRU
        elif rnn_type == 'lstm':
            self.RNN = CuDNNLSTM
        else:
            raise ValueError("No RNN cell named {}!".format(rnn_type))

        self.input_length = input_length
        self.name = name
        self.l2 = l2
        self.rnn_units = rnn_units
        self.embedding_dropout = embedding_dropout
        self.attention = attention
        self.concat_dropout = concat_dropout
        self.prelu = prelu
        self.final_units = final_units
        self.final_dropout = final_dropout

        self.nb_metadata_features = nb_metadata_features

        # Embedding layer values
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_trainable = embedding_trainable
        self.embedding_weights = embedding_weights

        self.compiled = False
        self.model_layers = {}



        self.model_a, self.model_b = self._build_model()


    def _build_dense_layer(self, layer_input, dropout):
        """

        :param layer_input:
        :param dropout:
        :return:
        """
        activation = None if self.prelu else 'relu'

        dense = Dense(self.final_units, activation=activation)(layer_input)

        if self.prelu:
            dense = PReLU()(dense)

        if dropout:
            dense = Dropout(dropout)(dense)

        return dense


    def _add_layer(self, layer, model):
        """

        :param layer:
        :param model:
        :return:
        """

        model_layers = self._get_model_layers(model)
        model_layers.append(layer)

        return layer

    def _build_word_rnn(self, model_name):
        """

        :return:
        """

        model_name += '_rnn'
        self.model_layers[model_name] = []

        model_input = Input(shape=(self.input_length, ), dtype='int32', name='word_input')

        embedding_kwargs = {
            "input_dim": self.vocab_size + 1,
            "output_dim": self.embedding_dim,
            "input_length": self.input_length,
            "trainable": self.embedding_trainable
        }

        if self.embedding_weights is not None:
            embedding_kwargs['weights'] = [self.embedding_weights]

        embedding = self._add_layer(Embedding(**embedding_kwargs)(model_input),
                                    model_name)

        RNN = self.RNN

        if self.embedding_dropout:
            embedding = self._add_layer(Dropout(self.embedding_dropout)(embedding), model_name)

        bi_rnn = self._add_layer(Bidirectional(RNN(self.rnn_units, return_sequences=True))(embedding),
                                 model_name)

        max_pool = self._add_layer(GlobalMaxPool1D()(bi_rnn), model_name)
        average_pool = self._add_layer(GlobalAveragePooling1D()(bi_rnn), model_name)

        features = [max_pool, average_pool]

        if self.attention:
            attention = self._add_layer(Attention()(bi_rnn), model_name)
            features.append(attention)

        concat = self._add_layer(Concatenate()(features), model_name)

        if self.concat_dropout:
            concat = self._add_layer(Dropout(self.concat_dropout)(concat), model_name)

        dense = self._add_layer(self._build_dense_layer(concat, self.final_dropout),
                                model_name)

        return dense, model_input

    def _get_model_layers(self, model_branch_name):
        if model_branch_name not in self.model_layers:
            assert ValueError('Model branch {} not defined!')

        return self.model_layers[model_branch_name]


    def _build_metadata_network(self, model_name):
        """

        :return:
        """
        model_name += '_dnn'
        self.model_layers[model_name] = []

        model_input = Input(shape=(self.nb_metadata_features, ), name='metadata_input')

        dense = self._add_layer(Dense(50)(model_input), model_name)

        return dense, model_input

    def _set_trainable(self, model_name, trainable):
        """

        :param model_name:
        :param trainable:
        :return:
        """

        for layer in self._get_model_layers(model_name):
            layer.trainable = trainable


    def _build_single_model(self, model_name, word_rnn_trainable, metadata_network_trainable):
        """
        Builds a single model.
        This method exists since in the interleaved training
        there has to be two models one with a network with frozen weights
        and the other input with trainable weights

        :return: a model with a branch that has frozen weights
        and the other branch with trainable weights
        """
        word_rnn, word_rnn_input = self._build_word_rnn(model_name)
        metadata_network, metadata_network_input = self._build_metadata_network(model_name)

        # # set each auxilary input to trainable
        self._set_trainable(model_name + '_rnn', word_rnn_trainable)
        self._set_trainable(model_name + '_dnn', metadata_network_trainable)

        concat = Concatenate()([word_rnn, metadata_network])

        output = Dense(6, activation='sigmoid', name='output',
                       kernel_regularizer=regularizers.l2(self.l2))(concat)

        model = Model(inputs=[word_rnn_input, metadata_network_input],
                      outputs=output)

        return model

    def _build_model(self):
        model_a = self._build_single_model('a', word_rnn_trainable=True,
                                           metadata_network_trainable=False)

        model_b = self._build_single_model('b', word_rnn_trainable=False,
                                           metadata_network_trainable=True)

        return model_a, model_b

    def _compile_model(self, learning_rate):
        if not self.compiled:
            adam = Adam(lr=learning_rate)

            # All of the models right now have 6 sigmoid outputs so binary cross entropy
            self.model_a.compile(optimizer=adam, loss='binary_crossentropy',
                               metrics=['accuracy'])
            self.model_b.compile(optimizer=adam, loss='binary_crossentropy',
                                 metrics=['accuracy'])

    def train(self, x, y, nb_epochs, batch_size, learning_rate=0.001):
        """
        The default training method for this network is interleaved training

        :param learning_rate:
        :return:
        """
        self._compile_model(learning_rate)

        def generator():
            idxs = np.arange(x[0].shape[0])
            np.random.shuffle(idxs)

            for i in range(0, x[0].shape[0], batch_size):
                idx_batch = idxs[i: i + batch_size]
                x_batch = [_x[idx_batch] for _x in x]
                print([l.shape for l in x_batch])
                y_batch = y[idx_batch]

                yield x_batch, y_batch

        for epoch in range(nb_epochs):
            for mini_batch_i, mini_batch in enumerate(generator()):
                if (mini_batch_i + epoch) % 2 == 0:
                    model = self.model_a
                else:
                    model = self.model_b

                history = model.fit(x=x, y=y, batch_size=batch_size, epochs=1,
                                    verbose=0)
                #
                # if (mini_batch_i + epoch) % 2 == 0:
                #     self.model_b.set_weights(model.get_weights())
                # else:
                #     self.model_a.set_weights(model.get_weights())

                if mini_batch_i == 2:
                    break
            break