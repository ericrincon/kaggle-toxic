"""

"""
import numpy as np

from keras.models import Input, Model
from keras.layers import Dense, Embedding, Flatten

from models.cnn import sentence, deep_pyramid


_MODELS = {
    "sentence": sentence,
    'dpcnn': deep_pyramid
}


def build_embedding_matrix(tokenizer, word2vec):
    """
    Builds an embedding matrix of shape (vocab size) x (trained word2vec embedding dim)

    The embedding matrix is used to initialize an Embedding layer

    :param tokenizer: a keras tokenizer object that has already been fit to the
    corpus
    :param word2vec: A trained gensim word2vec model
    :return: embedding matrix as a numpy matrix
    """

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, word2vec.vector_size))

    for word, index in tokenizer.word_index.items():
        if word in word2vec.wv:
            embedding_matrix[index, :] = word2vec.wv[word]
        else:
            embedding_matrix[index, :] = np.random.uniform(low=-.05, high=.05, size=word2vec.vector_size)

    return embedding_matrix


def build_model_input(model_input, vocab_size, embedding_dim, input_length,
                      embedding_matrix=None):
    """
    This function build the input layers for all models
    :return:
    """
    kwargs = {
        "input_dim": vocab_size + 1,
        "output_dim": embedding_dim,
        "input_length": input_length
    }

    if embedding_matrix is not None:
        kwargs['weights'] = [embedding_matrix]

    return Embedding(**kwargs)(model_input)


def build_multi_head_model_output(model):
    """
    This function builds the output layer which has 6 output nodes
    where each node is a binary classifier for one of the 6 types of
    toxic comments
    :return:
    """
    names = ['toxic', 'severe_toxic', 'obscene', 'threat',
             'insult', 'identity_hate']

    return list(map(lambda name: Dense(1, activation='sigmoid', name=name)(model), names))


def get_model(model_name):
    """
    Returns the corresponding model layers

    :param model_name:
    :return:
    """
    if model_name not in _MODELS:
        raise ValueError('The model {} is not a valid model type!'.format(model_name))

    return _MODELS[model_name]


def _build_model(model_input,  model_outputs):
    """

    :param model:
    :return:
    """

    model = Model(inputs=model_input, outputs=model_outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_multi_head_model(model, vocab_size, vector_dim, input_length, embedding_matrix=None):
    """

    :param model:
    :return:
    """

    model_input = Input(shape=(input_length,), dtype='int32', name='input')

    model_inputs = build_model_input(model_input, vocab_size, vector_dim,
                                     input_length, embedding_matrix=embedding_matrix)

    model = model(model_inputs)
    model_outputs = build_multi_head_model_output(model)

    return _build_model(model_input, model_outputs)


def build_single_head_model_output(model, name):
    return Dense(1, activation='sigmoid', name=name)(model)


def build_single_head_model(model, vocab_size, vector_dim, input_length, name,
                            embedding_matrix=None):
    model_input = Input(shape=(input_length,), dtype='int32', name='input')

    model_inputs = build_model_input(model_input, vocab_size, vector_dim,
                                     input_length, embedding_matrix=embedding_matrix)

    model = model(model_inputs)
    model_outputs = build_single_head_model_output(model, name)

    return _build_model(model_input, model_outputs)
