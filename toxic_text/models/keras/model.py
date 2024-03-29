"""

"""
import numpy as np

from keras.models import Input, Model
from keras.layers import Dense, Embedding, TimeDistributed, Flatten, Concatenate, \
    GlobalAveragePooling1D
from keras.optimizers import Adam
from keras import regularizers

from toxic_text.models.keras.cnn import sentence, deep_pyramid
from toxic_text.models.keras.rnn import simple_birnn, hierarchical_attention_network, clstm
from toxic_text.models.keras.nn import fast_text, logistic, d2v

_MODELS = {
    "sentence": sentence,
    'dpcnn': deep_pyramid,
    'birnn': simple_birnn,
    'han': hierarchical_attention_network,
    'clstm': clstm,
    'fasttext': fast_text,
    'logistic': logistic,
    'd2v': d2v
}


def get_all_models():
    return list(_MODELS.keys())


def build_topic_embedding_layer(nb_topics, embedding_dim, max_topics, average=False):
    """
    Easy helper function for building the embedding layer for topic features
    :param nb_topics:
    :param embedding_dim:
    :param max_topics:
    :return:
    """

    lda_input = Input(shape=(max_topics,), dtype='int32', name='topic_embeddings')
    lda_embeddings = Embedding(input_dim=nb_topics, output_dim=embedding_dim,
                               input_length=max_topics)(lda_input)
    return GlobalAveragePooling1D()(lda_embeddings) if average \
               else Flatten()(lda_embeddings), lda_input


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


def build_pos_embedding_layer(max_pos_tags, nb_pos_tags, embedding_dim):
    """
    Embedding matrix for part of speech tags. The input will be a sequence
    of integeres mapping pos tags to vectors

    :param max_pos_tags: note this should be the same as the maximum number of
    words
    :param nb_pos_tags: the total unique number of pos tags
    :param embedding_dim: the size of the embedding vector for the pos tags
    :return: the embedding vectors as a tensor resulting from the input
    """

    pos_tags = Input(shape=(max_pos_tags,), dtype='int32', name='pos_embeddings')
    pos_embeddings = Embedding(input_dim=nb_pos_tags, output_dim=embedding_dim,
                               input_length=max_pos_tags)(pos_tags)

    return pos_embeddings


def build_model_input(model_input, vocab_size, embedding_dim, input_length,
                      embedding_matrix=None, time_distributed=True, trainable=True):
    """
    This function build the input layers for all models
    :return:
    """
    kwargs = {
        "input_dim": vocab_size + 1,
        "output_dim": embedding_dim,
        "input_length": input_length,
        "trainable": trainable
    }

    if embedding_matrix is not None:
        kwargs['weights'] = [embedding_matrix]

    embedding = Embedding(**kwargs)

    return TimeDistributed(embedding)(model_input) if time_distributed else embedding(model_input)


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


def _build_model(model_input, model_outputs):
    """
    Sets the optimizer, inputs and outputs of the model, and returns the compiled model
    :param model_input:
    :return:
    """

    adam = Adam(lr=.001)
    model = Model(inputs=model_input, outputs=model_outputs)

    # All of the models right now have 6 sigmoid outputs so binary cross entropy
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_single_head_model_output(model, name, l2=0.0001):
    """
    Creates the output for models that is 6 sigmoid neurons
    :param model: a keras model )
    :param name: name of the model output
    :return:
    """
    return Dense(6, activation='sigmoid', name=name,
                 kernel_regularizer=regularizers.l2(l2))(model)


def build_single_head_model(model, vocab_size, vector_dim, input_length, name,
                            embedding_matrix=None, lda=None):
    model_input = Input(shape=(input_length,), dtype='int32', name='input')

    model_inputs = build_model_input(model_input, vocab_size, vector_dim,
                                     input_length, embedding_matrix=embedding_matrix,
                                     time_distributed=False)

    model = model(model_inputs)

    if lda:
        lda_features, lda_input = build_topic_embedding_layer(**lda)
        model = Concatenate()([model, lda_features])
        model_input = [model_input, lda_input]

    model_outputs = build_single_head_model_output(model, name)

    return _build_model(model_input, model_outputs)


def build_time_dist_model(model, vocab_size, vector_dim, max_words=15, max_sentences=8,
                          name='han', embedding_matrix=None):
    model_input = Input(shape=(max_sentences, max_words), dtype='int32', name='input')

    model_inputs = build_model_input(model_input, vocab_size, vector_dim,
                                     max_words, embedding_matrix=embedding_matrix)

    model = model(model_inputs)
    model_outputs = build_single_head_model_output(model, name)

    return _build_model(model_input, model_outputs)
