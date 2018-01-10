"""
Feed forward style architectures are defined here
"""

from keras.layers import Dense, Flatten, GlobalAveragePooling1D


def fast_text(model_input):
    """
    Fast text style network without all the fancy c optimizations but
    written in keras :)

    :return:
    """
    hidden_layer = GlobalAveragePooling1D()(model_input)

    return hidden_layer



def logistic(model_input):
    return model_input

def d2v(model_input, hidden_layer_units=50):
    """
    Simple single hidden layer neural network for doc2vec representation
    :param model_input:
    :param hidden_layer_units:
    :return:
    """

    hidden_layer = Dense(hidden_layer_units)(model_input)

    return hidden_layer