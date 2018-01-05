"""
Feed forward style architectures are defined here
"""

from keras.layers import Dense, Flatten, GlobalAveragePooling1D


def fast_text(model_input, hidden_layer_size=100):
    """
    Fast text style network without all the fancy c optimizations but
    written in keras :)

    :return:
    """
    hidden_layer = GlobalAveragePooling1D()(model_input)

    return hidden_layer



def logistic(model_input):
    return model_input