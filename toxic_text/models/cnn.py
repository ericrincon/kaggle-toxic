from keras.layers import Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout, \
    Activation, Add, BatchNormalization

from keras import regularizers


def sentence(model_input, nb_filters=100, filter_sizes=None, dropout=0.5):
    """

    :return:
    """

    # set up mutable data types
    if not filter_sizes:
        filter_sizes = [2, 3, 4]

    features = []

    for filter_size in filter_sizes:
        feature_map = Conv1D(nb_filters, filter_size, activation='relu')(model_input)
        pooling = MaxPooling1D(pool_size=filter_size)(feature_map)
        flattened_features = Flatten()(pooling)
        features.append(flattened_features)

    concatenated_features = Concatenate()(features)


    return Dropout(dropout)(concatenated_features)


def dpcnn_convolution_block(conv_input, filter_size, nb_filters, pool=True,
                            l2_weight=0.0001):
    """
    A convolution block for the CNN type Deep Pyramid CNNs
    as defined in the paper.


    :param conv_input: the input into the block
    :param filter_size:
    :param nb_filters:
    :param pool:
    :return:
    """

    if pool:
        conv_input = MaxPooling1D(pool_size=filter_size, strides=2)(conv_input)
    shortcut = conv_input

    pre_activation_1 = Activation('relu')(conv_input)
    conv_1 = Conv1D(nb_filters, filter_size, kernel_regularizer=regularizers.l2(l2_weight),
                    activation=None, padding='same')(pre_activation_1)
    pre_activation_2 = Activation('relu')(conv_1)
    conv_2 = Conv1D(nb_filters, filter_size, kernel_regularizer=regularizers.l2(l2_weight),
                    activation=None, padding='same')(pre_activation_2)

    shortcut_connection = Add()([conv_2, shortcut])

    return shortcut_connection


def deep_pyramid(model_input, nb_convoultion_blocks=5,
                 nb_filters=128, filter_size=3, dropout=0.5):
    """
    http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf


    :param model_input:
    :param nb_filters:
    :param filter_size: this parameter must be the same size as the embedding dimension
    this is to prevent dimension matching at the shortcut connection
    :return:
    """

    # In the first convolution block max pooling is not applied
    conv_block = dpcnn_convolution_block(model_input, filter_size,
                                           nb_filters,  pool=False)

    for i in range(nb_convoultion_blocks):
        conv_block = dpcnn_convolution_block(conv_block, filter_size,
                                               nb_filters, pool=True)

    pooling = MaxPooling1D()(conv_block)

    flattened_features = Flatten()(pooling)

    dropout_layer = Dropout(dropout)(flattened_features)

    return dropout_layer


