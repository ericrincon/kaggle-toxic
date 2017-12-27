from keras.layers import Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout, \
    Activation, Add, BatchNormalization


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


def deep_pyramid(model_input, nb_filters=300, filter_size=3, dropout=0.5):
    """
    http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf


    :param model_input:
    :param nb_filters:
    :param filter_size: this parameter must be the same size as the embedding dimension
    this is to prevent dimension matching at the shortcut connection
    :return:
    """
    def convolution_block(conv_input, pool=True):
        if pool:
            conv_input = MaxPooling1D(pool_size=filter_size, strides=2)(conv_input)
        shortcut = conv_input

        pre_activation_1 = Activation('relu')(conv_input)
        conv_1 = Conv1D(nb_filters, filter_size,
                        activation=None, padding='same')(pre_activation_1)
        pre_activation_2 = Activation('relu')(conv_1)
        conv_2 = Conv1D(nb_filters, filter_size,
                        activation=None, padding='same')(pre_activation_2)
        # conv_2 = Flatten()(conv_2)
        shortcut_connection = Add()([conv_2, shortcut])

        return shortcut_connection

    conv_block_1 = convolution_block(model_input, pool=False)
    norm_one = BatchNormalization()(conv_block_1)

    conv_block_2 = convolution_block(norm_one, pool=True)
    norm_one = BatchNormalization()(conv_block_1)

    conv_block_3 = convolution_block(conv_block_2, pool=True)
    norm_one= BatchNormalization()(conv_block_1)

    conv_block_4 = convolution_block(conv_block_3, pool=True)
    norm_one= BatchNormalization()(conv_block_1)

    pooling = MaxPooling1D()(conv_block_4)

    flattened_features = Flatten()(pooling)

    dropout_layer = Dropout(dropout)(flattened_features)

    return dropout_layer


