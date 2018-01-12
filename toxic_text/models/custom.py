from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dense, GRU, \
    TimeDistributed, CuDNNGRU, CuDNNLSTM
from keras import backend as K
from keras.engine.topology import Layer

from keras.layers import Conv1D, Concatenate, Flatten, Dropout, \
    Reshape

from keras import regularizers

from toxic_text.models.cnn import dpcnn_convolution_block
