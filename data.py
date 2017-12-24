import pickle as p

import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import StratifiedShuffleSplit
from train import TARGET_NAMES

from keras.preprocessing.text import Tokenizer


def get_training_data(path):
    train_data = pd.read_csv(path)

    texts = train_data['comment_text']
    labels = [train_data.toxic, train_data.severe_toxic, train_data.obscene,
              train_data.threat, train_data.insult, train_data.identity_hate]
    labels = list(map(lambda y: y.values, labels))

    return texts, labels

def get_train_valid_split(x, y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.1, random_state=0)
    train_index = []
    valid_index = []

    for _y in y:
        train_split, valid_split = [(train, test) for train, test in sss.split(X=x,
                                                                           y=_y)][0]
        train_index.extend(train_split.tolist())
        valid_index.extend(valid_split.tolist())

    # train_index = np.concatenate(train_index)
    # valid_index = np.concatenate(valid_index)
    train_index = list(set(train_index))
    valid_index = list(set(valid_index))

    return x[train_index], x[valid_index], \
           [_y[train_index] for _y in y], [_y[valid_index] for _y in y]


def create_submission(prob_predictions_df, test_data):
    """
    Simple function that creates a pandas dataframe from predictions and
    creates a csv file for submission to kaggle

    :param prob_predictions_df:
    :param test_data:
    :return:
    """

    # p.dump(class_predictions, open('class_preds.p', 'wb'))
    ids = test_data['id'].to_frame()

    preds_df = pd.concat([ids, prob_predictions_df], axis=1)
    columns = ['id']
    columns.extend(TARGET_NAMES)
    preds_df.columns = columns

    preds_df.to_csv('submission.csv', index=False)


def setup_fit_tokenizer(texts, seq_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    examples = tokenizer.texts_to_sequences(texts)
    examples = pad_sequences(examples, seq_length)

    return tokenizer, examples


def preds_to_df(prob_predictions):
    """

    :param prob_predictions: a list of predictiosn from a keras model
    each element in the list should be a column vector
    :return:
    """
    preds_as_list_of_series = list(map(lambda x: pd.Series(x.flatten()), prob_predictions))

    return pd.concat(preds_as_list_of_series, axis=1)