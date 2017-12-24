import pickle as p

import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import StratifiedShuffleSplit

from evaluate import get_targets

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


def create_submission(model, test_data, tokenizer, seq_length):
    test_texts = test_data['comment_text'].astype(str)

    test_examples = tokenizer.texts_to_sequences(test_texts)
    test_examples = pad_sequences(test_examples, seq_length)

    preds = model.predict(test_examples)

    class_predictions = get_targets(preds)
    p.dump(class_predictions, open('class_preds.p', 'wb'))


    series = list(map(lambda x: pd.Series(x.flatten()), preds))
    preds_df = [test_data['id']]
    preds_df.extend(series)

    preds_df = pd.concat(preds_df, axis=1)
    preds_df.columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat',
                        'insult', 'identity_hate']
    preds_df.to_csv('submission.csv', index=False)


def setup_fit_tokenizer(texts, seq_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    examples = tokenizer.texts_to_sequences(texts)
    examples = pad_sequences(examples, seq_length)

    return tokenizer, examples
