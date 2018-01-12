import numpy as np
import pandas as pd
import spacy
import h5py

from tqdm import tqdm

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import StratifiedShuffleSplit
from toxic_text.test.evaluate import TARGET_NAMES

from keras.preprocessing.text import Tokenizer

from nltk import ngrams
from keras.preprocessing.text import text_to_word_sequence

# Load spacy once
nlp = spacy.load('en')


def load_train_hdf5(path):
    data = h5py.File(path, 'r')

    return data['x'][:], data['y'][:]


def load_test_hdf5(path):
    data = h5py.File(path, 'r')

    return data['x'][:]

def get_training_data(path):
    train_data = pd.read_csv(path)

    texts = train_data['comment_text'].astype(str)
    labels = get_labels(train_data)

    return texts, labels


def get_train_valid_split(x, y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.1, random_state=0)

    train, test = [(train, test) for train, test in sss.split(X=x, y=y)][0]

    return x[train], x[test], [_y[train] for _y in y], [_y[test] for _y in y]

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

    return preds_df


def setup_fit_tokenizer(texts, max_words=25000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)

    return tokenizer

def load_setup_fit_tokenizer(texts, seq_length, max_words=25000):
    tokenizer = setup_fit_tokenizer(texts, max_words=max_words)

    examples = tokenizer.texts_to_sequences(texts)
    examples = pad_sequences(examples, seq_length)

    return tokenizer, examples


def load_setup_fit_sent_tokenizer(texts, max_words, max_sentences=5):
    tokenizer = setup_fit_tokenizer(texts)

    examples = np.zeros((texts.shape[0], max_sentences, max_words), dtype=np.int32)

    for (example_i, comment) in tqdm(enumerate(texts[:100])):
        tokens = nlp(comment)

        for sentence_i, sentence in enumerate(tokens.sents):
            if sentence_i == max_sentences:
                break

            tokenized_sentence = tokenizer.texts_to_sequences([sentence.text])
            tokenized_sentence = pad_sequences(tokenized_sentence, max_words)
            examples[example_i, sentence_i, :] = tokenized_sentence[0]

    return tokenizer, examples

def preds_to_df(prob_predictions):
    """

    :param prob_predictions: a list of predictiosn from a keras model
    each element in the list should be a column vector
    :return:
    """
    preds_as_list_of_series = list(map(lambda x: pd.Series(x.flatten()), prob_predictions))

    return pd.concat(preds_as_list_of_series, axis=1)


def get_labels(df):
    labels = [df.toxic, df.severe_toxic, df.obscene,
              df.threat, df.insult, df.identity_hate]
    labels = list(map(lambda y: y.values, labels))
    labels = np.array(labels, dtype=np.int32).transpose()

    return labels


def get_ngram_tokens(documents, ngram):
    document_ngrams = []

    for document in documents:
        ngram_tuples = list(ngrams(text_to_word_sequence(document), ngram))
        ngram_tokens = ' '.join(list(map(lambda ngram: '_'.join(ngram), ngram_tuples)))
        document_ngrams.append(ngram_tokens)

    return document_ngrams



