import argparse
import pandas as pd
import numpy as np
import h5py
import pickle as p

from tqdm import tqdm

from toxic_text.data.load import setup_fit_tokenizer, get_labels
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



def parse_sentences_words(documents, tokenizer, max_sentences=None):
    parsed_documents = []

    for (example_i, comment) in tqdm(enumerate(documents)):
        parsed_document = []

        sentences = sent_tokenize(comment)

        for sentence_i, sentence in enumerate(sentences):
            words = word_tokenize(sentence)

            if max_sentences and sentence_i == max_sentences:
                break
            filtered_sentence = ' '.join(map(lambda word: word,
                                             list(filter(lambda word: word not in stop_words, words))))
            tokenized_sentence = tokenizer.texts_to_sequences([filtered_sentence])[0]
            parsed_document.append(tokenized_sentence)
        parsed_documents.append(parsed_document)

    return parsed_documents


def pad_parsed_sequences(documents, max_sentences, max_words):
    if isinstance(documents, list):
        nb_rows = len(documents)
    else:
        raise ValueError('The documents must be of type list')

    padded_documents = np.zeros((nb_rows, max_sentences, max_words), dtype=np.int32)

    for parsed_document_i, parsed_document in enumerate(documents):

        i = len(parsed_document) - max_sentences
        i = 0 if i < 0 or i > max_sentences else i

        for parsed_sentence_i, parsed_sentence in enumerate(parsed_document):

            if i == max_sentences or i == len(parsed_document):
                break
            padded_documents[parsed_document_i, i, :] = pad_sequences([parsed_sentence],
                                                                      max_words)
            i += 1

    return padded_documents


def save_hdf5(padded_documents, labels=None, name=''):
    padded_documents_f = h5py.File('dataset/{}.hdf5'.format(name), 'w')
    padded_documents_f.create_dataset('x', padded_documents.shape, dtype='i', data=padded_documents)

    if labels is not None:
        padded_documents_f.create_dataset('y', labels.shape, dtype='i', data=labels)

    padded_documents_f.close()


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--train', default='dataset/train.csv')
    argument_parser.add_argument('--test', default='dataset/test.csv')
    argument_parser.add_argument('--max-sentences', default=8, type=int)
    argument_parser.add_argument('--max-words', default=15, type=int)
    args = argument_parser.parse_args()

    train_data = pd.read_csv(args.train)
    train_data = train_data.dropna()
    train_documents = train_data.comment_text.astype(str)
    train_documents = train_documents.str.lower()

    test_data = pd.read_csv(args.test)
    test_documents = test_data.comment_text.astype(str)
    test_documents = test_documents.fillna('')
    test_documents = test_documents.str.lower()

    all_documents = pd.concat([train_documents, test_documents]).reset_index(drop=True)
    print('fitting tokenizer...')
    tokenizer = setup_fit_tokenizer(all_documents)
    print('saving tokenizer...')
    p.dump(tokenizer, open('dataset/tokenizer.p', 'wb'))

    # Parse and prepare the training data
    parsed_train_documents = parse_sentences_words(train_documents, tokenizer)
    padded_train_documents = pad_parsed_sequences(parsed_train_documents, args.max_sentences,
                                                  args.max_words)
    train_labels = get_labels(train_data)
    save_hdf5(padded_train_documents, train_labels, 'padded_train')

    # Do the same for the test data
    parsed_test_documents = parse_sentences_words(test_documents, tokenizer)
    padded_test_documents = pad_parsed_sequences(parsed_test_documents, args.max_sentences,
                                                 args.max_words)
    save_hdf5(padded_test_documents, name='padded_test')

if __name__ == '__main__':
    main()