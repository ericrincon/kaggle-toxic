"""
NOTE: these functions are not optimized for speed since the dataset they are being applied to
are small...
"""

import spacy
import numpy as np


class WordFeatures:
    def __init__(self, documents):
        self.nlp = spacy.load('en')
        self.documents = documents

        # Since this can be expensive
        self.word_count_list = None


    def tokenize_documents(self):
        """
        Tokennizes each document into words
        :return:
        """

        self.word_count_list = self.documents.apply(
            lambda x: [word for word in self.nlp(x, disable=['parser', 'ner', 'tagger'])])

        return self.word_count_list

    def get_word_counts(self):
        """
        Return the number of words in each string
        :param text: a pandas series of strings
        :return: a series with the number of words in each element
        """

        if self.word_count_list is None:
            self.tokenize_documents()

        return self.word_count_list.apply(lambda x: len(x))


    def get_average_word_length(self):
        """
        Computes the average word length for each document
        :return:
        """

        if self.word_count_list is None:
            self.tokenize_documents()

        return self.word_count_list.apply(lambda x: np.average([len(w) for w in x]))


    def get_herdans_c(self, text):
        """
        Herdan's C (Herdan, 1960, as cited in Tweedie & Baayen, 1998;
        sometimes referred to as LogTTR):

        C = log(V) / log(N)

        V: number of types (i.e., the unique number of words???? Not very well defined.....)
        N: number of tokens the total number of tokens in the individual text

        :param text: a pandas series of strings
        :return: a series with C as the value
        """
        V = text.apply(lambda x: len(set(x)))
        N = self.get_word_counts()

        return np.log(V) / np.log(N)
