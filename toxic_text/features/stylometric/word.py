import spacy


class WordFeatures:
    def __init__(self):
        self.nlp = spacy.load('en')

    def get_word_count(self, text):
        """
        Return the numebr of words in each string
        :param text: a pandas series of strings
        :return: a series with the number of words in each element
        """
        return text.apply(
            lambda x: len([word for word in self.nlp(x, parse=False, tag=False, entity=False)]))
    