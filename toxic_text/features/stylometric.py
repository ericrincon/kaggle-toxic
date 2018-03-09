import re


class CharacterFeatures:
    def __init__(self):
        """
        exclamation_point_count: number of exclamation points in a string
        """
        self.exclamation_point_count = re.compile('\!')
        self.char_count = re.compile('[^\s]]')

    def get_nb_exclamation_points(self, text):
        """
        Returns the number of "!" in a comment

        :param text:
        :return:
        """
        return text.apply(lambda x: len(self.exclamation_point_count.findall(x)))

    def get_char_count(self, text):
        """
        Returns the number of characters in a string excluding whitespace
        :param text: a pandas series of strings
        :return: number of characters in each string
        """
        return text.apply(lambda x: len(self.char_count.findall(x)))

