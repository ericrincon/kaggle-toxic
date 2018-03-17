import re


class CharacterFeatures:
    def __init__(self, documents):
        """
        exclamation_point_count: number of exclamation points in a string
        """
        self.documents = documents
        self.exclamation_point_count = re.compile('\!')
        self.char_count = re.compile('[^\s]')
        self.upper_case = re.compile('[A-Z]')



    def get_nb_exclamation_points(self):
        """
        Returns the number of "!" in a comment

        :param text: a pandas series of strings
        :return: pandas series where each element is the numebr of exclamation
        points in that element
        """
        return self.documents.apply(lambda x: len(self.exclamation_point_count.findall(x)))

    def get_char_count(self):
        """
        Returns the number of characters in a string excluding whitespace
        :param text: a pandas series of strings
        :return: pandas series where each element is
        the number of characters in that string
        """
        return self.documents.apply(lambda x: len(self.char_count.findall(x)))

    def upper_case_ratio(self):
        """
        Get the number of upper case characters over the total number
        of characters excluding whitespace

        :return: pandas series where each element is
        the ratio of uppercase over total counts
        """

        char_counts = self.get_char_count()
        upper_case_counts = self.documents.apply(lambda x: len(self.upper_case.findall(x)))

        return upper_case_counts.astype(float) / char_counts.astype(float)

