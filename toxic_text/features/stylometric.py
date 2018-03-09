import re


class CharacterFeatures:
    def __init__(self):
        """
        exclamation_point_count: number of exclamation points in a string
        """
        self.exclamation_point_count = re.compile('\!')

    def get_nb_exclamation_points(self, text):
        """
        Returns the number of "!" in a comment

        :param text:
        :return:
        """
        return text.apply(lambda x: len(self.exclamation_point_count.findall(x)))

