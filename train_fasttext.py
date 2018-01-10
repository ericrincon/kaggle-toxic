"""
Experiment script for the FastText style model
"""
from toxic_text.train.util import build_base_arg_parser
from toxic.train.experiment import run_experiment

from toxic_text.data.load import load_train_hdf5, load_test_hdf5


def load_train_data(path):
    x_train, y_train = load_train_hdf5(path)

    return x_train, y_train, 25001

def main():
    argument_parser = build_base_arg_parser()
    args = argument_parser.parse_args()

    run_experiment(args, model='fasttext', load_test_data=load_test_hdf5,
                   load_train_data=load_train_data)


if __name__ == '__main__':
    main()
