"""
Experiment script for the FastText style model
"""
from util import build_base_arg_parser
from train import run_experiment

from data import load_train, load_test


def load_train_data(path):
    x_train, y_train = load_train(path)

    return x_train, y_train, 25001

def main():
    argument_parser = build_base_arg_parser()
    args = argument_parser.parse_args()

    run_experiment(args, model='fasttext', load_test_data=load_test,
                   load_train_data=load_train_data)


if __name__ == '__main__':
    main()
