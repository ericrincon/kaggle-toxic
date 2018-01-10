import argparse
import os

def build_base_arg_parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--train', default='dataset/preprocessed_train.csv')
    argument_parser.add_argument('--test', default='dataset/test.csv')
    argument_parser.add_argument('--model', default='sentence')
    argument_parser.add_argument('--epochs', default=10, type=int)
    argument_parser.add_argument('--seq-length', default=150, type=int)
    argument_parser.add_argument('--word2vec')
    argument_parser.add_argument('--embedding-dim', default=64, type=int)
    argument_parser.add_argument('--batch-size', default=64, type=int)
    argument_parser.add_argument('--balanced', default=0, type=int)
    argument_parser.add_argument('--patience', default=5, type=int)
    argument_parser.add_argument('--experiment-name')
    argument_parser.add_argument('--valid-split', default=.1, type=float)

    return argument_parser


def get_experiment_name(args):
    if args.experiment_name is None:
        experiment_name = args.model
    else:
        experiment_name = args.experiment_name

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    return experiment_name