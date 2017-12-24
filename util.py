import argparse


def build_base_arg_parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--train', default='dataset/preprocessed_train.csv')
    argument_parser.add_argument('--test', default='dataset/test.csv')
    argument_parser.add_argument('--model', default='sentence')
    argument_parser.add_argument('--epochs', default=10, type=int)
    argument_parser.add_argument('--seq-length', default=80, type=int)
    argument_parser.add_argument('--word2vec', default='word2vec_models/word2vec.w2v')
    argument_parser.add_argument('--embedding-dim', default=64, type=int)
    argument_parser.add_argument('--batch-size', default=64, type=int)

    return argument_parser
