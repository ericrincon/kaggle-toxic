import pandas as pd
import numpy as np
import os.path
import pickle as p

from keras.preprocessing.sequence import pad_sequences

from models.model import get_model

from gensim.models.word2vec import Word2Vec

from util import build_base_arg_parser
from train import setup_callbacks
from data import create_submission, load_train, load_test
from models.model import build_embedding_matrix, \
    build_single_head_model, build_time_dist_model

from gensim.models import KeyedVectors


def main():
    argument_parser = build_base_arg_parser()
    args = argument_parser.parse_args()

    if args.experiment_name is None:
        experiment_name = args.model
    else:
        experiment_name = args.experiment_name

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    x_train, y_train = load_train(args.train)
    model = get_model(args.model)
    tokenizer = p.load(open('tokenizer.p', 'rb'))
    vocab_size = len(tokenizer.word_index)

    print('loading word2vec model...')

    if args.word2vec:
        word2vec = KeyedVectors.load_word2vec_format(args.word2vec)
        embedding_matrix = build_embedding_matrix(tokenizer, word2vec)
        embedding_dim = word2vec.vector_size
    else:
        embedding_matrix = None
        embedding_dim = args.embedding_dim

    model = build_time_dist_model(model, vocab_size, embedding_dim, 20,
                                  embedding_matrix=embedding_matrix, name='output')

    log_dir = 'logs/{}'.format(args.experiment_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    callbacks = setup_callbacks(log_dir=log_dir, patience=args.patience,
                                filepath='{}/model_checkpoint'.format(experiment_name))

    model.fit(x_train, y_train, args.batch_size,
              verbose=1, epochs=args.epochs, callbacks=callbacks, validation_split=args.valid_split)

    print('loading best model....')
    model.load_weights('model_checkpoint')
    print('\nCreating submission file...')

    test_data = pd.read_csv('dataset/test.csv')
    x_test = load_test(args.test)
    prob_predictions = model.predict(x_test)
    preds_df = pd.DataFrame(prob_predictions)

    # Create submission
    create_submission(preds_df, test_data, filepath='{}/submission.csv'.format(experiment_name))


if __name__ == '__main__':
    main()
