import pandas as pd
import numpy as np
import os.path
import pickle as p

from keras.preprocessing.sequence import pad_sequences

from models.model import get_model

from gensim.models.word2vec import Word2Vec

from util import build_base_arg_parser
from train import balanced_fit, get_steps_per_epoch, setup_callbacks
from data import get_training_data, create_submission, load_train, load_test
from models.model import build_embedding_matrix, build_multi_head_model, \
    build_single_head_model, build_time_dist_model


def main():
    argument_parser = build_base_arg_parser()
    args = argument_parser.parse_args()

    x_train, y_train = load_train(args.train)
    model = get_model(args.model)
    tokenizer = p.load(open('tokenizer.p', 'rb'))
    vocab_size = len(tokenizer.word_index)

    if args.word2vec:
        word2vec = Word2Vec.load(args.word2vec)
        embedding_matrix = build_embedding_matrix(tokenizer, word2vec)
        embedding_dim = word2vec.vector_size
    else:
        embedding_matrix = None
        embedding_dim = args.embedding_dim

    model = build_time_dist_model(model, vocab_size, embedding_dim, 15,
                            embedding_matrix=embedding_matrix, name='output')

    # x_train, x_valid, y_train, y_valid = get_train_valid_split(examples, labels)

    log_dir = 'logs/{}'.format(args.model)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    callbacks = setup_callbacks(log_dir=log_dir, patience=args.patience)

    if args.balanced:
        balanaced_fit_gen = balanced_fit(x_train, y_train, args.batch_size,
                                         class_name=None)
        model.fit_generator(balanaced_fit_gen, steps_per_epoch=get_steps_per_epoch(args.batch_size),
                            verbose=2, epochs=args.epochs, callbacks=callbacks,
                            validation_data=(x_valid, y_valid))
    else:
        model.fit(x_train, y_train, args.batch_size,
                  verbose=1, epochs=args.epochs, callbacks=callbacks, validation_split=0.1)

    print('loading best model....')
    model.load_weights('model_checkpoint')
    print('\nCreating submission file...')
    test_data = pd.read_csv(args.test)
    x_test = load_test(args.train)

    test_texts = test_data['comment_text'].astype(str)

    test_examples = tokenizer.texts_to_sequences(test_texts)
    test_examples = pad_sequences(test_examples, args.seq_length)

    prob_predictions = model.predict(test_examples)

    preds_df = pd.DataFrame(prob_predictions)
    create_submission(preds_df, test_data)




if __name__ == '__main__':
    main()