import pandas as pd

from keras.preprocessing.sequence import pad_sequences

from models.model import get_model

from gensim.models.word2vec import Word2Vec

from util import build_base_arg_parser
from train import balanced_fit, get_steps_per_epoch, setup_callbacks
from data import get_training_data, get_train_valid_split, create_submission, \
    setup_fit_tokenizer
from models.model import build_embedding_matrix, build_multi_head_model


def main():
    argument_parser = build_base_arg_parser()
    args = argument_parser.parse_args()

    texts, labels = get_training_data(args.train)
    model = get_model(args.model)
    tokenizer, examples = setup_fit_tokenizer(texts, args.seq_length)
    vocab_size = len(tokenizer.word_index)

    if args.word2vec:
        word2vec = Word2Vec.load(args.word2vec)
        embedding_matrix = build_embedding_matrix(tokenizer, word2vec)
        embedding_dim = word2vec.vector_size
    else:
        embedding_matrix = None
        embedding_dim = args.embedding_dim

    model = build_multi_head_model(model, vocab_size, embedding_dim, args.seq_length,
                            embedding_matrix=embedding_matrix)

    print('------------------Summary------------------')
    print(model.summary())
    print('Vocab size: {}\n'.format(vocab_size))

    x_train, x_valid, y_train, y_valid = get_train_valid_split(examples, labels)
    print(x_train.shape)

    balanaced_fit_gen = balanced_fit(x_train, y_train, args.batch_size)

    # Set up callbacks
    callbacks = setup_callbacks()

    model.fit_generator(balanaced_fit_gen, steps_per_epoch=get_steps_per_epoch(args.batch_size),
                        verbose=2, epochs=args.epochs, callbacks=callbacks,
                        validation_data=(x_valid, y_valid))
    print('\nCreating submission file...')
    test_data = pd.read_csv(args.test)

    create_submission(model, test_data, tokenizer, args.seq_length)




if __name__ == '__main__':
    main()