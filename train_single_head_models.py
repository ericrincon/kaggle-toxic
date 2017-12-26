import pandas as pd
import os.path
from models.model import get_model

from gensim.models.word2vec import Word2Vec

from util import build_base_arg_parser
from train import balanced_fit, get_steps_per_epoch, setup_callbacks, TARGET_NAMES
from data import get_training_data, get_train_valid_split, create_submission, \
    setup_fit_tokenizer, preds_to_df
from keras.preprocessing.sequence import pad_sequences

from models.model import build_embedding_matrix, build_single_head_model


def main():
    argument_parser = build_base_arg_parser()
    args = argument_parser.parse_args()

    texts, labels = get_training_data(args.train)
    tokenizer, examples = setup_fit_tokenizer(texts, args.seq_length)
    vocab_size = len(tokenizer.word_index)

    if args.word2vec:
        word2vec = Word2Vec.load(args.word2vec)
        embedding_matrix = build_embedding_matrix(tokenizer, word2vec)
        embedding_dim = word2vec.vector_size
    else:
        embedding_matrix = None
        embedding_dim = args.embedding_dim


    test_data = pd.read_csv(args.test)

    test_texts = test_data['comment_text'].astype(str)

    test_examples = tokenizer.texts_to_sequences(test_texts)
    test_examples = pad_sequences(test_examples, args.seq_length)

    x_train, x_valid, y_train, y_valid = get_train_valid_split(examples, labels)

    preds_list = []

    for i, (model_name, y_train_i) in enumerate(zip(TARGET_NAMES, y_train)):
        model = get_model(args.model)
        model = build_single_head_model(model, vocab_size, embedding_dim, args.seq_length,
                                        name=model_name, embedding_matrix=embedding_matrix)
        print('\nTraining model on {} data'.format(model_name))




        # Set up callbacks
        log_dir = 'logs/{}'.format(model_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        callbacks = setup_callbacks(log_dir=log_dir)

        if args.balanced:
            balanaced_fit_gen = balanced_fit(x_train, [y_train_i], args.batch_size,
                                             class_name=model_name)
            model.fit_generator(balanaced_fit_gen, steps_per_epoch=get_steps_per_epoch(args.batch_size),
                                verbose=2, epochs=args.epochs, callbacks=callbacks,
                                validation_data=(x_valid, y_valid[i]))
        else:
            model.fit(x_train, y_train_i, args.batch_size,
                        verbose=2, epochs=args.epochs, callbacks=callbacks,
                        validation_data=(x_valid, y_valid[i]))

        preds = model.predict(test_examples)
        preds_list.append(preds)

    print('\nCreating submission file...')
    preds_df = preds_to_df(preds_list)

    create_submission(preds_df, test_data)




if __name__ == '__main__':
    main()