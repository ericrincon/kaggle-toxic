import argparse
import pandas as pd
import pickle as p

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard

from models.model import get_model, build_model, build_embedding_matrix

from gensim.models.word2vec import Word2Vec

from evaluate import get_targets, plot_target_freq
from train import balanced_fit, get_steps_per_epoch

def create_submission(model, test_data, tokenizer, seq_length):
    test_texts = test_data['comment_text'].astype(str)

    test_examples = tokenizer.texts_to_sequences(test_texts)
    test_examples = pad_sequences(test_examples, seq_length)

    preds = model.predict(test_examples)

    class_predictions = get_targets(preds)
    p.dump(class_predictions, open('class_preds.p', 'wb'))

    plot_target_freq(class_predictions)


    series = list(map(lambda x: pd.Series(x.flatten()), preds))
    preds_df = [test_data['id']]
    preds_df.extend(series)

    preds_df = pd.concat(preds_df, axis=1)
    preds_df.columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat',
                        'insult', 'identity_hate']
    preds_df.to_csv('submission.csv', index=False)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--train', default='data/preprocessed_train.csv')
    argument_parser.add_argument('--test', default='data/test.csv')
    argument_parser.add_argument('--model', default='sentence')
    argument_parser.add_argument('--epochs', default=10, type=int)
    argument_parser.add_argument('--seq-length', default=80, type=int)
    argument_parser.add_argument('--word2vec', default='word2vec_models/word2vec.w2v')
    argument_parser.add_argument('--embedding-dim', default=64, type=int)
    argument_parser.add_argument('--batch-size', default=64, type=int)
    args = argument_parser.parse_args()

    tokenizer = Tokenizer()

    train_data = pd.read_csv(args.train)

    texts = train_data['comment_text']
    labels = [train_data.toxic, train_data.severe_toxic, train_data.obscene,
                        train_data.threat, train_data.insult, train_data.identity_hate]
    labels = list(map(lambda x: x.values, labels))


    tokenizer.fit_on_texts(texts)

    examples = tokenizer.texts_to_sequences(texts)
    examples = pad_sequences(examples, args.seq_length)
    model = get_model(args.model)

    vocab_size = len(tokenizer.word_index)

    if args.word2vec:
        word2vec = Word2Vec.load(args.word2vec)
        embedding_matrix = build_embedding_matrix(tokenizer, word2vec)
        embedding_dim = word2vec.vector_size
    else:
        embedding_matrix = None
        embedding_dim = args.embedding_dim

    model = build_model(model, vocab_size, embedding_dim, args.seq_length,
                        embedding_matrix=embedding_matrix)

    print('------------------Summary------------------')
    print(model.summary())
    print('Vocab size: {}\n'.format(vocab_size))

    # model.fit(examples, labels, epochs=args.epochs, verbose=2, batch_size=64,
    #           validation_split=0.1)
    balanaced_fit_gen = balanced_fit(examples, labels, args.batch_size)

    # Set up callbacks
    tensorboard = TensorBoard(log_dir='logs')


    model.fit_generator(balanaced_fit_gen, steps_per_epoch=get_steps_per_epoch(args.batch_size),
                        verbose=2, epochs=args.epochs, callbacks=[tensorboard])
    print('\nCreating submission file...')
    test_data = pd.read_csv(args.test)

    create_submission(model, test_data, tokenizer, args.seq_length)




if __name__ == '__main__':
    main()