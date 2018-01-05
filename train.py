import numpy as np
import pandas as pd
import os.path

from evaluate import TARGET_NAMES

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

from models.model import get_model

from gensim.models.word2vec import Word2Vec

from data import create_submission, get_training_data, load_setup_fit_tokenizer
from models.model import build_embedding_matrix, \
    build_single_head_model

from util import get_experiment_name

from gensim.models import KeyedVectors


def get_steps_per_epoch(batch_size):
    return int(24166 / batch_size)


def balanced_fit(x, y, batch_size, class_name=None):
    undersample_class = {
        'toxic': 1,
        'severe_toxic': 0,
        'obscene': 1,
        'threat': 0,
        'insult': 0,
        'identity_hate': 0
    }

    if class_name:
        target_names = [class_name]
    else:
        target_names = TARGET_NAMES

    class_undersample_indices = {name: np.where(y[i] == undersample_class[name])[0]
                                 for i, name in enumerate(target_names)}

    while True:
        class_undersample_indices_copy = class_undersample_indices.copy()

        indices = [[] for _ in range(len(target_names))]

        for i, name in enumerate(target_names):
            undersampled_indices = class_undersample_indices_copy[name]

            mask = np.ones(x.shape[0], dtype=bool)
            mask[undersampled_indices] = False

            nonundersampled_indices = np.arange(x.shape[0])[mask]
            undersampled_indices = np.random.choice(undersampled_indices,
                                                    nonundersampled_indices.shape[0],
                                                    replace=False)

            indices[i].extend(undersampled_indices.tolist())
            indices[i].extend(nonundersampled_indices.tolist())


        for class_i, class_indices in enumerate(indices):
            np.random.shuffle(class_indices)
            batch_index_start = 0

            for batch_index_end in range(batch_size, len(class_indices), batch_size):
                batch_indices = class_indices[batch_index_start: batch_index_end]
                batch_index_start = batch_index_end

                yield {'input': x[batch_indices]}, \
                      {name: y[i][batch_indices] for i, name in enumerate(target_names)}


def setup_callbacks(log_dir='logs', patience=5, filepath='model_checkpoint'):
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True)

    return [tensorboard, early_stopping, model_checkpoint]


def run_experiment(args, model=None, load_train_data=None, load_test_data=None):
    """
    Takes in parsed command line arguments and an optional model name that overrides
    the args.model and runs a generic experiment
    :param args: parsed command line arguments
    :param model: a model name will override name passsed in the args
    :return:
    """

    experiment_name = get_experiment_name(args)

    if model is None:
        model = args.model
    model = get_model(model)

    if load_train_data:
        x_train, y_train, vocab_size = load_train_data(args.train)
    else:
        texts, y_train = get_training_data(args.train)
        tokenizer, x_train = load_setup_fit_tokenizer(texts, args.seq_length)
        vocab_size = len(tokenizer.word_index)

    if args.word2vec:
        if args.word2vec.split('.')[-1] == 'w2v':
            word2vec = Word2Vec.load(args.word2vec)
        elif args.word2vec.split('.')[-1] == 'vec':
            word2vec = KeyedVectors.load_word2vec_format(args.word2vec)
        else:
            raise ValueError('Cant load format {}'.format(args.word2vec.split('.')[-1]))
        embedding_matrix = build_embedding_matrix(tokenizer, word2vec)
        embedding_dim = word2vec.vector_size
    else:
        embedding_matrix = None
        embedding_dim = args.embedding_dim

    test_data = pd.read_csv('dataset/test.csv')

    if load_test_data:
        x_test = load_test_data(args.test)
    else:
        test_texts = test_data['comment_text'].astype(str)

        x_test = tokenizer.texts_to_sequences(test_texts)
        x_test = pad_sequences(x_test, args.seq_length)


    model = build_single_head_model(model, vocab_size, embedding_dim, args.seq_length,
                                    name=args.model, embedding_matrix=embedding_matrix)
    log_dir = 'logs/{}'.format(args.experiment_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    callbacks = setup_callbacks(log_dir=log_dir, patience=args.patience,
                                filepath='{}/model_checkpoint'.format(experiment_name))

    model.fit(x_train, y_train, args.batch_size,
                    verbose=1, epochs=args.epochs, callbacks=callbacks, validation_split=args.valid_split)

    preds = model.predict(x_test)
    preds_df = pd.DataFrame(preds)

    create_submission(preds_df, test_data, filepath='{}/submission.csv'.format(experiment_name))


