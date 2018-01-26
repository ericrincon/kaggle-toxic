import numpy as np
import pandas as pd
import os.path
from operator import itemgetter

from toxic_text.test.evaluate import TARGET_NAMES

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

from toxic_text.models.keras.model import get_model

from gensim.models.word2vec import Word2Vec

from toxic_text.data.load import create_submission, get_training_data, load_setup_fit_tokenizer
from toxic_text.models.keras.model import build_embedding_matrix, \
    build_single_head_model

from toxic_text.train.util import get_experiment_name

from gensim.models import KeyedVectors

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from keras import backend as K

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


def create_embedding_matrix(args, tokenizer=None):
    if args.word2vec:
        assert tokenizer is not None, 'you must pass a tokenizer object!'

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

    return embedding_matrix, embedding_dim


def setup_lda_parameters(args):
    if args.use_lda:
        lda = {
            'nb_topics': args.nb_topics,
            'embedding_dim': args.lda_dim,
            'max_topics': 10
        }

    else:
        lda = None

    return lda


def setup_log_dir(args):
    log_dir = 'logs/{}'.format(args.experiment_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir


def setup_training_data(args, load_train_data=None):
    if load_train_data:
        x_train, y_train, vocab_size = load_train_data(args.train)
        tokenizer = None
    else:
        texts, y_train = get_training_data(args.train)
        tokenizer, x_train = load_setup_fit_tokenizer(texts, args.seq_length, args.max_words)
        vocab_size = len(tokenizer.word_index)

    return np.array(x_train), np.array(y_train), tokenizer, vocab_size


def run_experiment(args, model=None, load_train_data=None, load_test_data=None):
    """
    Takes in parsed command line arguments and an optional model name that overrides
    the args.model and runs a generic experiment
    :param args: parsed command line arguments
    :param model: a model name will override name passsed in the args
    :return:
    """

    experiment_name_output_dir = get_experiment_name(args)

    test_data = pd.read_csv('dataset/test.csv')

    if model is None:
        model = args.model
    model_builder = get_model(model)

    x_train, y_train, tokenizer, vocab_size = setup_training_data(args, load_train_data=load_train_data)

    embedding_matrix, embedding_dim  = create_embedding_matrix(args, tokenizer)

    lda = setup_lda_parameters(args)

    log_dir = setup_log_dir(args)

    callbacks = setup_callbacks(log_dir=log_dir, patience=args.patience,
                                filepath='{}/model_checkpoint'.format(experiment_name_output_dir))
    if load_test_data:
        x_test = load_test_data(args.test)
    else:
        test_texts = test_data['comment_text'].astype(str)

        x_test = tokenizer.texts_to_sequences(test_texts)
        x_test = pad_sequences(x_test, args.seq_length)

    if lda:
        x_train_lda = np.load(open(args.lda_train, 'rb'))
        x_train = [x_train, x_train_lda]

        x_test_lda = np.load(open(args.lda_test, 'rb'))
        x_test = [x_test, x_test_lda]


    def train(X_train, Y_train, X_test):
        model = build_single_head_model(model_builder, vocab_size, embedding_dim, args.seq_length,
                                        name=args.model, embedding_matrix=embedding_matrix,
                                        lda=lda)

        history = model.fit(X_train, Y_train, args.batch_size,
                            verbose=1, epochs=args.epochs, callbacks=callbacks, validation_split=args.valid_split)

        # Refit model with all the data at that epoch
        early_stop_nb_epochs = min(enumerate(history.history['val_loss']), key=itemgetter(1))[0] + 1

        print('--------------------------------------------------')
        print('Creating new model and retraining on all data...')
        del model

        model = build_single_head_model(model_builder, vocab_size, embedding_dim, args.seq_length,
                                        name=args.model, embedding_matrix=embedding_matrix, lda=lda)
        history = model.fit(X_train, Y_train, args.batch_size,
                            verbose=1, epochs=early_stop_nb_epochs, callbacks=callbacks
                            , validation_split=0)

        preds = model.predict(X_test)

        K.clear_session()

        return preds

    if args.cv:
        """
        Here we use the training data as run 5 fold cross validation to test different models 
        """
        kf = KFold(n_splits=5, random_state=1234)

        def index_x(X, indices):
            if not isinstance(X, list):
                X = [X]

            X = list(map(lambda x: x[indices, :], X))

            return X.pop(0) if len(X) == 0 else X

        log_loss_scores = []

        # Use the training data for the cross validation
        # so the training data will be split into test and training
        for train_indices, test_indices in kf.split(y_train):
            X_train, Y_train = index_x(x_train, train_indices), y_train[train_indices]
            X_test, Y_test = index_x(x_train, test_indices), y_train[test_indices]

            preds = train(X_train, Y_train, X_test)

            log_loss_score = float(sum(map(lambda t: log_loss(t[0], t[1]), [(Y_test[:, i], preds[:, i]) for i in range(Y_test.shape[1])])))\
                             / Y_test.shape[1]

            log_loss_scores.append(log_loss_score)

        average_log_loss_score = sum(log_loss_scores) / len(log_loss_scores)

        print('5 Fold Log loss score: {}'.format(average_log_loss_score))

    else:
        preds = train(x_train, y_train, x_test)
        output_results(args, preds, test_data)



def output_results(args, preds, test_data):
    """
    Helper function for calling both commonly called submission and ensemble
    output functions

    :param preds: the predicted probabilities from a model
    :param test_data: test data csv
    :param experiment_name: the name for the experiment e.g., birnn_first_run
    :param ensemble_dir: dir where to save submission for ensemble
    :return:
    """
    experiment_name_output_dir = get_experiment_name(args)

    submission_csv = output_submission(preds, test_data, experiment_name_output_dir)
    output_to_ensemble(submission_csv, args.ensemble_dir, args.experiment_name)


def output_submission(preds, test_data, experiment_name):
    """
    Creates a submission csv for submitting to the kaggle competition
    saves it to the the directory from the experiment_name variable and
    returns the dataframe

    :param preds: the predicted probabilities from a model
    :param test_data: test data csv
    :param experiment_name: the name for the experiment e.g., birnn_first_run
    :return:
    """

    preds_df = pd.DataFrame(preds)

    submission_csv = create_submission(preds_df, test_data)

    submission_csv.to_csv('{}/submission.csv'.format(experiment_name), index=False)

    return submission_csv


def output_to_ensemble(submission_csv, ensemble_dir, experiment_name):
    if not os.path.exists(ensemble_dir):
        print('The dir {} doesn\'t exist!\nCreating the dir...'.format(ensemble_dir))
        os.makedirs(ensemble_dir)

    submission_csv.to_csv('{}/{}_submission.csv'.format(ensemble_dir, experiment_name), index=False)


