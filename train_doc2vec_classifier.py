"""
Traiing script for doc2vec
arguments from --model <model name>
"""

from toxic_text.train.util import build_base_arg_parser
import os.path
import pandas as pd

from toxic_text.data.load import load_train_hdf5, load_test_hdf5


def main():
    argument_parser = build_base_arg_parser()
    args = argument_parser.parse_args()

    x_train, y_train = load_train_hdf5(args.train)
    x_test = load_test_hdf5(args.test)

    log_dir = 'logs/{}'.format(args.experiment_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model =

    callbacks = setup_callbacks(log_dir=log_dir, patience=args.patience,
                                filepath='{}/model_checkpoint'.format(experiment_name))

    model.fit(x_train, y_train, args.batch_size,
              verbose=1, epochs=args.epochs, callbacks=callbacks, validation_split=args.valid_split)

    preds = model.predict(x_test)
    preds_df = pd.DataFrame(preds)

    test_data = pd.read_csv('dataset/test.csv')

    create_submission(preds_df, test_data, filepath='{}/submission.csv'.format(experiment_name))



if __name__ == '__main__':
    main()