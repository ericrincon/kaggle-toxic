import numpy as np

from evaluate import TARGET_NAMES

from keras.callbacks import TensorBoard, EarlyStopping

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


def setup_callbacks():
    tensorboard = TensorBoard(log_dir='logs')
    early_stopping = EarlyStopping()

    return [tensorboard]