import numpy as np
from evaluate import TARGET_NAMES

# class BalancedFit():
#     """
#     Samples the data so that each epoch has
#     equal number of positive and negative examples
#
#     """
#
#     def __init__(self, x, y, batch_size):
#         self.x = x
#         self.y = y
#         self.batch_size = batch_size
#
#         self.undersample_class = {
#             'toxic': 1,
#             'severe_toxic': 0,
#             'obscene':1,
#             'threat': 0,
#             'insult': 0,
#             'identity_hate': 0
#         }
#
#
#         self.class_undersample_indices = {}
#
#         for i, name in enumerate(TARGET_NAMES):
#             self.class_undersample_indices[name] = np.where(y[i] == self.undersample_class[name])[0]
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         while True:
#             class_undersample_indices = self.class_undersample_indices.copy()
#
#             # x = []
#             # y = [[] for _ in range(len(TARGET_NAMES))]
#             indices = []
#
#             for i, name in enumerate(TARGET_NAMES):
#                 undersampled_indices = class_undersample_indices[name]
#
#                 mask = np.ones(self.x.shape[0], dtype=bool)
#                 mask[undersampled_indices] = False
#                 nonundersampled_indices = np.arange(self.x.shape[0])[mask]
#                 undersampled_indices = np.random.choice(undersampled_indices,
#                                                         nonundersampled_indices.shape[0],
#                                                         replace=False)
#                 # mask[undersampled_indices] = False
#                 #
#                 # nonundersampled_x = self.x[mask]
#                 # nonundersampled_y = self.[y][mask]
#                 #
#                 # undersampled_x = self.x[undersampled_indices]
#                 # undersampled_y = self.y[i][undersampled_indices]
#
#                 #
#                 # x.extend([nonundersampled_x, undersampled_x])
#                 # y[i].extend([nonundersampled_y, undersampled_y])
#                 indices.extend(undersampled_indices.tolist())
#                 indices.extend(nonundersampled_indices.tolist())
#
#             # indices = np.arange(self.x.shape[0])
#             np.random.shuffle(indices)
#
#
#             batch_index_start = 0
#
#             for batch_index_end in range(0, len(indices), self.batch_size):
#                 batch_indices = indices[batch_index_start: batch_index_end]
#                 batch_index_start = batch_index_end
#                 print(batch_index_end)
#                 print({'input': self.x[batch_indices]})
#                 print({name: self.y[i][batch_indices] for i, name in enumerate(TARGET_NAMES)})
#                 yield {'input': self.x[batch_indices]},\
#                        {name: self.y[i][batch_indices] for i, name in enumerate(TARGET_NAMES)}

def get_steps_per_epoch(batch_size):
    return int(24166 / batch_size)


def balanced_fit(x, y, batch_size):
    undersample_class = {
        'toxic': 1,
        'severe_toxic': 0,
        'obscene': 1,
        'threat': 0,
        'insult': 0,
        'identity_hate': 0
    }

    class_undersample_indices = {}

    for i, name in enumerate(TARGET_NAMES):
        class_undersample_indices[name] = np.where(y[i] == undersample_class[name])[0]

    while True:
        class_undersample_indices = class_undersample_indices.copy()

        # x = []
        # y = [[] for _ in range(len(TARGET_NAMES))]
        indices = []

        for i, name in enumerate(TARGET_NAMES):
            undersampled_indices = class_undersample_indices[name]

            mask = np.ones(x.shape[0], dtype=bool)
            mask[undersampled_indices] = False
            nonundersampled_indices = np.arange(x.shape[0])[mask]
            undersampled_indices = np.random.choice(undersampled_indices,
                                                    nonundersampled_indices.shape[0],
                                                    replace=False)

            indices.extend(undersampled_indices.tolist())
            indices.extend(nonundersampled_indices.tolist())

        # indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        batch_index_start = 0

        for batch_index_end in range(batch_size, len(indices), batch_size):
            batch_indices = indices[batch_index_start: batch_index_end]
            batch_index_start = batch_index_end

            yield {'input': x[batch_indices]}, \
                  {name: y[i][batch_indices] for i, name in enumerate(TARGET_NAMES)}
