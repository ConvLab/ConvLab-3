import numpy as np
import logging


class BaseDataLoaders(object):
    def __init__(self, name):
        self.data_size = None
        self.indexes = None
        self.name = name

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // config.batch_size

        if verbose:
            print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))

        if shuffle and not fix_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i*self.batch_size: (i+1)*self.batch_size])

        if shuffle and fix_batch:
            self._shuffle_batch_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

    def _prepare_batch(self, *args, **kwargs):
        raise NotImplementedError('Have to override _prepare_batch()')

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens


class LongDataLoader(object):
    """A special efficient data loader for TBPTT. Assume the data contains
    N long sequences, each sequence has length k_i

    :ivar batch_size: the size of a minibatch
    :ivar backward_size: how many steps in time to do BP
    :ivar step_size: how fast we move the window
    :ivar ptr: the current idx of batch
    :ivar num_batch: the total number of batch
    :ivar batch_indexes: a list of list. Each item is the IDs in this batch
    :ivar grid_indexes: a list of (b_id, s_id, e_id). b_id is the index of
    batch, s_id is the starting time id in that batch and e_id is the ending
    time id.
    :ivar indexes: a list, the ordered of sequences ID it should go through
    :ivar data_size: the number of sequences, N.
    :ivar data_lens: a list containing k_i
    :ivar prev_alive_size:
    :ivar name: the name of the this data loader
    """
    logger = logging.getLogger()

    def __init__(self, name):
        self.batch_size = 0
        self.backward_size = 0
        self.step_size = 0
        self.ptr = 0
        self.num_batch = None
        self.batch_indexes = None  # one batch is a dialog
        self.grid_indexes = None  # grid is the tokenized versiion
        self.indexes = None
        self.data_lens = None
        self.data_size = None
        self.name = name

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _shuffle_grid_indexes(self):
        np.random.shuffle(self.grid_indexes)

    def _prepare_batch(self, cur_grid, prev_grid):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):

        assert len(self.indexes) == self.data_size and \
               len(self.data_lens) == self.data_size
        # make sure backward_size can be divided by step size
        assert config.backward_size % config.step_size == 0

        self.ptr = 0
        self.batch_size = config.batch_size
        self.backward_size = config.backward_size
        self.step_size = config.step_size

        # create batch indexes
        temp_num_batch = self.data_size // config.batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            self.batch_indexes.append(
                self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        left_over = self.data_size - temp_num_batch * config.batch_size
        if shuffle:
            self._shuffle_batch_indexes()

        # create grid indexes
        self.grid_indexes = []
        for idx, b_ids in enumerate(self.batch_indexes):
            # assume the b_ids are sorted
            all_lens = [self.data_lens[i] for i in b_ids]
            max_len = self.data_lens[b_ids[0]]
            min_len = self.data_lens[b_ids[-1]]
            assert np.max(all_lens) == max_len
            assert np.min(all_lens) == min_len
            num_seg = (max_len - self.backward_size - self.step_size) // self.step_size
            cut_start, cut_end = [], []
            if num_seg > 1:
                cut_start = list(range(config.step_size, num_seg * config.step_size, config.step_size))
                cut_end = list(range(config.backward_size + config.step_size,
                                num_seg * config.step_size + config.backward_size,
                                config.step_size))
                assert cut_end[-1] < max_len

            actual_size = min(max_len, config.backward_size)
            temp_end = list(range(2, actual_size, config.step_size))
            temp_start = [0] * len(temp_end)

            cut_start = temp_start + cut_start
            cut_end = temp_end + cut_end

            assert len(cut_end) == len(cut_start)
            new_grids = [(idx, s_id, e_id) for s_id, e_id in
                         zip(cut_start, cut_end) if s_id < min_len - 1]

            self.grid_indexes.extend(new_grids)

        # shuffle batch indexes
        if shuffle:
            self._shuffle_grid_indexes()

        self.num_batch = len(self.grid_indexes)
        if verbose:
            self.logger.info("%s init with %d batches with %d left over samples" %
                             (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_grid = self.grid_indexes[self.ptr]
            if self.ptr > 0:
                prev_grid = self.grid_indexes[self.ptr - 1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid,
                                       prev_grid=prev_grid)
        else:
            return None

    def pad_to(self, max_len, tokens, do_pad=True):
        if len(tokens) >= max_len:
            return tokens[0:max_len - 1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens
