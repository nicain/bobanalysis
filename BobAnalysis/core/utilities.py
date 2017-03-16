import numpy as np

def get_sparse_noise_epoch_mask_list(st, number_of_acquisition_frames, threshold=7):

    delta = (st.start.values[1:] - st.end.values[:-1])
    cut_inds = np.where(delta > threshold)[0] + 1

    epoch_mask_list = []

    if len(cut_inds) > 2:
        warnings.warn('more than 2 epochs cut')
        print '    ', len(delta), cut_inds

    for ii in range(len(cut_inds)+1):

        if ii == 0:
            first_ind = st.iloc[0].start
        else:
            first_ind = st.iloc[cut_inds[ii-1]].start

        if ii == len(cut_inds):
            last_ind_inclusive = st.iloc[-1].end
        else:
            last_ind_inclusive = st.iloc[cut_inds[ii]-1].end

        # curr_epoch_mask = np.zeros(number_of_acquisition_frames, dtype=np.bool)
        # curr_epoch_mask[first_ind:last_ind_inclusive+1] = True
        epoch_mask_list.append((first_ind,last_ind_inclusive))

    return epoch_mask_list

def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__

def get_cache_array_numpy_memmap():

    return {'reader':lambda p: np.load(p, mmap_mode='r'),
            'writer':np.save}

def get_cache_array_sparse_h5_reader_writer(value=0):

    import h5py
    import scipy.sparse as sps
    import numpy as np

    def writer(p, x):

        inds = np.where(x != value)
        indices = np.array(inds)
        data = x[inds]
        shape = x.shape

        f = h5py.File(p, 'w')
        f['data'] = data
        f['indices'] = indices
        f['shape'] = shape
        f['value'] = value
        f.close()

    def reader(p):

        f = h5py.File(p, 'r')
        data = f['data'].value
        indices = f['indices'].value
        shape = f['shape'].value
        value = f['value'].value
        f.close()

        x = np.full(shape, value, dtype=data.dtype)
        x[tuple(indices)] = data
        # print indices.shape
        # import sys
        # sys.exit()

        return x

    return {
         'writer': writer,
         'reader': reader
    }

# import time
# @memoize
# def f():
#     time.sleep(1)
#     return 1
#
# for _ in range(2):
#     t0 = time.time()
#     f()
#     print time.time() - t0


# class memoize(object):
#     def __init__(self, func):
#         self.func = func
#
#     def __get__(self, obj, objtype=None):
#         if obj is None:
#             return self.func
#         return partial(self, obj)
#
#     def __call__(self, *args, **kw):
#         obj = args[0]
#         try:
#             cache = obj.__cache
#         except AttributeError:
#             cache = obj.__cache = {}
#         key = (self.func, args[1:], frozenset(kw.items()))
#         try:
#             res = cache[key]
#         except KeyError:
#             res = cache[key] = self.func(*args, **kw)
#         return res