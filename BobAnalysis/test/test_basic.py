
def test_basic():

    import sys
    import os
    sys.path.append(os.path.join(sys.path[0],'../..'))
    import BobAnalysis as tp
    reload(tp)

def test_cache():

    from BobAnalysis.core.utilities import get_cache_array_sparse_h5_reader_writer
    from BobAnalysis import cache_location
    from allensdk.api.cache import cacheable
    import numpy as np
    import os


    @cacheable(strategy='lazy', **get_cache_array_sparse_h5_reader_writer())
    def f():
        np.random.seed(0)
        data = np.random.rand(3, 3, 3)
        data[data>.5] = 0
        return data

    np.random.seed(0)
    data = np.random.rand(3,3,3)
    data[data > .5] = 0

    data_2 = f(path=os.path.join(cache_location, 'tmp.h5'))
    np.testing.assert_allclose(data, data_2)
    
if __name__ == '__main__':                                    # pragma: no cover
    test_basic()                                              # pragma: no cover
    test_cache()