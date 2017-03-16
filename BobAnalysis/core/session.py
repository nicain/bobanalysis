from BobAnalysis.core.sessionstimulus import SessionStimulus
from BobAnalysis.core.sessiondff import SessionDFF
from BobAnalysis.core.roi import ROI
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from allensdk.api.cache import cacheable, Cache
from BobAnalysis import cache_location
from BobAnalysis.core.utilities import get_sparse_noise_epoch_mask_list
from BobAnalysis.core.signature import session_signature_df_dict
import os
import pandas as pd
from BobAnalysis.core.utilities import memoize, get_cache_array_numpy_memmap

class Session(object):

    def __init__(self, **kwargs):

        if 'oeid' in kwargs:
            self.oeid = kwargs['oeid']

            if 'brain_observatory_cache' in kwargs:
                brain_observatory_cache = kwargs['brain_observatory_cache']
            else:
                if 'manifest_file' in kwargs:
                    manifest_file = kwargs['manifest_file']
                else:
                    manifest_file = os.path.join(cache_location, 'boc_manifest.json')

                brain_observatory_cache = BrainObservatoryCache(manifest_file=manifest_file)

            self.data = brain_observatory_cache.get_ophys_experiment_data(self.oeid)

        elif 'brain_observatory_nwb_data_set' in kwargs:

            self.data = kwargs['brain_observatory_nwb_data_set']
            self.oeid = self.data.get_metadata()['ophys_experiment_id']

        else:
            raise RuntimeError('Construction not recognized')

        self.stimulus = SessionStimulus(brain_observatory_nwb_data_set=self.data)
        self.dff = SessionDFF(brain_observatory_nwb_data_set=self.data)


    @staticmethod
    @cacheable(query_strategy='lazy', **get_cache_array_numpy_memmap())
    def get_dff_array_cache(brain_observatory_nwb_data_set):
        return brain_observatory_nwb_data_set.get_dff_traces()[1]


    @staticmethod
    @memoize
    def get_dff_array(brain_observatory_nwb_data_set, oeid):
        return Session.get_dff_array_cache(brain_observatory_nwb_data_set, path=os.path.join(cache_location, str(oeid), 'dff.npy'))

    @staticmethod
    @memoize
    def get_roi(brain_observatory_nwb_data_set, cell_index):
        return ROI(brain_observatory_nwb_data_set=brain_observatory_nwb_data_set, cell_index=cell_index)

    @property
    def number_of_acquisition_frames(self):
        return self.stimulus.number_of_acquisition_frames

    @property
    def number_of_cells(self):
        return self.dff.number_of_cells


if __name__ == "__main__":

    S = Session(oeid=530646083)
    print Session.get_dff_array(S.data, S.oeid).shape