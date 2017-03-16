from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from allensdk.api.cache import cacheable, Cache
from BobAnalysis import cache_location
from BobAnalysis.core.utilities import get_sparse_noise_epoch_mask_list
from BobAnalysis.core.signature import session_signature_df_dict
import os
import pandas as pd

class SessionDFF(object):

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

        self.session_type = self.data.get_session_type()
        self.stimuli = self.data.list_stimuli()
        self.timestamps, self.traces = self.data.get_dff_traces()
        self.number_of_acquisition_frames = len(self.timestamps)
        self.csid_list = self.data.get_cell_specimen_ids()
        self.number_of_cells = len(self.csid_list)

if __name__ == "__main__":

    SessionDFF(oeid=530646083)