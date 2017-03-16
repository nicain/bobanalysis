from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from allensdk.api.cache import cacheable, Cache
from BobAnalysis import cache_location
from BobAnalysis.core.utilities import get_sparse_noise_epoch_mask_list
from BobAnalysis.core.signature import session_signature_df_dict
import os
import pandas as pd
# import time

class SessionStimulus(object):

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
        self.timestamps = self.data.get_fluorescence_timestamps()
        self.number_of_acquisition_frames = len(self.timestamps)

        self.initialize_template_and_table_dicts()
        self.initialize_stimulus_lookup_dict()
        self.initialize_interval_df()

    def initialize_template_and_table_dicts(self):

        self.stimulus_template_dict = {}
        self.stimulus_table_dict = {}
        for stimulus in self.stimuli:

            self.stimulus_table_dict[stimulus] = self.data.get_stimulus_table(stimulus)

            if stimulus == 'spontaneous':
                self.stimulus_template_dict[stimulus] = np.full((1,16,28), 127, dtype=np.int8)
                self.stimulus_table_dict[stimulus]['frame'] = 0
            else:
                self.stimulus_template_dict[stimulus] = self.data.get_stimulus_template(stimulus)

    def initialize_interval_df(self):
        self.interval_df = self.get_interval_df(path=os.path.join(cache_location, str(self.oeid), 'interval_df.csv'))

    @cacheable(query_strategy='lazy', writer=lambda p, x : x.to_csv(p), reader=pd.DataFrame.from_csv)
    def get_interval_df(self):
        self.get_stimulus_lookup_dict(query_strategy='create')
        return self.interval_df

    def initialize_stimulus_lookup_dict(self):
        self.stimulus_lookup_dict = self.get_stimulus_lookup_dict(path=os.path.join(cache_location, str(self.oeid), 'session_stimulus.json'))

    @cacheable(query_strategy='lazy', **Cache.cache_json())
    def get_stimulus_lookup_dict(self):

        stimulus_lookup_dict = {}
        for stimulus in self.stimuli:
            for _, row in self.stimulus_table_dict[stimulus].iterrows():
                for fi in np.arange(row.start, row.end+1):
                    stimulus_lookup_dict[fi] = (stimulus, row.frame)


        # for fi in range(min(stimulus_lookup_dict.keys())):
        #     stimulus_lookup_dict[fi] = ('spontaneous', 0)
        #
        # for fi in np.arange(1+max(stimulus_lookup_dict.keys()), 1+self.number_of_acquisition_frames):
        #     stimulus_lookup_dict[fi] = ('spontaneous', 0)

        self.interval_list = []
        interval_stimulus_dict = {}
        for stimulus in self.stimuli:
            stimulus_interval_list = get_sparse_noise_epoch_mask_list(self.stimulus_table_dict[stimulus], self.number_of_acquisition_frames)
            for stimulus_interval in stimulus_interval_list:
                interval_stimulus_dict[stimulus_interval] = stimulus
            self.interval_list += stimulus_interval_list
        self.interval_list.sort(key=lambda x: x[0])

        stimulus_signature_list = ['gap']
        duration_signature_list = [self.interval_list[0][0]]
        interval_signature_list = [(0,self.interval_list[0][0])]
        for ii, interval in enumerate(self.interval_list):
            stimulus_signature_list.append(interval_stimulus_dict[interval])
            duration_signature_list.append(interval[1] - interval[0])
            interval_signature_list.append(interval)

            if ii != len(self.interval_list)-1:
                stimulus_signature_list.append('gap')
                duration_signature_list.append((self.interval_list[ii+1][0] - self.interval_list[ii][1]))
                interval_signature_list.append((self.interval_list[ii][1], self.interval_list[ii+1][0]))

        self.interval_df = pd.DataFrame({'stimulus':stimulus_signature_list,
                                         'duration':duration_signature_list,
                                         'interval':interval_signature_list})

        stimulus_signature_list.append('gap')
        interval_signature_list.append((self.interval_list[-1][1], self.number_of_acquisition_frames))
        duration_signature_list.append(interval_signature_list[-1][1]-interval_signature_list[-1][0])

        # print time.time() - t0

        for (ii, curr_row_template), (jj, curr_row), in zip(session_signature_df_dict[self.session_type].iterrows(), self.interval_df.head(len(self.interval_df)-1).tail(len(self.interval_df)-2).iterrows()):
            # print ii, jj
            assert ii == jj - 1
            assert curr_row_template.stimulus == curr_row.stimulus
            assert float((curr_row_template.duration - curr_row.duration))/curr_row_template.duration < 1e-2
            assert curr_row.duration == curr_row.interval[1] - curr_row.interval[0]

        for fi in range(self.number_of_acquisition_frames):
            if not fi in stimulus_lookup_dict:
                explained = False
                for _, curr_row in self.interval_df.iterrows():
                    if curr_row.interval[0] <= fi and fi <= curr_row.interval[1] and curr_row.stimulus == 'gap':
                        # try:
                        #     assert
                        # except:
                        #     print curr_row, fi
                            # raise
                        explained = True
                        stimulus_lookup_dict[fi] = ('spontaneous', 0)
                        break

                if explained == False:
                    for ii in np.arange(1,5):
                        if not stimulus_lookup_dict.get(fi-ii, None) is None:
                            stimulus_lookup_dict[fi] = stimulus_lookup_dict[fi-ii]
                            explained = True
                            break

                if explained == False:
                    raise Exception

        for fi in range(self.number_of_acquisition_frames):
            assert len(stimulus_lookup_dict[fi]) == 2

        return stimulus_lookup_dict

    def get_stimulus(self, fi):
        try:
            stimulus, frame = self.stimulus_lookup_dict[fi]
        except:
            stimulus, frame = self.stimulus_lookup_dict[str(fi)]
        return self.stimulus_template_dict[stimulus][frame, :, :]

if __name__ == "__main__":

    S = SessionStimulus(oeid=530646083)
    print S.interval_df
    # t0 = time.time()
    # S.get_stimulus_lookup_dict()
    # print time.time() - t0
    # print S.number_of_acquisition_frames
