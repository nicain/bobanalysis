from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from allensdk.api.cache import cacheable, Cache
from BobAnalysis import cache_location
from BobAnalysis.core.utilities import get_sparse_noise_epoch_mask_list
from BobAnalysis.core.signature import session_signature_df_dict
import os
import pandas as pd
from BobAnalysis.core.utilities import memoize, get_cache_array_sparse_h5_reader_writer
import time
import collections

class ROI(object):



    def __init__(self, **kwargs):

        if 'csid' in kwargs and 'oeid' in kwargs:

            self.oeid = kwargs['oeid']
            self.csid = kwargs['csid']

            if 'brain_observatory_cache' in kwargs:
                brain_observatory_cache = kwargs['brain_observatory_cache']
            else:
                if 'manifest_file' in kwargs:
                    manifest_file = kwargs['manifest_file']
                else:
                    manifest_file = os.path.join(cache_location, 'boc_manifest.json')

                brain_observatory_cache = BrainObservatoryCache(manifest_file=manifest_file)

            self.data = brain_observatory_cache.get_ophys_experiment_data(self.oeid)
            self.cell_index = ROI.get_all_cell_specimen_indices_dict(self.data)[self.csid]

        elif 'oeid' in kwargs and 'cell_index' in kwargs:

            self.oeid = kwargs['oeid']
            self.cell_index = kwargs['cell_index']

            if 'brain_observatory_cache' in kwargs:
                brain_observatory_cache = kwargs['brain_observatory_cache']
            else:
                if 'manifest_file' in kwargs:
                    manifest_file = kwargs['manifest_file']
                else:
                    manifest_file = os.path.join(cache_location, 'boc_manifest.json')

                brain_observatory_cache = BrainObservatoryCache(manifest_file=manifest_file)


            self.data = brain_observatory_cache.get_ophys_experiment_data(self.oeid)
            self.csid = ROI.get_cell_specimen_ids(self.data)[self.cell_index]

        elif 'brain_observatory_nwb_data_set' in kwargs and 'cell_index' in kwargs:

            self.data = kwargs['brain_observatory_nwb_data_set']
            self.cell_index = kwargs['cell_index']
            self.oeid = ROI.get_metadata(self.data)
            self.csid = ROI.get_cell_specimen_ids(self.data)[self.cell_index]

        elif 'brain_observatory_nwb_data_set' in kwargs and 'csid' in kwargs:

            self.data = kwargs['brain_observatory_nwb_data_set']
            self.csid = kwargs['csid']
            self.oeid = ROI.get_metadata(self.data)
            self.cell_index = ROI.get_all_cell_specimen_indices_dict(self.data)[self.csid]

        else:
            raise RuntimeError('Construction not recognized')

        self.mask = ROI.get_roi_mask_array(self.data, self.oeid)[self.cell_index, :, :]
        self.mask_sparse = map(np.array, np.where(self.mask == 1))
        self.center = tuple(map(lambda x: round(x,1), (self.mask_sparse[0].mean(), self.mask_sparse[1].mean())))

    @staticmethod
    @cacheable(query_strategy='lazy', **get_cache_array_sparse_h5_reader_writer())
    def get_roi_mask_array_cache(brain_observatory_nwb_data_set):
        return brain_observatory_nwb_data_set.get_roi_mask_array()


    @staticmethod
    @memoize
    def get_roi_mask_array(brain_observatory_nwb_data_set, oeid):
        return ROI.get_roi_mask_array_cache(brain_observatory_nwb_data_set, path=os.path.join(cache_location, str(oeid), 'roi_mask.h5'))

    @staticmethod
    @memoize
    def get_metadata(brain_observatory_nwb_data_set):
        return brain_observatory_nwb_data_set.get_metadata()['ophys_experiment_id']

    @staticmethod
    @memoize
    def get_cell_specimen_ids(brain_observatory_nwb_data_set):
        return brain_observatory_nwb_data_set.get_cell_specimen_ids()

    @staticmethod
    @memoize
    def get_all_cell_specimen_indices_dict(brain_observatory_nwb_data_set):
        id_list = ROI.get_cell_specimen_ids(brain_observatory_nwb_data_set)
        index_list = brain_observatory_nwb_data_set.get_cell_specimen_indices(ROI.get_cell_specimen_ids(brain_observatory_nwb_data_set))
        return dict((id, index) for id, index in zip(id_list, index_list))

    @property
    def size(self):
        return len(self.mask_sparse[0])

    @property
    def df(self):

        df = pd.DataFrame({'left':self.mask_sparse[0],
                           'right': self.mask_sparse[0]+1,
                           'bottom': self.mask_sparse[1],
                           'top': self.mask_sparse[1] + 1,
                             })
        df['csid'] = self.csid
        df['cell_index'] = self.cell_index
        df['center'] = [self.center]*len(df)
        df['size'] = [self.size] * len(df)

        # print self.center, df['left'].min(), df['right'].max(), df['bottom'].min(), df['top'].max()

        return df

    def get_covering_box_boundaries(self):

        l = self.mask_sparse[0].min()
        r = self.mask_sparse[0].max() + 1
        b = self.mask_sparse[1].min()
        t = self.mask_sparse[1].max() + 1

        return l, r, b, t

    def get_x_y_border_list(self):

        # t0 = time.time()

        df = self.df
        # print 'df_time', time.time() - t0
        # t0 = time.time()
        # df = pd.DataFrame({'left':[0], 'bottom':[0]})
        # df = pd.DataFrame({'left': [0, 0], 'bottom': [0, 1]})
        # df['top'] = df['bottom'] + 1
        # df['right'] = df['left'] + 1

        # def is_left_turn()

        # import matplotlib.pyplot as plt
        # img = np.zeros((300,300))
        edge_list = []
        lookup_dict = {}
        for left, right, bottom, top in zip(df.left.values, df.right.values, df.bottom.values, df.top.values):
            # img[bottom, left] = 1
            ll = ((left, bottom), (left, top))
            bb = ((left, top), (right, top))
            rr = ((right, top), (right, bottom))
            tt = ((right, bottom), (left, bottom))

            for x in [ll,bb,rr,tt]:
                edge_list.append(x)
                lookup_dict[(x[1],x[0])] = True

        # plt.imshow(img, interpolation='none')


        # print '    A', time.time() - t0
        # t0 = time.time()


        new_edge_list = []
        for x in edge_list:
            if not lookup_dict.get(x):
                new_edge_list.append(x)

        # print '    B', time.time() - t0
        # t0 = time.time()

        point_dict_start = collections.defaultdict(list)
        for x in new_edge_list:
            # assert not x[0] in point_dict_start
            point_dict_start[x[0]].append(x)

        # print '    C', time.time() - t0
        # t0 = time.time()

        def left_turn_landing_point(previous_section):
            # print previous_section
            if previous_section[0][0] == previous_section[1][0] and previous_section[0][1] > previous_section[1][1]:
                return (previous_section[1][0]+1, previous_section[1][1])
            elif previous_section[0][0] == previous_section[1][0] and previous_section[0][1] < previous_section[1][1]:
                return (previous_section[1][0]-1, previous_section[1][1])
            elif previous_section[0][0] < previous_section[1][0] and previous_section[0][1] == previous_section[1][1]:
                return (previous_section[1][0], previous_section[1][1] + 1)
            elif previous_section[0][0] > previous_section[1][0] and previous_section[0][1] == previous_section[1][1]:
                return (previous_section[1][0], previous_section[1][1] - 1)
            else:
                raise Exception

        curr_line = [new_edge_list[0]]
        while curr_line[0][0] != curr_line[-1][1]:
            if len(point_dict_start[curr_line[-1][1]]) == 1:
                curr_line.append(point_dict_start[curr_line[-1][1]][0])
            else:
                for curr_segment in point_dict_start[curr_line[-1][1]]:
                    if curr_segment[1] == left_turn_landing_point(curr_line[-1]):
                        curr_line.append(curr_segment)
                        break

        assert len(curr_line) == len(new_edge_list)


        # print '    D', time.time() - t0
        # t0 = time.time()


        x_list, y_list = [],[]
        for curr_segment in curr_line:
            x_list.append(curr_segment[0][0])
            y_list.append(curr_segment[0][1])

        # print '    E', time.time() - t0

        return x_list, y_list







if __name__ == "__main__":

    # oeid=530646083
    # manifest_file = os.path.join(cache_location, 'boc_manifest.json')
    # brain_observatory_cache = BrainObservatoryCache(manifest_file=manifest_file)
    # brain_observatory_nwb_data_set = brain_observatory_cache.get_ophys_experiment_data(oeid)
    #
    # for cell_index in range(200):
    #     print cell_index
    #     roi = ROI(brain_observatory_nwb_data_set=brain_observatory_nwb_data_set, cell_index=cell_index)

    roi = ROI(cell_index=56, oeid=530646083)
    # roi = ROI(csid=541095890, oeid=530646083)

    x, y = roi.get_x_y_border_list()

    # for xi, yi in zip(x, y):
    #     print xi, yi

    # print df.head()
    # print len(df)
    # import collections
    #
    #
    # import matplotlib.pyplot as plt
    #
    # img = np.zeros((200,200))
    #
    # edge_list = []
    # # edge_dict = {}
    # for _, row in df.iterrows():
    #     img[row.left, row.bottom] = 1
    #
    #     ll = ((row.left,row.bottom), (row.left, row.top))
    #     bb = ((row.left, row.top), (row.right, row.top))
    #     rr = ((row.right,row.top), (row.right, row.bottom))
    #     tt = ((row.right, row.bottom), (row.left, row.bottom))
    #
    #     # edge_dict[(), ()] = x
    #
    #     edge_list.append(ll)
    #     edge_list.append(rr)
    #     edge_list.append(tt)
    #     edge_list.append(bb)
    #
    # new_edge_list = []
    # for x in edge_list:
    #     if not (x[1], x[0]) in edge_list:
    #         new_edge_list.append(x)
    #
    #
    # # cc = collections.Counter(edge_list)
    # # # tmp = []
    # point_dict_start = {}
    # for x in new_edge_list:
    #
    #         assert not x[0] in point_dict_start
    #         point_dict_start[x[0]] = x
    #
    # curr_line = [new_edge_list[0]]
    # print curr_line
    # while curr_line[0][0] != curr_line[-1][1]:
    #     curr_line.append(point_dict_start[curr_line[-1][1]])
    # assert len(curr_line) == len(new_edge_list)
    # for x in curr_line:
    #     print x

    # for x in new_edge_list:

    #
    # for x in tmp:
    #     print x
    #
    # print len(x)

    # for x in edge_list:
    #     print x

    # plt.imshow(img)
    # plt.show()



    # csid=541095890
