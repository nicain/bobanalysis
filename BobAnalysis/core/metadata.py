from BobAnalysis.core.sessionstimulus import SessionStimulus
from BobAnalysis.core.sessiondff import SessionDFF
from BobAnalysis.core.roi import ROI
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from allensdk.api.cache import cacheable, Cache
from BobAnalysis import cache_location
from allensdk.api.cache import cacheable, Cache
from BobAnalysis.core.utilities import memoize
import os


@cacheable(strategy='lazy', **Cache.cache_json())
def get_csid_to_oeid_dict():
    csid_oeid_dict = {}
    manifest_file = os.path.join(cache_location, 'boc_manifest.json')
    boc = BrainObservatoryCache(manifest_file=manifest_file)
    oeid_list =[expt['id'] for expt in boc.get_ophys_experiments(session_types=['three_session_C'])]
    for oeid in oeid_list:
        data = boc.get_ophys_experiment_data(oeid)
        curr_csid_list = data.get_cell_specimen_ids()
        for curr_csid in zip(curr_csid_list):
            csid_oeid_dict[str(curr_csid[0])] = oeid
    return csid_oeid_dict

csid_oeid_dict = get_csid_to_oeid_dict(path=os.path.join(cache_location, 'csid_oeid_dict.json'))
def csid_to_oeid(csid):
    return csid_oeid_dict[str(csid)]

@cacheable(strategy='lazy', **Cache.cache_json())
def get_oeid_index_to_csid_dict():
    oeid_index_to_csid_dict = {}
    manifest_file = os.path.join(cache_location, 'boc_manifest.json')
    boc = BrainObservatoryCache(manifest_file=manifest_file)
    oeid_list =[expt['id'] for expt in boc.get_ophys_experiments(session_types=['three_session_C'])]
    for oeid in oeid_list:
        data = boc.get_ophys_experiment_data(oeid)
        curr_csid_list = data.get_cell_specimen_ids()
        curr_cell_index_list = data.get_cell_specimen_indices(curr_csid_list)
        for curr_csid, curr_cell_index in zip(curr_csid_list, curr_cell_index_list):
            oeid_index_to_csid_dict[str((oeid, curr_cell_index))] = curr_csid

    return oeid_index_to_csid_dict

oeid_index_to_csid_dict = get_oeid_index_to_csid_dict(path=os.path.join(cache_location, 'oeid_index_to_csid_dict.json'))
def oeid_index_to_csid(oeid, index):
    return oeid_index_to_csid_dict[str((oeid, index))]

@cacheable(strategy='lazy', **Cache.cache_json())
def get_oeid_csid_to_index_dict():
    oeid_csid_to_index_dict = {}
    manifest_file = os.path.join(cache_location, 'boc_manifest.json')
    boc = BrainObservatoryCache(manifest_file=manifest_file)
    oeid_list =[expt['id'] for expt in boc.get_ophys_experiments(session_types=['three_session_C'])]
    for oeid in oeid_list:
        data = boc.get_ophys_experiment_data(oeid)
        curr_csid_list = data.get_cell_specimen_ids()
        curr_cell_index_list = data.get_cell_specimen_indices(curr_csid_list)
        for curr_csid, curr_cell_index in zip(curr_csid_list, curr_cell_index_list):
            oeid_csid_to_index_dict[str((oeid, curr_csid))] = curr_cell_index

    return oeid_csid_to_index_dict

oeid_csid_to_index_dict = get_oeid_csid_to_index_dict(path=os.path.join(cache_location, 'oeid_csid_to_index_dict.json'))
# print oeid_csid_to_index_dict
def oeid_csid_to_index(oeid, csid):
    return oeid_csid_to_index_dict[str((oeid, csid))]




if __name__ == '__main__':
    print csid_to_oeid(517504349)
    print oeid_index_to_csid(510536157, 15)
    print oeid_csid_to_index(510536157, 517504593)

    import  pandas as pd
    boc = BrainObservatoryCache(manifest_file = os.path.join(cache_location, 'boc_manifest.json'))
    # Download cells for a set of experiments and convert to DataFrame
    cells = boc.get_cell_specimens()
    cells = pd.DataFrame.from_records(cells)
    cells = cells.rename(columns={'cell_specimen_id':'csid'})
    print cells


# oeid_list =[expt['id'] for expt in boc.get_ophys_experiments(session_types=['three_session_C'])]


# from LinearModelFit.cache import csid_oeid_dict_dict_location
# from LinearModelFit.cache import oeid_csid_dict_dict_location
# from LinearModelFit.cache import oeid_index_to_csid_dict_location
# from LinearModelFit.cache import oeid_csid_to_index_dict_location
# import pickle
# from LinearModelFit.core.utilities import get_boc_iwarehouse

# boc = get_boc_iwarehouse()


# csid_oeid_dict = {}
# oeid_csid_dict = {}
# oeid_index_to_csid_dict = {}
# oeid_csid_to_index_dict = {}
# for oeid in oeid_list:
#     # print oeid
#     data = boc.get_ophys_experiment_data(oeid)
#     curr_csid_list = data.get_cell_specimen_ids()
#     curr_cell_index_list = data.get_cell_specimen_indices(curr_csid_list)
#     oeid_csid_dict[oeid] = curr_csid_list
#     for curr_csid, curr_cell_index in zip(curr_csid_list, curr_cell_index_list):
#         csid_oeid_dict[curr_csid] = oeid
#         oeid_index_to_csid_dict[oeid, curr_cell_index] = curr_csid
#         oeid_csid_to_index_dict[oeid, curr_csid] = curr_cell_index

# pickle.dump(csid_oeid_dict, open(csid_oeid_dict_dict_location, 'w'))
# pickle.dump(oeid_csid_dict, open(oeid_csid_dict_dict_location, 'w'))
# pickle.dump(oeid_index_to_csid_dict, open(oeid_index_to_csid_dict_location, 'w'))
# pickle.dump(oeid_csid_to_index_dict, open(oeid_csid_to_index_dict_location, 'w'))

# csid = 517527963
# oeid = csid_oeid_dict[csid]
# expt = boc.get_ophys_experiments(ids=[oeid])
# print expt