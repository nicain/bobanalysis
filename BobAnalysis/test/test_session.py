from BobAnalysis.core.session import Session
from BobAnalysis import cache_location
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os

def test_session_oeid():

    oeid = 530646083
    Session(oeid=oeid)

def test_session_data():

    oeid=530646083
    manifest_file = os.path.join(cache_location, 'boc_manifest.json')
    brain_observatory_cache = BrainObservatoryCache(manifest_file=manifest_file)
    brain_observatory_nwb_data_set = brain_observatory_cache.get_ophys_experiment_data(oeid)
    Session(brain_observatory_nwb_data_set=brain_observatory_nwb_data_set)

if __name__ == "__main__":

    test_session_oeid() # pragma: no cover
    test_session_data()  # pragma: no cover

