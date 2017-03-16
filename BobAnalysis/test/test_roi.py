from BobAnalysis.core.roi import ROI
from BobAnalysis import cache_location
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os

def test_roi_csid_oeid():

    roi = ROI(csid=541095385, oeid=530646083)
    assert roi.cell_index == 7

def test_roi_index_oeid():

    roi = ROI(cell_index=7, oeid=530646083)
    assert roi.csid == 541095385

def test_roi_index_data():

    oeid=530646083
    manifest_file = os.path.join(cache_location, 'boc_manifest.json')
    brain_observatory_cache = BrainObservatoryCache(manifest_file=manifest_file)
    brain_observatory_nwb_data_set = brain_observatory_cache.get_ophys_experiment_data(oeid)
    roi = ROI(brain_observatory_nwb_data_set=brain_observatory_nwb_data_set, cell_index=7)
    print
    assert roi.csid == 541095385


def test_roi_csid_data():

    oeid = 530646083
    manifest_file = os.path.join(cache_location, 'boc_manifest.json')
    brain_observatory_cache = BrainObservatoryCache(manifest_file=manifest_file)
    brain_observatory_nwb_data_set = brain_observatory_cache.get_ophys_experiment_data(oeid)
    roi = ROI(brain_observatory_nwb_data_set=brain_observatory_nwb_data_set, csid=541095385)
    assert roi.cell_index == 7

if __name__ == "__main__":
    test_roi_csid_oeid() # pragma: no cover
    test_roi_index_oeid()  # pragma: no cover
    test_roi_index_data() # pragma: no cover
    test_roi_csid_data() # pragma: no cover

