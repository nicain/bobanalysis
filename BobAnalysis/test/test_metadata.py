
def test_csid_to_oeid():

    from BobAnalysis.core.metadata import csid_to_oeid
    assert csid_to_oeid(517504349) == 510536157

def test_oeid_index_to_csid():

    from BobAnalysis.core.metadata import oeid_index_to_csid
    assert oeid_index_to_csid(530646083, 15) == 541097101

def test_oeid_csid_to_index():

    from BobAnalysis.core.metadata import oeid_csid_to_index
    assert oeid_csid_to_index(510536157, 517504593) == 15
    
if __name__ == '__main__':                                    # pragma: no cover
    test_csid_to_oeid()                                              # pragma: no cover
    test_oeid_index_to_csid()  # pragma: no cover
    test_oeid_csid_to_index()  # pragma: no cover