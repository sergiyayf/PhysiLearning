import zmq
from physilearning.subscriber import PhysiCellDataListener
import time

def test_construct_pcdl():
    pcdl = PhysiCellDataListener(port=5556)
    assert pcdl.port == 5556

