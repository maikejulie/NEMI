'''
Test basic functionality of the method
'''

import pytest
import numpy as np
from nemi import NEMI

def test_micro_nemi_pack():
    '''
    Run the method on noise for an ensemble
    '''

    X = np.random.random((100,5))

    nemi  = NEMI()
    nemi.run(X, n=3)
    nemi.plot('clusters')

    

def test_micro_nemi():
    '''
    Run the method on noise for single member
    '''

    X = np.random.random((100,5))

    nemi  = NEMI()
    nemi.run(X, n=1)
    nemi.plot('clusters')
