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


def test_micro_nemi_dbscan():
    """
    Run the method on noise for single member using DBSCAN clustering
    """
    params = dict(
        embedding_dict=dict(min_dist=0.0, n_components=3, n_neighbors=20),
        clustering_dict=dict(method='dbscan', eps=0.5, min_samples=5, metric='euclidean')
    )

    X = np.random.random((100, 5))

    nemi = NEMI(params)
    nemi.run(X, n=1)
    nemi.plot('clusters')
