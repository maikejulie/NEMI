====
NEMI
====

About NEMI
==========

Native Emergent Manifold Interrogation (NEMI) is a workflow to determine regions of interest in large or highly complex and nonlinear data. 

Algorithms to determine regions of interest in large or highly complex and nonlinear data is becoming increasingly important. 
Novel methodologies from computer science and dynamical systems are well placed as analysis tools, but are underdeveloped for 
applications within the Earth sciences, and many produce misleading results.  NEMI is able to quantify and leverage the highly 
complex latent space presented by noisy, nonlinear and unbalanced data common in the Earth sciences. 
NEMI uses dynamical systems and probability theory to strengthen associations, simplifying covariance structures, 
within the data with a manifold, or a Riemannian, methodology that uses domain specific charting of the underlying space. 
On the manifold, an agglomerative clustering methodology is applied to isolate the now observable  areas of interest. The 
construction of the manifold introduces a stochastic component which is beneficial to the analysis as it enables latent space 
regularization. NEMI uses an ensemble methodology to quantify the sensitivity of the results noise. The areas of interest, or clusters, 
are sorted within individual ensemble members and co-located across the set. A metric such as a majority vote, entropy, or similar the 
quantifies if a data point within the original data belongs to a certain cluster. NEMI is clustering method agnostic, but the use of an 
agglomerative methodology and sorting in the described case study allows a filtering, or nesting, of clusters to tailor to a desired application.

Requirements
============
Python 3.7 or greater

We also recommend installing in a virtual environment. For more information see documentation for e.g., `Mamba <https://mamba.readthedocs.io/en/latest/>`__.

Quick start guide
=================

Given an array X with dimensions (n_samples, n_features), these Python commands will run the NEMI workflow and bring up a plot::

    from nemi import NEMI
    nemi = NEMI()
    nemi.run()
    nemi.plot('clusters')

Installation from source
========================

If you wish to install from the source code follow the steps below. This will allow you to e.g., personalize
the embedding or clustering steps in the pipeline.

1. Clone the repository

2. (optional) Create and activate your virtual environment

3. Navigate to the root of the repository and install::

    pip install -e .

