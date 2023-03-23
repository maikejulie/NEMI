====
NEMI
====

About NEMI
==========

The Native Emergent Manifold Interrogation (NEMI; submitted JAMES) is a method to determine regions of interest in large or highly complex and nonlinear data. 

Citation: Sonnewald, M., submitted. A hierarchical ensemble manifold methodology for new knowledge on spatial data: An application to ocean physics. Journal of Advances in Modeling Earth Systems.

Short description/abstract:
---------------------------

Algorithms to determine regions of interest in large or highly complex and nonlinear data is becoming increasingly important. 
Novel methodologies from computer science and dynamical systems are well placed as analysis tools, but are underdeveloped for 
applications within the Earth sciences, and many produce misleading results.  I present a novel and general workflow, the Native Emergent Manifold Interrogation (NEMI) method, which is easy to use and widely applicable. 
NEMI is able to quantify and leverage the highly 
complex latent space presented by noisy, nonlinear and unbalanced data common in the Earth sciences. 
NEMI uses dynamical systems and probability theory to strengthen associations, simplifying covariance structures, 
within the data with a manifold, or a Riemannian, methodology that uses domain specific charting of the underlying space. 
On the manifold, an agglomerative clustering methodology is applied to isolate the now observable  areas of interest. The 
construction of the manifold introduces a stochastic component which is beneficial to the analysis as it enables latent space 
regularization. NEMI uses an ensemble methodology to quantify the sensitivity of the results noise. The areas of interest, or clusters, 
are sorted within individual ensemble members and co-located across the set. A metric such as a majority vote, entropy, or similar the 
quantifies if a data point within the original data belongs to a certain cluster. NEMI is clustering method agnostic, but the use of an 
agglomerative methodology and sorting in the described case study allows a filtering, or nesting, of clusters to tailor to a desired application.


.. image:: https://github.com/maikejulie/NEMI/raw/3bb2d5b090069e16685ae3d87d74856b5ac49760/docs/images/NEMI_sketch.png
    :width: 600px
    :alt: NEMI workflow
    :align: center

Figure: Sketch of NEMI workflow. Part 1 (top row) illustrates moving from the data in its rew form, through initial symbolic renditioning, manifold transformation and clustering. Part 2 (bottom row) shows the ensembling, agglomerative utility ranking and native (field specific) utility ranking within each ensemble member. Finally, the cluster for each location is determined looking across the ensemble. (Top left image of model adapted from encyclopedie-environnement.org).

Plain Language Summary:
-----------------------
Within the Earth sciences data is increasingly becoming unmanageably large, noisy and nonlinear. 
Most methods that are commonly in use employ highly restrictive assumptions regarding the underlying 
statistics of the data and may even offer misleading results. To enable and accelerate scientific 
discovery, I drew on tools from computer science, statistics and dynamical systems theory to develop 
the Native Emergent Manifold Interrogation (NEMI) method. Nemi is intended for wide use within the Earth 
sciences and applied to an oceanographic example here. Using domain specific theory, manifold representation 
of the data, clustering and sophisticated ensembling, NEMI is able to highlight particularly interesting 
areas within the data. In the paper, I stresses the underlying philosophy and appreciation of methods to 
facilitate understanding of data mining; a tool to gain new knowledge.


What is new with NEMI:
----------------------
NEMI is a generalisation of the methodology in `Sonnewald et al. (2020) <https://www.science.org/doi/10.1126/sciadv.aay4740>`__ that targeted plankton ecosystems, 
in that is is designed to scale to larger datasets and is agnostic to the source of the data. Scaling is one of the true bottlenecks in data mining for scientific applications. NEMI is generalised to work with any data, 
where the particular example application used here is geospatial data. I have used an explicitly hierarchical approach, making NEMI less parametric (fewer parameters to tune and less danger of noise interference) and 
intuitively useful both for global (for example the whole Earth in the present example) or more local applications (for example a basin or more regional assessment). Another novelty in NEMI is the lack of a fixed 
field-specific benchmark criteria (used in \cite{Sonnewald2020}), where I have generalised so a field agnostic option is available. 
Lastly, NEMI invites the use of a range of uncertainty quantification options in the final cluster evaluation. 

Requirements
============
Python 3.7 or greater

We also recommend installing in a virtual environment. For more information see documentation for e.g., `Mamba <https://mamba.readthedocs.io/en/latest/>`__.

Quick start guide
=================

Install with ``pip install nemi-learn``. Given an array X with dimensions (n_samples, n_features), these Python commands will run the NEMI workflow and bring up a plot::

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

