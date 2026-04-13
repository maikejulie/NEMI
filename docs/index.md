# Welcome to NEMI

Native Emergent Manifold Interrogation (NEMI) is a method to determine regions of interest in large or highly complex and nonlinear data.

Within the Earth sciences data is increasingly becoming unmanageably large, noisy and nonlinear.
Most methods that are commonly in use employ highly restrictive assumptions regarding the underlying
statistics of the data and may even offer misleading results. To enable and accelerate scientific
discovery, we draw on tools from computer science, statistics and dynamical systems theory to develop
the NEMI method. NEMI is intended for wide use within the Earth
sciences and applied to an oceanographic example here. Using domain specific theory, manifold representation
of the data, clustering and sophisticated ensembling, NEMI is able to highlight particularly interesting
areas within the data. The associated paper stresses the underlying philosophy and appreciation of methods to
facilitate understanding of data mining; a tool to gain new knowledge.

## What is new with NEMI

NEMI is a generalisation of the methodology in [Sonnewald et al. (2020)](https://www.science.org/doi/10.1126/sciadv.aay4740) that targeted plankton ecosystems,
in that it is designed to scale to larger datasets and is agnostic to the source of the data. Scaling is one of the true bottlenecks in data mining for scientific applications. NEMI is generalised to work with any data,
where the particular example application used here is geospatial data. NEMI uses an explicitly hierarchical approach, making NEMI less parametric (fewer parameters to tune and less danger of noise interference) and
intuitively useful both for global (for example the whole Earth in the present example) or more local applications (for example a basin or more regional assessment). Another novelty in NEMI is the lack of a fixed
field-specific benchmark criteria (used in [Sonnewald et al. (2020)](https://www.science.org/doi/10.1126/sciadv.aay4740)), where NEMI has generalised so a field agnostic option is available.
Lastly, NEMI invites the use of a range of uncertainty quantification options in the final cluster evaluation.
