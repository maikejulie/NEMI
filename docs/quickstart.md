# Tutorial: your first NEMI analysis

This tutorial is **learning-oriented**. You will install nothing new beyond what [Installation](installation.md) already covered, run a complete pipeline on synthetic data, and see a plot. It assumes basic Python and NumPy.

## What you will achieve

By the end you will have:

- Run embedding and clustering on a numeric array `X` of shape `(n_samples, n_features)`.
- Produced cluster labels and a matplotlib visualization.

## Before you start

- Python 3.9+ and NEMI installed (`pip install nemi-learn` or editable install from the repo—see [Installation](installation.md)).
- A Python environment where you can run scripts or a notebook.

## Step 1: Create sample data

NEMI expects a 2D array: rows are samples (e.g. locations or observations), columns are features.

```python
import numpy as np

rng = np.random.default_rng(42)
n_samples, n_features = 200, 10
X = rng.standard_normal((n_samples, n_features))
```

This stand-in data is enough for the pipeline to run; in your own work you would replace `X` with real measurements.

## Step 2: Run the workflow

```python
from nemi import NEMI

nemi = NEMI()
nemi.run(X)
```

You should see short progress messages in the terminal (`Fitting the embedding`, then clustering and sorting). The first run may take a little time while libraries initialise.

## Step 3: Plot the clusters

```python
nemi.plot("clusters")
```

If you are in a script, show the figure explicitly:

```python
import matplotlib.pyplot as plt

nemi.plot("clusters")
plt.show()
```

In Jupyter or IPython, the figure often appears without `plt.show()`.

## What happened

- The data were embedded in a lower-dimensional space (see default settings in the [API reference](reference.md)).
- Clusters were found in that space and sorted by size.
- The plot overlays cluster assignments on the embedding.

For **why** this workflow is designed this way, read [About NEMI](about.md). For **changing parameters** or advanced use, use the [API reference](reference.md) after you are comfortable with this flow.

## Next steps

- Try your own `X` with the same three steps (`NEMI()`, `run`, `plot`).
- To run multiple stochastic runs and combine them, see `NEMI.run(X, n=...)` in the [API reference](reference.md).
