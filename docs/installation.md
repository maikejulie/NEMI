# How to install NEMI

These instructions are **goal-oriented**: you already know why you want the package; they skip background on the method (see [About NEMI](about.md) for that).

## Install with pip (recommended)

**Goal:** use the published package in any project.

### Prerequisites

- Python 3.9 or newer

### Steps

1. Create and activate a virtual environment (recommended). Any tool you prefer is fine (for example [venv](https://docs.python.org/3/library/venv.html) or [Mamba](https://mamba.readthedocs.io/)).
2. Install:

```bash
pip install nemi-learn
```

3. Confirm import:

```bash
python -c "from nemi import NEMI; print(NEMI)"
```

## Install from source

**Goal:** develop NEMI or change the embedding/clustering steps in code.

### Steps

1. Clone the repository:

```bash
git clone https://github.com/maikejulie/NEMI.git
cd NEMI
```

2. Create and activate a virtual environment (recommended).
3. Install in editable mode:

```bash
pip install -e .
```

Optional: install extras for running the test suite:

```bash
pip install -e ".[full]"
```

## Build and preview the documentation

**Goal:** render this site locally (MkDocs Material).

From the repository root, with your environment active:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open the URL shown in the terminal (usually `http://127.0.0.1:8000`). To write a static site to `site/`:

```bash
mkdocs build
```

## After installation

- New to the library: follow [Tutorial: your first NEMI analysis](quickstart.md).
- API details: [API reference](reference.md).
