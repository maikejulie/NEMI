# NEMI documentation

[![GitHub](https://img.shields.io/badge/GitHub-maikejulie%2FNEMI-blue.svg?style=flat)](https://github.com/maikejulie/NEMI)
[![License](https://img.shields.io/github/license/maikejulie/NEMI)](https://github.com/maikejulie/NEMI/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/nemi-learn.svg?style=flat)](https://badge.fury.io/py/nemi-learn)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7764719.svg)](https://doi.org/10.5281/zenodo.7764719)

**Native Emergent Manifold Interrogation (NEMI)** is a method for finding regions of interest in large, noisy, nonlinear data, especially in the Earth sciences. It combines manifold learning, clustering, and ensembling to highlight structure without relying on narrow statistical assumptions.

## How this documentation is organized

The docs follow the [Diataxis](https://diataxis.fr/) idea: separate **learning**, **tasks**, **facts**, and **background** so you can open the right page for what you need now.

| You want to... | Start here |
|--------------|------------|
| **Do** something specific (install, build docs) | [Installation](docs/installation.md) |
| **Learn** by doing a minimal, end-to-end run | [Tutorial: first analysis](docs/quickstart.md) |
| **Understand** the method and workflow | [About NEMI](docs/about.md) |
| **Look up** classes, functions, and parameters | [API reference](docs/reference.md) |

## What is new in this version

NEMI generalises the approach in [Sonnewald et al. (2020)](https://www.science.org/doi/10.1126/sciadv.aay4740) (plankton ecosystems) to **larger datasets** and **arbitrary data sources**, with a **hierarchical** pipeline (fewer tuning knobs, useful from global to regional scales). It drops fixed field-specific benchmarks in favour of a **field-agnostic** option and supports **several ways to quantify uncertainty** in the final cluster assignment. For the full conceptual picture, see [About NEMI](docs/about.md).
