# NEMI documentation

**Native Emergent Manifold Interrogation (NEMI)** is a method for finding regions of interest in large, noisy, nonlinear data—especially in the Earth sciences. It combines manifold learning, clustering, and ensembling to highlight structure without relying on narrow statistical assumptions.

## How this documentation is organized

The docs follow the [Diataxis](https://diataxis.fr/) idea: separate **learning**, **tasks**, **facts**, and **background** so you can open the right page for what you need now.

| You want to… | Start here |
|--------------|------------|
| **Learn** by doing a minimal, end-to-end run | [Tutorial: first analysis](quickstart.md) |
| **Do** something specific (install, build docs) | [Installation](installation.md) |
| **Understand** the method and workflow | [About NEMI](about.md) |
| **Look up** classes, functions, and parameters | [API reference](reference.md) |

## What is new in this version

NEMI generalises the approach in [Sonnewald et al. (2020)](https://www.science.org/doi/10.1126/sciadv.aay4740) (plankton ecosystems) to **larger datasets** and **arbitrary data sources**, with a **hierarchical** pipeline (fewer tuning knobs, useful from global to regional scales). It drops fixed field-specific benchmarks in favour of a **field-agnostic** option and supports **several ways to quantify uncertainty** in the final cluster assignment. For the full conceptual picture, see [About NEMI](about.md).
