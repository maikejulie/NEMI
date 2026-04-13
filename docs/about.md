# About NEMI

This page is **explanation**: it describes how NEMI works and why it is structured that way—not step-by-step tasks. For a hands-on first run, use the [tutorial](quickstart.md). To install the package, see [Installation](installation.md).

---

Algorithms for finding regions of interest in large, highly nonlinear data are increasingly important. Methods from computer science and dynamical systems are promising, but many are underdeveloped for Earth-science applications or can mislead when assumptions are wrong. NEMI is designed to **quantify and use** the complex latent structure in noisy, nonlinear, imbalanced data typical of those domains.

NEMI uses dynamical systems and probability to **strengthen associations** and **simplify covariance structure** via a manifold (Riemannian) view, with **domain-specific charting** of the underlying space. On the manifold, **agglomerative clustering** isolates candidate regions of interest. The manifold construction introduces **stochasticity**, which helps **regularize** the latent representation. An **ensemble** of runs captures sensitivity to noise; clusters are aligned across members, and a rule such as **majority vote** or **entropy** decides membership in the combined result. The implementation is **clustering-algorithm agnostic**; the reference case uses agglomerative clustering and sorted clusters so results can be **nested or filtered** for the application.

<img src="https://github.com/maikejulie/NEMI/raw/3bb2d5b090069e16685ae3d87d74856b5ac49760/docs/images/NEMI_sketch.png" width="600" alt="Sketch of NEMI workflow" />

**Figure:** Sketch of the NEMI workflow. Part 1 (top row): raw data → symbolic representation → manifold embedding → clustering. Part 2 (bottom row): ensembling, agglomerative utility ranking, and field-specific utility ranking per ensemble member; then cluster assignment by comparing across the ensemble. (Top-left panel adapted from encyclopedie-environnement.org.)
