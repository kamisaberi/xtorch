---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
  - name: Lyman Spitzer, Jr. Fellow, Princeton University, United States
    index: 1
    ror: 00hx57361
  - name: Institution Name, Country
    index: 2
  - name: Independent Researcher, Country
    index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

xTorch is a modular, high-level C++ extension to the PyTorch C++ frontend (LibTorch). It significantly simplifies the
process of model definition, training, data loading, and deployment by introducing practical abstractions missing in
LibTorch after 2019. While LibTorch exposes the full power of the PyTorch backend, it lacks user-friendly APIs available
in Python. xTorch fills this usability gap and enables developers to create deep learning pipelines directly in C++ with
ease and clarity.

xTorch introduces structured components for neural network architecture (XTModule), training management (Trainer),
dataset handling (ImageFolder, CSVDataset, and transformations), model checkpointing, and TorchScript export. It
enhances developer productivity without sacrificing the performance and flexibility of LibTorch.

# Statement of need

Although Python remains the primary language for machine learning, many real-world systems are written in C++ due to its
speed, compatibility, and suitability for embedded, robotics, game engines, and high-performance environments. PyTorch’s
C++ frontend (LibTorch) enables such integration, but lacks many high-level abstractions that make Python so productive.

xTorch addresses this gap by wrapping LibTorch’s low-level tools into a modular and extensible framework that is more
intuitive to use. With xTorch, C++ developers no longer need to reinvent training loops or manually register every
neural module. Instead, they can prototype, train, evaluate, and export models using a syntax close to PyTorch in
Python. This drastically reduces development time, increases adoption of C++ for ML, and aligns the C++ experience with
modern deep learning workflows.

# Citations

If citing relevant work in the paper (optional but encouraged), you can include citations like:

Paszke et al., 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

PyTorch C++ API Documentation. https://pytorch.org/cppdocs/

TorchScript export guide: https://pytorch.org/tutorials/advanced/cpp_export.html

Let me know if you want these turned into BibTeX or @citation markdown style.

[//]: # (# Figures)

[//]: # ()
[//]: # (Figures can be included like this:)

[//]: # (![Caption for example figure.\label{fig:example}]&#40;figure.png&#41;)

[//]: # (and referenced from text using \autoref{fig:example}.)

[//]: # ()
[//]: # (Figure sizes can be customized by adding an optional second parameter:)

[//]: # (![Caption for example figure.]&#40;figure.png&#41;{ width=20% })

# Acknowledgements

This project builds upon the open-source work of the PyTorch community. Special thanks to the developers of LibTorch,
whose computational backend powers this library. The xTorch project also thanks the broader machine learning and C++
community for providing open tools and knowledge that make development and research more accessible.

# References

[1] Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

[2] PyTorch C++ API Documentation: https://pytorch.org/cppdocs/

[3] TorchScript Export Tutorial: https://pytorch.org/tutorials/advanced/cpp_export.html
