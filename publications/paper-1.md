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

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

[//]: # (# Mathematics)

[//]: # ()
[//]: # (Single dollars &#40;$&#41; are required for inline mathematics e.g. $f&#40;x&#41; = e^{\pi/x}$)

[//]: # ()
[//]: # (Double dollars make self-standing equations:)

[//]: # ()
[//]: # ($$\Theta&#40;x&#41; = \left\{\begin{array}{l})

[//]: # (0\textrm{ if } x < 0\cr)

[//]: # (1\textrm{ else})

[//]: # (\end{array}\right.$$)

[//]: # ()
[//]: # (You can also use plain \LaTeX for equations)

[//]: # (\begin{equation}\label{eq:fourier})

[//]: # (\hat f&#40;\omega&#41; = \int_{-\infty}^{\infty} f&#40;x&#41; e^{i\omega x} dx)

[//]: # (\end{equation})

[//]: # (and refer to \autoref{eq:fourier} from text.)

[//]: # (# Citations)

[//]: # ()
[//]: # (Citations to entries in paper.bib should be in)

[//]: # ([rMarkdown]&#40;http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html&#41;)

[//]: # (format.)

[//]: # ()
[//]: # (If you want to cite a software repository URL &#40;e.g. something on GitHub without a preferred)

[//]: # (citation&#41; then you can do it with the example BibTeX entry below for @fidgit.)

[//]: # ()
[//]: # (For a quick reference, the following citation commands can be used:)

[//]: # (- `@author:2001`  ->  "Author et al. &#40;2001&#41;")

[//]: # (- `[@author:2001]` -> "&#40;Author et al., 2001&#41;")

[//]: # (- `[@author1:2001; @author2:2001]` -> "&#40;Author1 et al., 2001; Author2 et al., 2002&#41;")

[//]: # (# Figures)

[//]: # ()
[//]: # (Figures can be included like this:)

[//]: # (![Caption for example figure.\label{fig:example}]&#40;figure.png&#41;)

[//]: # (and referenced from text using \autoref{fig:example}.)

[//]: # ()
[//]: # (Figure sizes can be customized by adding an optional second parameter:)

[//]: # (![Caption for example figure.]&#40;figure.png&#41;{ width=20% })

## Functionality

xTorch provides:

- High-level neural network module definitions (e.g., XTModule, ResNetExtended, XTCNN)
- A simplified training loop with the Trainer class, handling loss computation, metrics, and callbacks
- Enhanced data handling with ImageFolderDataset, CSVDataset, and OpenCV-backed transformations
- Utility functions for logging, metrics computation, and device management
- Extended optimizers like AdamW, RAdam, and learning rate schedulers
- Model serialization and TorchScript export helpers (save_model(), export_to_jit())
- Inference utilities for loading models and making predictions

The library is modular and extensible, built on top of LibTorch, and supports both CPU and CUDA devices.



## Example Use

```cpp
// Example: CNN Training Pipeline
auto trainData = xt::datasets::ImageFolder("data/train", xt::transforms::Compose({
    xt::transforms::Resize({224, 224}),
    xt::transforms::ToTensor(),
    xt::transforms::Normalize({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5})
}));

auto trainLoader = xt::data::DataLoader(trainData, 64, true);

auto model = xt::models::ResNet18(10);
auto optimizer = xt::optim::Adam(model.parameters(), 1e-3);
auto criterion = xt::loss::CrossEntropyLoss();

xt::Trainer trainer;
trainer.setMaxEpochs(20)
       .setOptimizer(optimizer)
       .setCriterion(criterion)
       .fit(model, trainLoader);

// Export model to TorchScript
xt::export_to_jit(model, "model.pt");
```



# Acknowledgements

The xTorch project builds upon the PyTorch (LibTorch) C++ API. Thanks to the open-source contributors to PyTorch for enabling access to their high-performance machine learning framework via C++.

# References
- PyTorch C++ API Documentation: https://pytorch.org/cppdocs/
- TorchScript for Deployment: https://pytorch.org/tutorials/advanced/cpp_export.html











---
title: "xTorch: A High-Level C++ Extension Library for PyTorch (LibTorch)"
authors:
- name: "Kamran Saberifard"
  affiliation: ""
---
## Tags

## Summary

PyTorch’s C++ library (LibTorch) emerged as a powerful way to use PyTorch outside Python, but after 2019 it became challenging for developers to use it for end-to-end model development. Early on, LibTorch aimed to mirror the high-level Python API, yet many convenient abstractions and examples never fully materialized or were later removed.
As of 2020, the C++ API had achieved near feature-parity with Python’s core operations, but it lagged in usability and community support. Fewer contributors focused on C++ meant that only low-level building blocks were provided, with high-level components (e.g. ready-made network architectures, datasets) largely absent. This left C++ practitioners to rewrite common tools from scratch – implementing standard models or data loaders manually – which is time-consuming and error-prone.
Another factor was PyTorch’s emphasis on the Python-to-C++ workflow. The official recommended path for production was to prototype in Python, then convert models to TorchScript for C++ deployment. This approach deprioritized making the pure C++ experience as friendly as Python’s.
As a result, developers who preferred or needed to work in C++ (for integration with existing systems, performance, or deployment constraints) found LibTorch cumbersome. Simple tasks like data augmentation (e.g. random crops or flips) had no built-in support in LibTorch C++. Defining neural network modules in C++ involved boilerplate macros and manual registration, an awkward process compared to Python’s concise syntax. Crucial functionality for model serialization was limited – for instance, LibTorch could load Python-exported models but not easily export its own models to a portable format.
xTorch was created to address this gap. It is a C++ library that extends LibTorch with the high-level abstractions and utilities that were missing or removed after 2019. By building on LibTorch’s robust computational core, xTorch restores ease-of-use without sacrificing performance. The motivation is to empower C++ developers with a productive experience similar to PyTorch in Python – enabling them to build, train, and deploy models with minimal fuss. In essence, xTorch revives and modernizes the “batteries-included” ethos for C++ deep learning, providing an all-in-one toolkit where the base library left off.

## Statement of Need

xTorch addresses the lack of high-level APIs in LibTorch for C++ developers, which is critical for high-performance machine learning, robotics, embedded applications, and large-scale deployment scenarios. By reintroducing high-level utilities that were deprecated in the Python API post-2019, xTorch enables C++ developers to build, train, evaluate, and deploy models more intuitively and efficiently.

C++ remains a critical language for high-performance machine learning systems, robotics, embedded applications, and large-scale deployment. However, PyTorch’s C++ frontend (LibTorch) is difficult to use on its own due to the lack of high-level APIs, forcing users to write verbose and repetitive code.

xTorch was created to fill this gap by wrapping LibTorch with practical utilities such as `Trainer`, `XTModule`, `DataLoader`, and `export_to_jit()`. These abstractions drastically reduce boilerplate, increase accessibility, and allow developers to build, train, and deploy models entirely in C++. Unlike other frameworks that require switching to Python or writing extensive C++ glue code, xTorch makes the entire ML workflow intuitive and modular in C++.


