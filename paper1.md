---
title: 'xTorch: A High-Level C++ Extension Library for PyTorch (LibTorch)'
tags:

- C++
- machine learning
- deep learning
- libtorch
- PyTorch
- AI
- training authors:
- name: Kamal Saberi orcid: 0000-0000-0000-0000 affiliation: 1 affiliations:
- name: Independent Developer index: 1 date: 2025-04-04
  software_repository_url: https://github.com/kamisaberi/libtorch-extended
  archive_url: https://zenodo.org/  # Replace this with Zenodo DOI once archived
  paper_url: https://github.com/kamisaberi/libtorch-extended/blob/main/paper.md

---

# Summary

**xTorch** is a high-level extension to PyTorch’s C++ API (LibTorch) that simplifies model development, training,
evaluation, and deployment workflows. While LibTorch offers powerful low-level access to PyTorch’s engine, many
high-level utilities available in the Python API were deprecated or removed after 2019. xTorch bridges this usability
gap by providing an organized set of abstractions for defining neural networks, managing data, training loops, device
management, and serialization.

xTorch enhances developer productivity by reintroducing a “batteries-included” ethos for C++, similar to PyTorch’s
Python experience. It is modular, extensible, and built entirely on top of LibTorch without modifying the core, ensuring
compatibility and performance. It supports tasks such as CNN model training, JIT export, and inference pipelines — all
from native C++ code.

# Statement of Need

C++ remains a critical language for high-performance machine learning systems, robotics, embedded applications, and
large-scale deployment. However, PyTorch’s C++ frontend (LibTorch) is difficult to use on its own due to the lack of
high-level APIs, forcing users to write verbose and repetitive code.

xTorch was created to fill this gap by wrapping LibTorch with practical utilities such as `Trainer`, `XTModule`
, `DataLoader`, and `export_to_jit()`. These abstractions drastically reduce boilerplate, increase accessibility, and
allow developers to build, train, and deploy models entirely in C++. Unlike other frameworks that require switching to
Python or writing extensive C++ glue code, xTorch makes the entire ML workflow intuitive and modular in C++.

# Functionality

xTorch provides:

- High-level neural network module definitions
- Trainer class for managing training, loss, metrics, and callbacks
- Data loaders with support for image datasets, CSVs, and transformations
- Model serialization and JIT export helpers
- Support for CPU and CUDA backends (via LibTorch)
- Clean separation of layers: Models, Data, Utils, and Training
- Compatibility with the PyTorch ecosystem

# Example Use

```cpp
auto trainData = xt::datasets::ImageFolder("data/train", xt::transforms::Compose({ ... }));
auto model = xt::models::ResNet18(10);
xt::Trainer trainer;
trainer.setMaxEpochs(20).fit(model, trainLoader, valLoader);
xt::export_to_jit(model, "model.pt");
```

# Acknowledgements

The xTorch project builds upon the PyTorch (LibTorch) C++ API. Thanks to the open-source contributors to PyTorch for
enabling access to their high-performance machine learning framework via C++.

# References

- PyTorch C++ API Documentation: https://pytorch.org/cppdocs/
- TorchScript for Deployment: https://pytorch.org/tutorials/advanced/cpp_export.html
