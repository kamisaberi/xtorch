# API Reference

Welcome to the xTorch API reference. This section provides detailed documentation on all the public namespaces, classes, and functions that make up the xTorch library. It is designed to be a technical resource for understanding the specific components available for building your deep learning models.

## Relationship with LibTorch

xTorch is built as a high-level extension of LibTorch (the PyTorch C++ API). Its goal is to provide usability and convenience, not to replace the core engine.

While this reference covers all components added by **xTorch**, it does not duplicate the documentation for the underlying PyTorch C++ library. For fundamental operations involving `torch::Tensor`, `torch::autograd`, standard optimizers, and the base `torch::nn` modules (like `torch::nn::Linear`, `torch::nn::Conv2d`, `torch::nn::ReLU`), please refer to the **[official PyTorch C++ documentation](https://pytorch.org/cppdocs/)**.

A good mental model is:
- **LibTorch** provides the core building blocks (tensors, layers, autograd).
- **xTorch** provides the complete toolkit to assemble those blocks into production-ready workflows (trainers, data loaders, pre-built models, transforms).

---

## Library Modules

The xTorch API is organized into a series of modules, each corresponding to a specific part of the machine learning workflow. Use the links below to navigate to the detailed documentation for each component.

-   **[Activations](activations.md)**: A comprehensive collection of modern and experimental activation functions to introduce non-linearity into your models.

-   **[DataLoaders](dataloaders.md)**: High-performance, easy-to-use utilities for batching, shuffling, and parallel loading of datasets.

-   **[Datasets](datasets/index.md)**: A collection of built-in dataset handlers for various domains (vision, NLP, audio) and general-purpose classes like `ImageFolder` for loading custom data.

-   **[Dropouts](dropouts.md)**: An extensive library of advanced dropout techniques for model regularization, going far beyond standard dropout.

-   **[Losses](losses.md)**: A collection of specialized loss functions for tasks like metric learning, object detection, and robust training.

-   **[Models](models/index.md)**: A model zoo containing pre-built, ready-to-use implementations of popular and state-of-the-art architectures.

-   **[Normalizations](normalizations.md)**: Implementations of various normalization layers beyond standard `BatchNorm`, such as `LayerNorm`, `InstanceNorm`, and more experimental variants.

-   **[Optimizations](optimizations.md)**: Advanced and recent optimization algorithms to complement the standard optimizers provided by LibTorch.

-   **[Regularizations](regularizations.md)**: A collection of explicit regularization techniques that can be applied during training.

-   **[Trainers](trainers.md)**: The core training engine, featuring the `Trainer` class and a `Callback` system to abstract and manage the training loop.

-   **[Transforms](transforms/index.md)**: A rich library of data preprocessing and augmentation functions for images, audio, text, and other data modalities.

-   **[Utilities](utils.md)**: A set of helper functions and tools for common tasks like logging, device management, and filesystem operations.
