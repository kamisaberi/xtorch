# Architecture & Design Philosophy

The design of xTorch is guided by a simple yet powerful principle: **extend, don't reinvent**. Instead of creating a new deep learning framework from scratch, xTorch is architected as a thin, non-intrusive layer on top of PyTorch's C++ API (LibTorch). This approach allows xTorch to provide a high-level, user-friendly experience without sacrificing the underlying performance and flexibility of LibTorch's computational engine.

The core philosophy is to provide the "batteries-included" components that make the Python experience productive, directly in C++.

---

## Core Principles

1.  **Leverage LibTorch's Power**: xTorch relies entirely on LibTorch for its core functionality. All tensor operations, automatic differentiation (autograd), and neural network primitives (`torch::nn`) are delegated to the battle-tested LibTorch backend. We focus on the API, not the engine.

2.  **Usability and Productivity**: The primary goal is to reduce boilerplate and make the C++ API as expressive and intuitive as its Python counterpart. This is achieved through high-level abstractions like the `Trainer` loop and pre-built data loaders.

3.  **Modularity**: The library is organized into distinct, logical modules (`models`, `datasets`, `transforms`, `losses`, etc.). This makes the framework easy to navigate, learn, and contribute to. You can use as much or as little of xTorch as you need.

## Architectural Layers

xTorch's architecture can be visualized as a simple stack. Your application code interacts with the high-level xTorch API, which in turn orchestrates the powerful but lower-level LibTorch core.

```mermaid
graph TD
    subgraph User Space
        A[Your C++ Application]
    end

    subgraph xTorch Abstraction Layer
        B(trainer.fit(model, data_loader, ...))
        C{Trainer, Callbacks, Logging}
        D{Datasets, DataLoaders, Transforms}
        E{Pre-built Models, Losses, Optimizers}
    end

    subgraph LibTorch Core Engine
        F[torch::Tensor]
        G[torch::autograd]
        H[torch::nn]
        I[torch::optim]
    end

    A --> B
    B --> C
    B --> D
    B --> E
    C & D & E --> F & G & H & I
```

Let's break down these layers:

#### 1. Bottom Layer: The LibTorch Core Engine

This is the foundation. It provides all the essential building blocks for deep learning:
- **`torch::Tensor`**: The fundamental multi-dimensional array object.
- **`torch::autograd`**: The engine for automatically computing gradients.
- **`torch::nn`**: Primitives for building neural networks, like `Linear`, `Conv2d`, and activation functions.
- **`torch::optim`**: Core optimizers like `Adam` and `SGD`.

xTorch does not modify this layer; it simply uses it as a robust and high-performance backend.

#### 2. Middle Layer: The xTorch Abstraction Layer

This is where xTorch adds its value. It provides a set of C++ classes and functions that wrap common patterns and encapsulate complexity. This layer is designed for convenience and rapid development. Key components include:
- `xt::Trainer`: A complete, abstracted training and validation loop.
- `xt::datasets::ImageFolder`: A simple way to load image data from a directory structure.
- `xt::dataloaders::ExtendedDataLoader`: A powerful data loader with parallel processing.
- `xt::models::ResNet`: An example of a pre-built, ready-to-use model architecture.
- `xt::transforms`: A rich library of data augmentation and preprocessing functions.

#### 3. Top Layer: The User Application

This is your code. By using xTorch, your application logic becomes cleaner and more focused on the high-level aspects of your model and experiment. Instead of manually writing loops, moving data to devices, and calculating gradients, you interact with the intuitive xTorch API.

---

## Key Modules

The xTorch library is organized into several key modules, each serving a specific purpose in the ML workflow.

| Module | Purpose & Key Components |
|---|---|
| **Model Module** | Provides high-level model classes and a zoo of pre-built architectures. It simplifies model definition and reduces boilerplate. Includes `XTModule` and ready-made models like `LeNet5`, `ResNet`, and `DCGAN`. |
| **Data Module** | Streamlines data loading and preprocessing. Contains enhanced `Dataset` classes (`ImageFolderDataset`, `CSVDataset`) and a powerful `ExtendedDataLoader` with built-in support for transformations. |
| **Training Module** | Contains the core logic for model training and validation. The centerpiece is the `Trainer` class, which automates the training loop, supported by a system of `Callbacks` for logging, checkpointing, and metrics tracking. |
| **Transforms Module**| Offers a vast library of data augmentation and preprocessing functions for various modalities like images, audio, and text. Mirroring the functionality of `torchvision.transforms`. |
| **Utilities Module** | A collection of helper functions and classes for common tasks, including logging, device management (`torch::kCUDA` vs. `torch::kCPU`), model summary printing, and filesystem operations. |
| **Extended Optimizers & Losses** | Provides implementations of newer or more advanced optimizers (`AdamW`, `RAdam`) and loss functions that may not be available in the core LibTorch library. |

This modular and layered architecture allows xTorch to provide a modern, productive deep learning experience in C++, bridging the gap left by the core PyTorch C++ API.