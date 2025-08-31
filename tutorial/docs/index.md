![xTorch Logo](assets/logo.png)

# xTorch: The Batteries-Included C++ Library for PyTorch

**Bridging the usability gap in PyTorch’s C++ API with high-level abstractions for building, training, and deploying models entirely in C++.**

[GitHub Repository](https://github.com/kamisaberi/xtorch){ .md-button }
[Get Started](getting-started/installation/){ .md-button .md-button--primary }
[View Examples](examples/){ .md-button }

!!! danger "Library Under Development"
    xTorch is currently under active development. The API may change. For stability, please use an official release version for production workloads.

---

## The Motivation: A First-Class C++ Deep Learning Experience

PyTorch's C++ library, LibTorch, provides a powerful and performant core for deep learning. However, after 2019, its focus shifted primarily to a deployment-only role, leaving a significant gap for developers who wanted or needed to conduct end-to-end model development directly in C++. High-level utilities, data loaders with augmentations, and a rich model zoo—features that make the Python API so productive—were either missing or deprecated.

**xTorch was created to fill this gap.**

It extends LibTorch with the high-level, "batteries-included" abstractions that C++ developers have been missing. By building a thin, intuitive layer on top of LibTorch’s robust core, xTorch restores ease-of-use without sacrificing the raw performance of C++. Our goal is to empower C++ developers with a productive experience on par with PyTorch in Python, enabling them to build, train, and deploy models with minimal boilerplate and maximum efficiency.

## Key Features

| Feature                       | Description                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| :material-layers-triple: **High-Level Abstractions** | Simplified model classes, pre-built architectures (`ResNet`, `DCGAN`), and intuitive APIs.      |
| :material-run-fast: **Simplified `Trainer` Loop**  | A powerful, callback-driven training loop that handles optimization, metrics, logging, and checkpointing. |
| :material-database-search: **Enhanced Data Handling**  | Built-in datasets (`ImageFolder`, `MNIST`), powerful `DataLoader`, and a rich library of data transforms. |
| :material-rocket-launch: **Seamless Serialization** | Easily save, load, and export models to TorchScript for production-ready inference pipelines.         |
| :material-chart-line: **Uncompromised Performance** | Eliminate Python overhead. Achieve significant speedups over standard PyTorch workflows.             |
| :material-wrench: **Extensive Toolkit**       | A massive collection of optimizers, loss functions, normalizations, and regularization techniques.      |

## A Quick Look

See how xTorch transforms a verbose C++ training task into a few lines of clean, expressive code.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // 1. Load data with transforms
    auto dataset = xt::datasets::MNIST("./data", xt::datasets::DataMode::TRAIN,
        std::make_unique<xt::transforms::Compose>(
            std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}),
            std::make_shared<xt::transforms::general::Normalize>(0.5, 0.5)
        )
    );
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 64, true);

    // 2. Define model and optimizer
    xt::models::LeNet5 model(10);
    torch::optim::Adam optimizer(model.parameters(), 1e-3);

    // 3. Configure and run the trainer
    xt::Trainer trainer;
    trainer.set_max_epochs(10)
           .set_optimizer(optimizer)
           .set_loss_fn(torch::nll_loss)
           .add_callback(std::make_shared<xt::LoggingCallback>());

    trainer.fit(model, data_loader, nullptr, torch::kCPU);

    return 0;
}
```

## Who is this for?

-   **C++ Developers** who want to leverage a PyTorch-like ML framework without leaving their primary ecosystem.
-   **Performance Engineers** needing to eliminate Python bottlenecks for data-intensive training or inference workloads.
-   **Researchers & Students** in HPC, robotics, or embedded systems where pure C++ deployment is a necessity.
-   **Educators** looking for a tool to teach performance-aware machine learning concepts in C++.

---

Ready to dive in? Check out the [**Installation Guide**](getting-started/installation.md) to set up your environment.
