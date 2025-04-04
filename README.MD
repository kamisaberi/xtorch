
# Libtorch-Extended: Bridging the Usability Gap in PyTorch’s C++ API

## Motivation

PyTorch’s C++ library (LibTorch) emerged as a powerful way to use PyTorch outside Python, but after 2019 it became challenging for developers to use it for end-to-end model development. Early on, LibTorch aimed to mirror the high-level Python API, yet many convenient abstractions and examples never fully materialized or were later removed.

As of 2020, the C++ API had achieved near feature-parity with Python’s core operations, but it lagged in usability and community support. Fewer contributors focused on C++ meant that only low-level building blocks were provided, with high-level components (e.g. ready-made network architectures, datasets) largely absent. This left C++ practitioners to rewrite common tools from scratch – implementing standard models or data loaders manually – which is time-consuming and error-prone.

Another factor was PyTorch’s emphasis on the Python-to-C++ workflow. The official recommended path for production was to prototype in Python, then convert models to TorchScript for C++ deployment. This approach deprioritized making the pure C++ experience as friendly as Python’s.

As a result, developers who preferred or needed to work in C++ (for integration with existing systems, performance, or deployment constraints) found LibTorch cumbersome. Simple tasks like data augmentation (e.g. random crops or flips) had no built-in support in LibTorch C++. Defining neural network modules in C++ involved boilerplate macros and manual registration, an awkward process compared to Python’s concise syntax. Crucial functionality for model serialization was limited – for instance, LibTorch could load Python-exported models but not easily export its own models to a portable format.

Libtorch-Extended was created to address this gap. It is a C++ library that extends LibTorch with the high-level abstractions and utilities that were missing or removed after 2019. By building on LibTorch’s robust computational core, libtorch-extended restores ease-of-use without sacrificing performance. The motivation is to empower C++ developers with a productive experience similar to PyTorch in Python – enabling them to build, train, and deploy models with minimal fuss. In essence, libtorch-extended revives and modernizes the “batteries-included” ethos for C++ deep learning, providing an all-in-one toolkit where the base library left off.

## Design and Architecture

Libtorch-Extended is architected as a thin layer on top of LibTorch’s C++ API, carefully integrating with it rather than reinventing it. The design follows a modular approach, adding a higher-level API that wraps around LibTorch’s lower-level classes. At its core, libtorch-extended relies on LibTorch for tensor operations, autograd, and neural network primitives – effectively using LibTorch as the computational engine. The extended library then introduces its own set of C++ classes that encapsulate common patterns (model definitions, training loops, data handling, etc.), providing a cleaner interface to the developer.

### Architecture Layers
- **LibTorch Core (Bottom Layer):** Provides `torch::Tensor`, `torch::autograd`, `torch::nn`, optimizers, etc.
- **Extended Abstraction Layer (Middle):** Simplified classes inheriting from LibTorch core (e.g., `ExtendedModel`, `Trainer`).
- **User Interface (Top Layer):** Intuitive APIs and boilerplate-free interaction.

### Modules
- **Model Module:** High-level model class extensions.
- **Data Module:** Enhanced datasets and DataLoader.
- **Training Module:** Training logic, checkpointing, metrics.
- **Utilities Module:** Logging, device helpers, summaries.

## Features and Enhancements

- **High-Level Model Classes:** `XTModule`, prebuilt models like `ResNetExtended`, `XTCNN`.
- **Simplified Training Loop (Trainer):** Full training abstraction with callbacks and metrics.
- **Enhanced Data Handling:** `ImageFolderDataset`, `CSVDataset`, OpenCV-backed support.
- **Utility Functions:** Logging, metrics, summary, device utils.
- **Extended Optimizers:** AdamW, RAdam, schedulers, learning rate strategies.
- **Model Serialization & Deployment:** `save_model()`, `export_to_jit()`, inference helpers.

## Use Cases and Examples

### Example: CNN Training Pipeline (Simplified)

```cpp
#include <xtorch/xtorch.hpp>

int main() {
    auto trainData = xt::datasets::ImageFolder("data/train",
        xt::transforms::Compose({
            xt::transforms::Resize(224),
            xt::transforms::RandomCrop(224),
            xt::transforms::Normalize({0.5,0.5,0.5},{0.5,0.5,0.5})
        }));
    auto trainLoader = xt::DataLoader(trainData, 64, true, 4);
    auto valLoader = xt::DataLoader(xt::datasets::ImageFolder("data/val"), 64);

    auto model = xt::models::ResNet18(10);
    model.to(torch::kCUDA);

    auto optimizer = xt::optim::AdamW(model.parameters(), 1e-3);
    auto criterion = torch::nn::CrossEntropyLoss();

    xt::Trainer trainer;
    trainer.setMaxEpochs(20)
           .setOptimizer(optimizer)
           .setLossFn(criterion)
           .enableCheckpoint("checkpoints/resnet18.pt", 5);
    trainer.fit(model, trainLoader, valLoader);

    xt::export_to_jit(model, "resnet18_script.pt");
}
```

### Example: C++ Inference Pipeline

```cpp
auto model = xt::load_model("resnet18_script.pt");
auto tensor = xt::utils::imageToTensor("input.jpg");
auto outputs = xt::utils::predict(model, tensor);
int predictedClass = xt::utils::argmax(outputs);
std::cout << "Predicted class = " << predictedClass << std::endl;
```

## Impact and Potential Applications

- **C++ Developers:** Enables use of PyTorch-like training without Python.
- **Research in Embedded / HPC:** Pure C++ training and deployment possible.
- **Industrial Use:** On-device training, edge deployment workflows.
- **Education:** Useful for teaching performance-aware ML in C++.
- **Ecosystem Growth:** Boosts community contributions, reuse, and experimentation.

## Comparison with Related Tools

| Feature                     | LibTorch | Libtorch-Extended | PyTorch Lightning (Python) |
|----------------------------|----------|-------------------|-----------------------------|
| Training Loop Abstraction  | ❌       | ✅                | ✅                          |
| Data Augmentation Built-in | ❌       | ✅                | ✅                          |
| Built-in Model Zoo         | Limited  | ✅                | ✅                          |
| Target Language            | C++      | C++               | Python                      |
| TorchScript Export         | Limited  | ✅                | ✅                          |

Libtorch-Extended complements PyTorch’s C++ API like PyTorch Lightning does in Python, enabling expressive ML development in C++ with clean, modular code structures.

