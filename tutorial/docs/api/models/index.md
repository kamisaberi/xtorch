# Models

The `xt::models` module provides a comprehensive "Model Zoo" containing pre-built, ready-to-use implementations of many popular and state-of-the-art deep learning architectures.

This allows you to get started on standard tasks quickly without having to implement well-known models from scratch. It is perfect for benchmarking, transfer learning, and as a starting point for your own custom architectures.

## Core Concept

All models in the `xt::models` namespace are implemented as standard `torch::nn::Module`s. This means they integrate seamlessly with the entire LibTorch and xTorch ecosystem. You can inspect their parameters, move them between devices, and pass them to any xTorch `Trainer` or standard LibTorch `Optimizer`.

## General Usage

Using a pre-built model from xTorch is straightforward. The typical workflow is:
1.  Include the header for the specific model you want to use.
2.  Instantiate the model class, providing any necessary configuration (e.g., number of classes).
3.  Move the model to the desired device (`CPU` or `CUDA`).
4.  Use the model for training or inference.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // Define the device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // 1. & 2. Instantiate a pre-built ResNet18 model for a 100-class problem
    xt::models::ResNet model(
        xt::models::ResNetImpl::BlockType::BASIC, // BASIC block for ResNet18/34
        {2, 2, 2, 2},                             // Layers in each stage for ResNet18
        /*num_classes=*/100
    );

    // 3. Move the model to the GPU
    model.to(device);

    // 4. Perform a dummy forward pass
    // Create a random input tensor: Batch size 16, 3 channels, 224x224 image
    auto input = torch::randn({16, 3, 224, 224}).to(device);
    auto output = model.forward(input);

    std::cout << "Model instantiated successfully." << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl; // Should be

    // The model can now be passed to an optimizer and the xt::Trainer
    // torch::optim::Adam optimizer(model.parameters());
    // xt::Trainer trainer;
    // trainer.fit(model, ...);
}
```

!!! info "Model Variants"
    Many model families (like ResNet or EfficientNet) have multiple variants (e.g., ResNet18, ResNet50). These are typically configured through constructor arguments. Please refer to the specific model's header file for details on the available options.

---

## Available Models by Domain

The xTorch Model Zoo is organized by machine learning domain. Follow the links below for a detailed list of available architectures in each category.

-   **[Computer Vision](computer-vision.md)**: Includes classic and modern architectures for image classification (`ResNet`, `VGG`, `EfficientNet`), object detection (`YOLO`, `Faster R-CNN`), segmentation (`U-Net`, `DeepLab`), and Vision Transformers (`ViT`, `Swin`).

-   **[Generative Models](generative.md)**: Contains implementations of popular generative architectures, including Generative Adversarial Networks (`DCGAN`, `StyleGAN`), Variational Autoencoders (`VAE`), and Diffusion Models (`DDPM`).

-   **[Natural Language Processing](nlp.md)**: A collection of models for text-based tasks, including recurrent architectures (`Seq2Seq`) and a wide range of Transformer-based models (`BERT`, `GPT`, `T5`, `Llama`).

-   **[Graph Neural Networks](gnn.md)**: Implementations of common GNN architectures for learning on graph-structured data, such as `GCN`, `GraphSAGE`, and `GAT`.

-   **[Reinforcement Learning](rl.md)**: A collection of models and policies for reinforcement learning, including `DQN`, `A3C`, and `PPO`.

-   **[Multimodal](multimodal.md)**: Models designed to process and fuse information from multiple data types, such as `CLIP` (text and image) and `ViLBERT`.
