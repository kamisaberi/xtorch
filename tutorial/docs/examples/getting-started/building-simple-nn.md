# Getting Started: Build and Train a Simple Neural Network

This tutorial follows the [Quick Start](quick-start-cnn.md) example, but instead of using a pre-built model like `LeNet5`, we will define our own simple neural network from scratch.

This is a crucial step in learning any deep learning framework. You will learn how to:
1.  Define a custom network architecture by inheriting from `torch::nn::Module`.
2.  Register layers (like `Linear` and `ReLU`) within your model.
3.  Implement the `forward` pass to define how data flows through your network.
4.  Train your custom model using the `xt::Trainer`.

Our goal is to build a simple Multi-Layer Perceptron (MLP) to classify digits from the MNIST dataset.

---

## 1. Defining the Network (`Net` module)

The core of any model in PyTorch or xTorch is a class that inherits from `torch::nn::Module`. Inside this class, you define the layers of your network and then implement the `forward` method, which specifies the computation that happens at every forward pass.

Our MLP will have a simple architecture:
-   An input layer that flattens the 28x28 images into a 784-element vector.
-   A linear layer that maps 784 features to 64 hidden features.
-   A ReLU activation function.
-   A second linear layer that maps the 64 hidden features to the 10 output classes.
-   A LogSoftmax layer to convert the output logits into log-probabilities, which is suitable for use with the `nll_loss` function.

```cpp
#include <xtorch/xtorch.h>

// Define a custom module named 'Net'
struct Net : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    // The constructor is where we define and register our layers.
    Net() {
        // First fully-connected layer (784 in, 64 out)
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        // Second fully-connected layer (64 in, 10 out)
        fc2 = register_module("fc2", torch::nn::Linear(64, 10));
    }

    // The forward method defines the data flow.
    torch::Tensor forward(torch::Tensor x) {
        // Flatten the image tensor from [Batch, 1, 28, 28] to [Batch, 784]
        x = x.view({-1, 784});
        // Apply the first linear layer, followed by a ReLU activation
        x = torch::relu(fc1(x));
        // Apply the second linear layer
        x = fc2(x);
        // Apply log_softmax to get log-probabilities for the loss function
        return torch::log_softmax(x, /*dim=*/1);
    }
};
```

## 2. The Full Training Pipeline

The rest of the code is very similar to the Quick Start guide. We load the MNIST dataset, create a `DataLoader`, instantiate our custom `Net` model, define an optimizer, and then use the `xt::Trainer` to handle the entire training process.

---

## Full C++ Code

Below is the complete source code for this example. The original file can be found at `getting_started/building_and_training_a_simple_neural_network.cpp`.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

// --- 1. Define the custom Neural Network Module ---
struct Net : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({-1, 784});
        x = torch::relu(fc1(x));
        x = fc2(x);
        return torch::log_softmax(x, /*dim=*/1);
    }
};


int main() {
    // --- 2. Setup Device and Data ---
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Define a simple normalization transform
    auto transform = std::make_unique<xt::transforms::Compose>(
        std::make_shared<xt::transforms::general::Normalize>(0.5, 0.5)
    );

    auto dataset = xt::datasets::MNIST(
        "./data",
        xt::datasets::DataMode::TRAIN,
        /*download=*/true,
        std::move(transform)
    );

    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 64, true);

    // --- 3. Instantiate Model and Optimizer ---
    auto model = std::make_shared<Net>();
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    // --- 4. Configure and Run the Trainer ---
    xt::Trainer trainer;
    trainer.set_max_epochs(5)
           .set_optimizer(optimizer)
           .set_loss_fn(torch::nll_loss)
           .add_callback(std::make_shared<xt::LoggingCallback>("[SimpleNN-MNIST]", 100));

    trainer.fit(*model, data_loader, nullptr, device);

    std::cout << "Training finished!" << std::endl;
    return 0;
}
```

## How to Compile and Run

You can find this example in the `xtorch-examples` repository. To compile and run it:
1.  Navigate to the `getting_started/` directory within the examples.
2.  Create a build directory and use CMake:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
3.  Run the compiled executable:
    ```bash
    ./build_simple_nn
    ```

## Expected Output

You will see the training progress printed to the console, showing that your custom model is successfully learning to classify the MNIST digits.

```
[SimpleNN-MNIST] Epoch 1/5, Batch 100/938 - Loss: 0.4512345678 - Time per batch: ...ms
[SimpleNN-MNIST] Epoch 1/5, Batch 200/938 - Loss: 0.3125487566 - Time per batch: ...ms
...
Training finished!
```

This example demonstrates a fundamental skill: defining and training your own architectures. You can now use this pattern to build more complex and powerful models for your own tasks.