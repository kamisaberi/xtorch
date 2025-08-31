# Quick Start: Training a CNN

This guide will walk you through a complete, end-to-end example of training a simple Convolutional Neural Network (CNN) using xTorch. It is designed to be your first practical step after completing the [Installation](installation.md).

We will train the classic **LeNet-5** model on the **MNIST** dataset of handwritten digits.

This example showcases the core components of the xTorch workflow:
1.  Defining data transformations.
2.  Loading a dataset and creating a `DataLoader`.
3.  Initializing a pre-built model.
4.  Configuring the `Trainer` for a seamless training loop.

---

## The Workflow Explained

### Step 1: Data Transformations

Before feeding data to the model, we need to process it. Here, we define a sequence of two transformations:
- `Resize`: The LeNet-5 model expects `32x32` images, so we resize the `28x28` MNIST images.
- `Normalize`: We scale the pixel values to a standard range, which helps the model converge faster.

These are combined into a single pipeline using `xt::transforms::Compose`.

### Step 2: Dataset and DataLoader

We load the built-in `xt::datasets::MNIST`. The `ExtendedDataLoader` then takes this dataset and prepares batches of data for the GPU. It handles shuffling, parallel data loading (`num_workers`), and batching automatically.

### Step 3: Model and Optimizer

We instantiate a pre-built `xt::models::LeNet5` model from the xTorch model zoo. We then define a `torch::optim::Adam` optimizer and tell it which parameters (from our model) it should update.

### Step 4: The Trainer

The `xt::Trainer` is the heart of the xTorch workflow. It abstracts away the boilerplate of a typical training loop. We configure it with:
- The maximum number of epochs to run.
- The optimizer to use.
- The loss function to calculate (`torch::nll_loss`).
- A `LoggingCallback` to print progress to the console.

Finally, we call `trainer.fit()`, passing in the model and data loaders. The trainer handles the entire process: iterating through epochs, moving data to the correct device, performing forward and backward passes, updating weights, and logging metrics.

---

## Full C++ Code

This is the complete code for our training pipeline. You can find this example in the `examples` directory of the repository.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // Ensure floating point numbers are printed with sufficient precision
    std::cout.precision(10);

    // --- 1. Define Data Transformations ---
    auto compose = std::make_unique<xt::transforms::Compose>(
        // Resize images from 28x28 to the 32x32 expected by LeNet
        std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}),
        // Normalize pixel values to have a mean of 0.5 and stddev of 0.5
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5})
    );

    // --- 2. Load Dataset and Create DataLoader ---
    // Note: Replace "/path/to/datasets/" with the actual path on your system
    auto dataset = xt::datasets::MNIST(
        "/path/to/datasets/",
        xt::datasets::DataMode::TRAIN,
        /*download=*/true, // Set to true to download if not found
        std::move(compose)
    );

    xt::dataloaders::ExtendedDataLoader data_loader(
        dataset,
        /*batch_size=*/64,
        /*shuffle=*/true,
        /*num_workers=*/2
    );

    // --- 3. Initialize Model and Optimizer ---
    xt::models::LeNet5 model(/*num_classes=*/10);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model.to(device);
    model.train(); // Set the model to training mode

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    // --- 4. Configure and Run the Trainer ---
    xt::Trainer trainer;
    auto logger = std::make_shared<xt::LoggingCallback>("[LeNet-MNIST]", /*log_every_N_batches=*/100);

    trainer.set_max_epochs(10)
           .set_optimizer(optimizer)
           .set_loss_fn([](const auto& output, const auto& target) {
               return torch::nll_loss(output, target);
           })
           .add_callback(logger);

    // The `fit` method starts the training process
    trainer.fit(model, data_loader, /*validation_loader=*/nullptr, device);

    std::cout << "\nTraining finished!" << std::endl;

    return 0;
}
```

## How to Compile and Run

Assuming you have successfully installed xTorch system-wide, you can compile this file with a simple `CMakeLists.txt`.

1.  **Create a `CMakeLists.txt` file:**
    ```cmake
    cmake_minimum_required(VERSION 3.16)
    project(QuickStartCNN)

    set(CMAKE_CXX_STANDARD 17)

    # Find the installed xTorch library
    find_package(xTorch REQUIRED)

    add_executable(quick_start main.cpp)

    # Link the executable against xTorch
    target_link_libraries(quick_start PRIVATE xTorch::torch xTorch::xtorch)
    ```

2.  **Build the project:**
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

3.  **Run the executable:**
    ```bash
    ./quick_start
    ```

## Expected Output

When you run the program, you will see the `LoggingCallback` printing the training progress to your console, similar to this:

```
[LeNet-MNIST] Epoch 1/10, Batch 100/938 - Loss: 0.3125487566 - Time per batch: 5.23ms
[LeNet-MNIST] Epoch 1/10, Batch 200/938 - Loss: 0.1568745211 - Time per batch: 4.98ms
...
[LeNet-MNIST] Epoch 10/10, Batch 800/938 - Loss: 0.0215478943 - Time per batch: 4.81ms
[LeNet-MNIST] Epoch 10/10, Batch 900/938 - Loss: 0.0458712354 - Time per batch: 4.85ms

Training finished!
```

---

## Next Steps

Congratulations! You've successfully trained your first neural network using xTorch.

From here, you can explore more advanced topics:
-   **User Guide**: Dive deeper into the concepts of the [Trainer](user-guide/trainer.md) and [Data Handling](user-guide/data-handling.md).
-   **Examples**: Browse the full list of [Examples & Tutorials](examples/index.md) for more complex tasks in computer vision, NLP, and more.
