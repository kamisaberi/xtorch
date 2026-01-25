# Getting Started: Using the xTorch Trainer for Easy Training

In the previous tutorials, you learned how to create Tensors and build a simple neural network. Now, we will focus on the most powerful high-level utility in xTorch: the **`xt::Trainer`**.

The `Trainer` is an abstraction that handles the entire training loop for you. Manually writing a training loop involves several repetitive steps:
-   Looping through epochs.
-   Looping through batches of data.
-   Moving data to the correct device (CPU/GPU).
-   Zeroing the optimizer's gradients.
-   Making a forward pass.
-   Calculating the loss.
-   Making a backward pass to compute gradients.
-   Stepping the optimizer to update weights.
-   Logging progress and metrics.

The `xt::Trainer` automates all of this, allowing you to write cleaner, more readable code and focus on your model's architecture and data.

---

## 1. The Manual Training Loop (The Hard Way)

To appreciate what the `Trainer` does, let's first look at a standard, manual training loop written in pure LibTorch. We will train the simple `Net` we defined in the previous tutorial on the MNIST dataset.

```cpp
// --- This is the verbose, manual way ---

// Assume model, data_loader, optimizer, and device are already defined
const int num_epochs = 5;
size_t batch_idx = 0;

for (int epoch = 1; epoch <= num_epochs; ++epoch) {
    model->train(); // Set the model to training mode
    for (auto& batch : data_loader) {
        auto data = batch.first.to(device);
        auto target = batch.second.to(device);

        // 1. Zero the gradients
        optimizer.zero_grad();
        // 2. Forward pass
        auto output = model->forward(data);
        // 3. Calculate loss
        auto loss = torch::nll_loss(output, target);
        // 4. Backward pass
        loss.backward();
        // 5. Update weights
        optimizer.step();

        if (batch_idx % 100 == 0) {
            std::cout << "Epoch: " << epoch << " | Batch: " << batch_idx
                      << " | Loss: " << loss.item<float>() << std::endl;
        }
        batch_idx++;
    }
}
```
This works, but it's verbose. Imagine adding validation loops, model checkpointing, or early stopping—the code would quickly become much more complex.

---

## 2. Using the `xt::Trainer` (The Easy Way)

Now, let's achieve the exact same result using the `xt::Trainer`. The process involves three simple steps:
1.  **Instantiate** the `Trainer`.
2.  **Configure** it using its chainable setter methods (`set_max_epochs`, `set_optimizer`, etc.). We also add a `LoggingCallback` to handle printing our progress.
3.  **Call** the `fit()` method to start the training.

The `Trainer` handles all the steps from the manual loop internally.

```cpp
// --- This is the clean, xTorch way ---

// Assume model, data_loader, optimizer, and device are already defined

// 1. Instantiate the Trainer
xt::Trainer trainer;

// 2. Configure the Trainer
trainer.set_max_epochs(5)
       .set_optimizer(optimizer)
       .set_loss_fn(torch::nll_loss)
       .add_callback(std::make_shared<xt::LoggingCallback>("[Trainer-MNIST]", 100));

// 3. Run the training process
trainer.fit(*model, data_loader, nullptr, device);
```
That's it! The `Trainer` provides a clean, high-level API that encapsulates the complexity of the training loop, making your code more modular and easier to read and maintain.

---

## Full C++ Code

Below is the complete, runnable source code for this example. The original file is located at `getting_started/using_the_xtorch_trainer_for_easy_training.cpp`.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

// A simple MLP for MNIST classification
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
    // --- 1. Setup ---
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    auto dataset = xt::datasets::MNIST("./data", xt::datasets::DataMode::TRAIN, true);
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 64, true);

    auto model = std::make_shared<Net>();
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    // --- 2. Instantiate and Configure the Trainer ---
    xt::Trainer trainer;
    trainer.set_max_epochs(5)
           .set_optimizer(optimizer)
           .set_loss_fn(torch::nll_loss) // Use the standard NLL loss function
           .add_callback(std::make_shared<xt::LoggingCallback>("[Trainer-MNIST]", 100));

    std::cout << "Starting training with the xTorch Trainer..." << std::endl;

    // --- 3. Run the Training ---
    trainer.fit(*model, data_loader, /*validation_loader=*/nullptr, device);

    std::cout << "\nTraining finished!" << std::endl;
    return 0;
}
```

## How to Compile and Run

This example can be found in the `xtorch-examples` repository.
1.  Navigate to the `getting_started/` directory.
2.  Build using CMake:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
3.  Run the executable:
    ```bash
    ./use_trainer
    ```

## Key Takeaway

The `xt::Trainer` is the cornerstone of xTorch's high-level API. By abstracting the training loop, it allows for cleaner code, easier experimentation, and the ability to add complex features like logging, checkpointing, and validation through a simple and extensible **Callback** system.