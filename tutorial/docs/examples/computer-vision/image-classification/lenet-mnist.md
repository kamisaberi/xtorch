# Image Classification: Training LeNet on MNIST

This example provides a complete, end-to-end walkthrough of training the classic LeNet-5 model on the MNIST handwritten digit dataset. It is a foundational "Hello, World!" for computer vision in C++.

We will cover two approaches:
1.  **The Manual Training Loop**: A step-by-step implementation of the training process to understand all the core components involved (forward pass, loss calculation, backward pass, optimizer step).
2.  **The `xt::Trainer` Loop**: The recommended high-level approach that abstracts away the boilerplate for cleaner code.

This tutorial is perfect for understanding what happens "under the hood" during model training.

---

## 1. The Manual Training Loop

While the `xt::Trainer` is recommended for most use cases, writing the training loop manually gives you maximum control and is excellent for learning. The process involves iterating through epochs and batches and manually executing each step of the learning algorithm.

### The Steps in a Manual Loop:
1.  **Zero Gradients**: Clear any gradients from the previous iteration.
2.  **Forward Pass**: Pass the input data through the model to get predictions.
3.  **Calculate Loss**: Compare the model's predictions to the true labels using a loss function.
4.  **Backward Pass**: Calculate the gradients of the loss with respect to the model's parameters using `autograd`.
5.  **Update Weights**: Instruct the optimizer to update the model's parameters using the calculated gradients.

---

## Full C++ Code

Below is the complete source code for training LeNet-5 on MNIST with a manual loop. It includes setup for the data, model, and optimizer, followed by the explicit loop.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>
#include <chrono>

int main() {
    // --- 1. Setup ---
    std::cout.precision(10);
    const int epochs = 10;
    const int num_threads = 16;
    const int num_workers = 16;
    const int batch_size = 64;

    torch::set_num_threads(num_threads);
    std::cout << "Using " << torch::get_num_threads() << " threads for LibTorch" << std::endl;

    // Use CPU for this example
    torch::Device device(torch::kCPU);

    // --- 2. Data Pipeline ---
    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}));
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5})
    );
    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);

    auto dataset = xt::datasets::MNIST(
        "/path/to/your/datasets/", // IMPORTANT: Change this path
        xt::datasets::DataMode::TRAIN,
        /*download=*/true,
        std::move(compose)
    );

    xt::dataloaders::ExtendedDataLoader data_loader(dataset, batch_size, true, num_workers);

    // --- 3. Model and Optimizer ---
    xt::models::LeNet5 model(10);
    model.to(device);
    model.train(); // Set the model to training mode

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    // --- 4. The Manual Training Loop ---
    std::cout << "\nStarting manual training loop..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        int batch_count = 0;
        for (auto& batch : data_loader) {
            auto data = batch.first.to(device);
            auto target = batch.second.to(device);

            // 1. Zero the gradients
            optimizer.zero_grad();

            // 2. Forward pass
            auto output = model.forward(data);

            // 3. Calculate loss
            auto loss = torch::nll_loss(output, target);

            // 4. Backward pass to compute gradients
            loss.backward();

            // 5. Update model weights
            optimizer.step();

            if (++batch_count % 100 == 0) {
                std::cout << "Epoch: " << epoch << "/" << epochs
                          << " | Batch: " << batch_count << "/" << *dataset.size() / batch_size
                          << " | Loss: " << loss.item<float>() << std::endl;
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nTotal training duration: " << duration_ms.count() << " milliseconds." << std::endl;

    return 0;
}
```

## 2. The Simplified `xt::Trainer` Approach

The manual loop above can be replaced entirely by the `xt::Trainer`. The code below achieves the exact same result with significantly less boilerplate.

```cpp
// --- This code REPLACES the manual for-loop section above ---

// xt::Trainer trainer;
// auto logger = std::make_shared<xt::LoggingCallback>("[LeNet-MNIST]", 100);

// trainer.set_max_epochs(10)
//        .set_optimizer(optimizer)
//        .set_loss_fn(torch::nll_loss)
//        .add_callback(logger);

// std::cout << "\nStarting training with xt::Trainer..." << std::endl;
// auto start_time = std::chrono::steady_clock::now();

// trainer.fit(model, data_loader, nullptr, device);

// auto end_time = std::chrono::steady_clock::now();
// auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
// std::cout << "\nTotal training duration: " << duration_ms.count() << " milliseconds." << std::endl;
```

This demonstrates the core value of xTorch: it handles the complex and repetitive parts of training, allowing you to focus on your experiment.