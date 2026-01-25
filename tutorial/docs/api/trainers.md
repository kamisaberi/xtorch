# Trainers and Callbacks

The `Trainer` module is the centerpiece of xTorch's high-level API. It encapsulates the entire training and validation loop, abstracting away the boilerplate code required to iterate over data, perform forward and backward passes, update model weights, and run validation checks.

This allows you to focus on the high-level architecture of your model and experiment, rather than the low-level mechanics of the training process.

---

## `xt::Trainer`

The `xt::Trainer` class is the main engine for model training. It is designed with a fluent, chainable interface (a builder pattern) that makes configuration clean and readable.

### Core Responsibilities

The `Trainer` handles all of the following automatically:
-   Iterating over the dataset for a specified number of epochs.
-   Iterating over batches from the `DataLoader`.
-   Moving data and models to the correct device (`CPU` or `CUDA`).
-   Setting the model to the correct mode (`train()` or `eval()`).
-   Zeroing gradients (`optimizer.zero_grad()`).
-   Performing the forward pass (`model.forward(data)`).
-   Calculating the loss.
-   Performing the backward pass (`loss.backward()`).
-   Updating the model's weights (`optimizer.step()`).
-   Executing custom logic at specific points via a callback system.

### Configuration

You configure a `Trainer` instance by chaining its setter methods.

| Method | Description |
|---|---|
| `set_max_epochs(int epochs)` | **Required.** Sets the total number of epochs to train for. |
| `set_optimizer(torch::optim::Optimizer& optim)` | **Required.** Sets the optimizer to use for updating weights. |
| `set_loss_fn(LossFn loss_fn)` | **Required.** Sets the loss function. This can be a `torch::nn::Module` (like `torch::nn::CrossEntropyLoss`) or a lambda function. |
| `add_callback(std::shared_ptr<Callback> cb)`| **Optional.** Adds a callback to inject custom logic into the training loop. |

### Execution

Once configured, you start the training process by calling the `fit()` method.

`fit(torch::nn::Module& model, dataloaders::ExtendedDataLoader& train_loader, dataloaders::ExtendedDataLoader* val_loader, torch::Device device)`

| Parameter | Type | Description |
|---|---|---|
| `model` | `torch::nn::Module&` | The model to be trained. |
| `train_loader` | `ExtendedDataLoader&` | The data loader for the training dataset. |
| `val_loader` | `ExtendedDataLoader*` | **Optional.** A pointer to the data loader for the validation dataset. If provided (`nullptr` otherwise), a validation loop will be run at the end of each epoch. |
| `device` | `torch::Device` | The device (`torch::kCPU` or `torch::kCUDA`) on which to run the training. |

---

## Callbacks

Callbacks are the primary mechanism for extending the `Trainer`'s functionality. A callback is an object that can perform actions at various stages of the training loop (e.g., at the end of an epoch, at the beginning of a batch).

This powerful system allows you to add custom logic for:
-   Logging metrics to the console or a file.
-   Saving model checkpoints.
-   Implementing early stopping.
-   Adjusting the learning rate.

### Creating a Custom Callback

To create your own callback, you inherit from the base class `xt::Callback` and override any of its virtual methods.

**Available Hooks (Methods to Override):**
- `on_train_begin()`
- `on_train_end()`
- `on_epoch_begin()`
- `on_epoch_end()`
- `on_batch_begin()`
- `on_batch_end()`

### Built-in Callbacks

xTorch provides a set of common callbacks to handle standard tasks.

#### `xt::LoggingCallback`
This is the most essential callback. It prints a formatted progress log to the console, showing the current epoch, batch, loss, and timing information.

**Constructor:**
`LoggingCallback(std::string name, int log_every_N_batches = 50, bool log_time = true)`

---

## Complete Usage Example

This snippet demonstrates how all the pieces fit together.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // 1. Initialize Model, DataLoaders, and Optimizer
    xt::models::LeNet5 model(10);
    torch::Device device(torch::kCUDA);
    model.to(device);

    auto dataset = xt::datasets::MNIST("./data");
    xt::dataloaders::ExtendedDataLoader train_loader(dataset, 64, true);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    // 2. Create a Logging Callback
    auto logger = std::make_shared<xt::LoggingCallback>("[MNIST-TRAIN]", /*log_every*/ 100);

    // 3. Instantiate and Configure the Trainer
    xt::Trainer trainer;
    trainer.set_max_epochs(10)
           .set_optimizer(optimizer)
           .set_loss_fn(torch::nll_loss) // Using a standard functional loss
           .add_callback(logger);        // Add the logger

    // 4. Start the training process
    trainer.fit(model, train_loader, /*val_loader=*/nullptr, device);

    std::cout << "Training complete." << std::endl;
}
```