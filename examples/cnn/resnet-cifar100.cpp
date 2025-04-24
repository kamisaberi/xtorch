#include "includes/base.h"
#include "../../include/datasets/image-classification/cifar.h"
#include "../../include/models/cnn/resnet/resnet18.h"
#include "../../include/definitions/transforms.h"

using namespace std;
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

/**
 * @brief Main function to train a ResNet18 model on the CIFAR-100 dataset using PyTorch.
 *
 * This program loads the CIFAR-100 dataset, applies transformations (resize and normalization),
 * creates a data loader with sequential sampling, initializes a ResNet18 model, sets up an Adam optimizer,
 * and trains the model for 10 epochs using negative log likelihood loss. Training progress is logged every 100 batches.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
    // Set precision for floating-point output to 10 decimal places
    std::cout.precision(10);

    // Define image size for resizing (224x224 as suitable for ResNet18)
    std::vector<int64_t> size = {224, 224};

    // Select device (CUDA if available, otherwise CPU)
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Load CIFAR-100 training dataset
    // - Path: /home/kami/Documents/temp/
    // - Mode: Training
    // - Download: Enabled
    auto dataset = xt::data::datasets::CIFAR100("/home/kami/Documents/temp/", DataMode::TRAIN, true);

    // Apply transformations to the dataset:
    // 1. Resize images to 224x224
    // 2. Normalize with mean 0.5 and std 0.5
    // 3. Stack the transformed data into batches
    auto transformed_dataset = dataset
            .map(xt::data::transforms::resize(size))
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());

    // Create a data loader with sequential sampling and batch size of 64
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(transformed_dataset), 64);

    // Initialize ResNet18 model with layer configuration {3, 4, 6, 3} for 100 classes (CIFAR-100) with 3 input channels (RGB)
    xt::models::ResNet18 model({3, 4, 6, 3}, 100, 3);

    // Move model to the selected device (CPU or CUDA)
    model.to(device);

    // Set model to training mode
    model.train();

    // Set up Adam optimizer with learning rate of 1e-3
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    // Train the model for 10 epochs
    for (size_t epoch = 0; epoch != 10; ++epoch) {
        // Track batch index for logging
        size_t batch_index = 0;

        // Get iterators for the data loader
        auto train_loader_iterator = train_loader->begin();
        auto train_loader_end = train_loader->end();

        // Iterate over batches in the current epoch
        while (train_loader_iterator != train_loader_end) {
            // Declare tensors for input data and targets
            torch::Tensor data, targets;

            // Get the current batch
            auto batch = *train_loader_iterator;

            // Extract data (images) and targets (labels) from the batch
            data = batch.data;
            targets = batch.target;

            // Move data and targets to the selected device
            data = data.to(device);
            targets = targets.to(device);

            // Zero out gradients from previous iteration
            optimizer.zero_grad();

            // Forward pass: compute model output
            torch::Tensor output;
            output = model.forward(data);

            // Compute loss using negative log likelihood
            torch::Tensor loss;
            loss = torch::nll_loss(output, targets);

            // Backward pass: compute gradients
            loss.backward();

            // Update model parameters using the optimizer
            optimizer.step();

            // Log training progress every 100 batches
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>()
                          << std::endl;
            }

            // Move to the next batch
            ++train_loader_iterator;
        }
    }

    // Return 0 to indicate successful execution
    return 0;
}