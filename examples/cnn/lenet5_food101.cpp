#include "includes/base.h"
#include "../../include/datasets/computer_vision/image_classification/food.h"
#include "../../include/models/computer_vision/image_classification/lenet5.h"
#include "../../include/definitions/transforms.h"

using namespace std;
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

/**
 * @brief Main function to train a LeNet5 model on the Food101 dataset using PyTorch.
 *
 * This program loads the Food101 dataset, applies transformations (resize and normalization),
 * creates a data loader with sequential sampling, initializes a LeNet5 model, sets up an Adam optimizer,
 * and trains the model for 10 epochs using negative log likelihood loss. Training progress is logged every 100 batches.
 * Debug print statements are included to track dataset loading, transformation, and data loader creation.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
    // Set precision for floating-point output to 10 decimal places
    std::cout.precision(10);

    // Define image size for resizing (32x32 as suitable for LeNet5)
    std::vector<int64_t> size = {32, 32};

    // Select device (CUDA if available, otherwise CPU)
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Load Food101 training dataset
    // - Path: /home/kami/Documents/temp/
    // - Mode: Training
    // - Download: Disabled
    auto dataset = xt::data::datasets::Food101("/home/kami/Documents/temp/", DataMode::TRAIN, false);

    // Print debug message to confirm dataset loading
    cout << "DATASET" << endl;

    // Commented-out alternative dataset (Imagenette, not used)
    // auto dataset = torch::ext::data::datasets::Imagenette("/home/kami/Documents/temp/", DataMode::TRAIN, true, torch::ext::data::datasets::ImageType::PX160);

    // Apply transformations to the dataset:
    // 1. Resize images to 32x32
    // 2. Normalize with mean 0.5 and std 0.5
    // 3. Stack the transformed data into batches
    auto transformed_dataset = dataset
            .map(xt::data::transforms::resize(size))
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());

    // Print debug message to confirm dataset transformation
    cout << "TRANSFORMED DATASET" << endl;

    // Create a data loader with sequential sampling and batch size of 64
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(transformed_dataset), 64);

    // Print debug message to confirm data loader creation
    cout << "DATALOADER" << endl;

    // Initialize LeNet5 model for 101 classes (Food101) with 3 input channels (RGB)
    xt::models::LeNet5 model(101, 3);

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