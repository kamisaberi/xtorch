#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// Custom transform to resize a single image tensor to 32x32 using bilinear interpolation.
struct ResizeTransform : public torch::data::transforms::Transform<torch::Tensor, torch::Tensor> {
    torch::Tensor operator()(torch::Tensor input)  {
        // MNIST images are 1x28x28 (C x H x W). Add a batch dimension for interpolation.
        input = input.unsqueeze(0);  // shape: [1, 1, 28, 28]
        input = torch::nn::functional::interpolate(
            input,
            torch::nn::functional::InterpolateFuncOptions().size({32, 32}).mode(torch::kBilinear).align_corners(false)
        );
        input = input.squeeze(0);    // remove batch dim, back to [1, 32, 32]
        return input;
    }
};

// Custom DataLoader inheriting from DataLoaderBase to iterate over the dataset.
template <typename Dataset>
class CustomDataLoader : public torch::data::DataLoaderBase<CustomDataLoader<Dataset>, Dataset> {
public:
    using BatchType = torch::data::Example<>;
    // Constructor: store reference to dataset and initialize index sequence.
    CustomDataLoader(Dataset& dataset, size_t batch_size, bool shuffle = true)
        : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle)
    {
        dataset_size_ = dataset_.size().value();               // total number of samples
        indices_.reserve(dataset_size_);
        for (size_t i = 0; i < dataset_size_; ++i) {
            indices_.push_back(i);
        }
        if (shuffle_) {
            // Shuffle indices for the first epoch
            std::mt19937 gen(std::random_device{}());
            std::shuffle(indices_.begin(), indices_.end(), gen);
        }
        current_index_ = 0;
    }

    // Reset loader at end of epoch: shuffle indices and reset counter.
    void reset() override {
        current_index_ = 0;
        if (shuffle_) {
            std::mt19937 gen(std::random_device{}());
            std::shuffle(indices_.begin(), indices_.end(), gen);
        }
    }

    // Fetch the next batch. Returns an optional Example (data=Tensor, target=Tensor).
    torch::optional<BatchType> next() override {
        if (current_index_ >= dataset_size_) {
            // No more data
            return torch::nullopt;
        }
        // Determine the range of indices for the next batch
        size_t start = current_index_;
        size_t end = std::min(current_index_ + batch_size_, dataset_size_);
        current_index_ = end;

        // Gather a batch of examples from the dataset
        std::vector<torch::Tensor> data_batch;
        std::vector<torch::Tensor> target_batch;
        data_batch.reserve(end - start);
        target_batch.reserve(end - start);

        for (size_t i = start; i < end; ++i) {
            // Get the sample at the index (this applies all dataset transforms)
            auto sample = dataset_.get(indices_[i]);
            data_batch.push_back(sample.data);
            target_batch.push_back(sample.target);
        }

        // Stack the individual tensors into batch tensors
        torch::Tensor stacked_data = torch::stack(data_batch, /*dim=*/0);
        torch::Tensor stacked_targets = torch::stack(target_batch, /*dim=*/0).squeeze();
        // .squeeze() to convert targets from shape [batch_size, 1] to [batch_size] if needed

        return BatchType(stacked_data, stacked_targets);
    }

private:
    Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    size_t dataset_size_;
    std::vector<size_t> indices_;
    size_t current_index_;
};

// Define a simple CNN model (two conv layers + one hidden fully-connected layer).
struct NetImpl : torch::nn::Module {
    // Layers: conv1, conv2, fc1, fc2
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    NetImpl() {
        // Initialize convolutional layers
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 8, /*kernel_size=*/3).stride(1).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, /*kernel_size=*/3).stride(1).padding(1)));
        // After two 2x2 poolings, the 32x32 input is reduced to 8x8, with 16 channels
        fc1   = register_module("fc1", torch::nn::Linear(16 * 8 * 8, 64));
        fc2   = register_module("fc2", torch::nn::Linear(64, 10));
    }

    // Implement the forward pass
    torch::Tensor forward(torch::Tensor x) {
        // Input x shape: [batch_size, 1, 32, 32]
        x = torch::relu(conv1(x));
        x = torch::max_pool2d(x, /*kernel_size=*/2);  // -> shape [batch, 8, 16, 16]
        x = torch::relu(conv2(x));
        x = torch::max_pool2d(x, /*kernel_size=*/2);  // -> shape [batch, 16, 8, 8]
        x = x.view({x.size(0), -1});                  // flatten to [batch, 16*8*8]
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x;
    }
};
TORCH_MODULE(Net);  // This creates Net (alias for std::shared_ptr<NetImpl>)

// Entry point
int main() {
    // Set device to CPU (as required)
    torch::Device device(torch::kCPU);

    // Set random seed for reproducibility (optional)
    torch::manual_seed(0);

    // **Load and transform the MNIST dataset**
    // Adjust the `data_path` to the location of your MNIST dataset (unzipped files).
    const std::string data_path = "./data/mnist";  // assume MNIST data is here
    auto train_dataset = torch::data::datasets::MNIST(data_path)
        .map(ResizeTransform())                             // resize images to 32x32
        .map(torch::data::transforms::Normalize<>(0.5, 0.5));  // normalize to mean=0.5, std=0.5
        // Note: We will stack samples into batches using the CustomDataLoader.

    // Create an instance of the custom data loader for the training set
    const size_t batch_size = 64;
    CustomDataLoader<decltype(train_dataset)> train_loader(train_dataset, batch_size, /*shuffle=*/true);

    // Initialize the neural network, loss function, and optimizer
    Net model;
    model->to(device);  // move model to CPU (redundant here since device is CPU)

    torch::nn::CrossEntropyLoss criterion;
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));

    // **Training loop for 10 epochs**
    size_t num_epochs = 10;
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        model->train();  // set model to training mode (important if using dropout/batch-norm)
        double total_loss = 0.0;
        size_t total_correct = 0;
        size_t total_samples = 0;

        // Iterate over batches from the data loader
        while (true) {
            auto batch = train_loader.next();
            if (!batch.has_value()) {
                break;  // no more batches
            }
            // Get data and targets from the batch, and move them to the device (CPU here)
            torch::Tensor data = batch->data.to(device);
            torch::Tensor targets = batch->target.to(device);

            // Forward pass: compute model predictions
            torch::Tensor output = model->forward(data);
            // Compute the loss between predictions and true labels
            torch::Tensor loss = criterion(output, targets);

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // Accumulate training loss and accuracy statistics
            total_loss += loss.item<double>() * data.size(0);  // sum of loss for this batch
            // Compute number of correct predictions in this batch
            auto pred = output.argmax(1);  // index of max log-probability
            total_correct += pred.eq(targets).sum().item<int64_t>();
            total_samples += data.size(0);
        }

        // Compute average loss and accuracy for the epoch
        double avg_loss = total_loss / total_samples;
        double accuracy = static_cast<double>(total_correct) / total_samples * 100.0;

        // Print epoch statistics
        std::cout << "Epoch [" << epoch << "/" << num_epochs << "] - "
                  << "Loss: " << avg_loss << " , "
                  << "Accuracy: " << accuracy << "%\n";

        // Reset the data loader for the next epoch (shuffles the data if enabled)
        train_loader.reset();
    }

    return 0;
}
