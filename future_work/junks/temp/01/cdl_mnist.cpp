#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

// Struct to hold a batch of data: input tensors and target tensors
struct Batch {
    torch::Tensor inputs;  // Batched images, e.g., shape [batch_size, 1, 28, 28]
    torch::Tensor targets; // Batched labels, e.g., shape [batch_size]
};

// Forward declaration of the iterator class
class DataLoaderIterator;

// Main DataLoader class
class DataLoader {
public:
    // Constructor: takes a dataset, batch size, and shuffle flag
    DataLoader(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset,
               int batch_size,
               bool shuffle = false)
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle) {
        // Initialize indices from 0 to dataset size - 1
        indices_.resize(dataset_.size());
        for (size_t i = 0; i < indices_.size(); ++i) {
            indices_[i] = i;
        }
        // Shuffle indices if shuffle is enabled
        if (shuffle_) {
            std::random_device rd;
            std::default_random_engine engine(rd());
            std::shuffle(indices_.begin(), indices_.end(), engine);
        }
        // Calculate the number of batches
        num_batches_ = (dataset_.size() + batch_size_ - 1) / batch_size_;
    }

    // Methods to get iterators for iteration support
    DataLoaderIterator begin();
    DataLoaderIterator end();

private:
    // Allow DataLoaderIterator to access private members
    friend class DataLoaderIterator;
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset_; // Reference to the dataset
    int batch_size_;                                                     // Size of each batch
    bool shuffle_;                                                       // Whether to shuffle data
    std::vector<size_t> indices_;                                        // Shuffled or ordered indices
    size_t num_batches_;                                                 // Total number of batches
};

// Iterator class for DataLoader
class DataLoaderIterator {
public:
    // Constructor: takes a pointer to the DataLoader and the current batch index
    DataLoaderIterator(const DataLoader* loader, size_t batch_idx)
        : loader_(loader), batch_idx_(batch_idx) {}

    // Dereference operator: returns the current batch
    Batch operator*() const {
        // Calculate the start and end indices for the current batch
        size_t start = batch_idx_ * loader_->batch_size_;
        size_t end = std::min(start + loader_->batch_size_, loader_->dataset_.size());

        // Collect tensors for the current batch
        std::vector<torch::Tensor> inputs_vec;
        std::vector<torch::Tensor> targets_vec;
        for (size_t i = start; i < end; ++i) {
            size_t idx = loader_->indices_[i];
            inputs_vec.push_back(loader_->dataset_[idx].first);  // Image tensor
            targets_vec.push_back(loader_->dataset_[idx].second); // Label tensor
        }

        // Stack the tensors into a single tensor for inputs and targets
        torch::Tensor inputs = torch::stack(inputs_vec);
        torch::Tensor targets = torch::stack(targets_vec);
        return {inputs, targets};
    }

    // Increment operator: moves to the next batch
    DataLoaderIterator& operator++() {
        ++batch_idx_;
        return *this;
    }

    // Inequality operator: checks if the iterator has reached the end
    bool operator!=(const DataLoaderIterator& other) const {
        return batch_idx_ != other.batch_idx_;
    }

private:
    const DataLoader* loader_;  // Pointer to the parent DataLoader
    size_t batch_idx_;          // Current batch index
};

// Implementation of begin() and end() methods
DataLoaderIterator DataLoader::begin() {
    return DataLoaderIterator(this, 0);
}

DataLoaderIterator DataLoader::end() {
    return DataLoaderIterator(this, num_batches_);
}

// Main function demonstrating usage with MNIST dataset
int main() {
    // Load the MNIST training dataset
    auto mnist_dataset = torch::data::datasets::MNIST("/home/kami/Documents/datasets/MNIST/raw/");

    // Collect data into a vector of pairs (image tensor, label tensor)
    std::vector<std::pair<torch::Tensor, torch::Tensor>> data;
    for (size_t i = 0; i < mnist_dataset.size().value(); ++i) {
        auto example = mnist_dataset.get(i);
        data.emplace_back(example.data, example.target);
    }

    // Create the custom DataLoader with batch size 64 and shuffling enabled
    int batch_size = 64;
    bool shuffle = true;
    DataLoader loader(data, batch_size, shuffle);

    // Iterate over batches and print the shape of the first batch
    int count = 0;
    for (const auto& batch : loader) {
        torch::Tensor inputs = batch.inputs;
        torch::Tensor targets = batch.targets;
        std::cout << "Batch inputs shape: " << inputs.sizes() << "\n";
        std::cout << "Batch targets shape: " << targets.sizes() << "\n";
        if (++count >= 1) break;  // Print only the first batch to avoid flooding output
    }

    return 0;
}