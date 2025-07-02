#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <random>

// Struct to hold a batch of data: input tensors and target tensors
struct Batch {
    torch::Tensor inputs;
    torch::Tensor targets;
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
    std::vector<std::pair<torch::Tensor, torch::Tensor>> dataset_; // The dataset
    int batch_size_;                                               // Size of each batch
    bool shuffle_;                                                 // Whether to shuffle data
    std::vector<size_t> indices_;                                  // Shuffled or ordered indices
    size_t num_batches_;                                           // Total number of batches
};

// Iterator class for DataLoader
class DataLoaderIterator {
public:
    // Constructor: takes a pointer to the DataLoader and the current batch index
    DataLoaderIterator(DataLoader* loader, size_t batch_idx)
        : loader_(loader), batch_idx_(batch_idx) {}

    // Dereference operator: returns the current batch
    Batch operator*() const {
        size_t start = batch_idx_ * loader_->batch_size_;
        size_t end = std::min(start + loader_->batch_size_, loader_->dataset_.size());
        std::vector<torch::Tensor> inputs;
        std::vector<torch::Tensor> targets;
        // Collect tensors for the current batch
        for (size_t i = start; i < end; ++i) {
            size_t idx = loader_->indices_[i];
            inputs.push_back(loader_->dataset_[idx].first);
            targets.push_back(loader_->dataset_[idx].second);
        }
        // Stack the tensors into a single tensor for inputs and targets
        torch::Tensor input_batch = torch::stack(inputs);
        torch::Tensor target_batch = torch::stack(targets);
        return {input_batch, target_batch};
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
    DataLoader* loader_;  // Pointer to the parent DataLoader
    size_t batch_idx_;    // Current batch index
};

// Implementation of begin() and end() methods
DataLoaderIterator DataLoader::begin() {
    return DataLoaderIterator(this, 0);
}

DataLoaderIterator DataLoader::end() {
    return DataLoaderIterator(this, num_batches_);
}

// Example usage
int main() {
    // Create a sample dataset
    std::vector<std::pair<torch::Tensor, torch::Tensor>> dataset;
    for (int i = 0; i < 10; ++i) {
        torch::Tensor input = torch::ones({2, 2}) * i;
        torch::Tensor target = torch::tensor(i);
        dataset.emplace_back(input, target);
    }

    // Initialize the DataLoader with batch size 4 and shuffling enabled
    DataLoader loader(dataset, 4, true);

    // Iterate over batches
    for (auto& batch : loader) {
        torch::Tensor inputs = batch.inputs;
        torch::Tensor targets = batch.targets;
        std::cout << "Inputs:\n" << inputs << "\n";
        std::cout << "Targets:\n" << targets << "\n\n";
    }

    return 0;
}