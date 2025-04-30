#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

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
    // Constructor: takes a dataset, batch size, shuffle flag, and number of workers
    DataLoader(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset,
               int batch_size,
               bool shuffle = false,
               int num_workers = 0)
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          num_workers_(std::max(0, num_workers)),
          stop_workers_(false),
          current_batch_idx_(0) {
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

        // Start worker threads
        if (num_workers_ > 0) {
            for (int i = 0; i < num_workers_; ++i) {
                workers_.emplace_back(&DataLoader::worker_thread, this);
            }
        }
    }

    // Destructor: stop workers and join threads
    ~DataLoader() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_workers_ = true;
        }
        queue_cond_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Methods to get iterators for iteration support
    DataLoaderIterator begin();
    DataLoaderIterator end();

private:
    // Allow DataLoaderIterator to access private members
    friend class DataLoaderIterator;

    // Worker thread function to pre-fetch batches
    void worker_thread() {
        while (true) {
            size_t batch_idx;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                // Check if we should stop or if all batches are processed
                if (stop_workers_ || current_batch_idx_ >= num_batches_) {
                    return;
                }
                batch_idx = current_batch_idx_++;
                lock.unlock();
            }

            // Construct the batch
            size_t start = batch_idx * batch_size_;
            size_t end = std::min(start + batch_size_, dataset_.size());
            std::vector<torch::Tensor> inputs_vec;
            std::vector<torch::Tensor> targets_vec;
            for (size_t i = start; i < end; ++i) {
                size_t idx = indices_[i];
                inputs_vec.push_back(dataset_[idx].first.clone());  // Clone to avoid memory issues
                targets_vec.push_back(dataset_[idx].second.clone());
            }
            torch::Tensor inputs = torch::stack(inputs_vec);
            torch::Tensor targets = torch::stack(targets_vec);
            Batch batch{inputs, targets};

            // Push the batch to the queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                batch_queue_.push(std::move(batch));
                lock.unlock();
                queue_cond_.notify_one();
            }
        }
    }

    // Get the next batch (used by iterator)
    Batch get_next_batch() {
        if (num_workers_ == 0) {
            // Single-threaded: construct batch directly
            size_t batch_idx = current_batch_idx_++;
            if (batch_idx >= num_batches_) {
                return {}; // Return empty batch to signal end
            }
            size_t start = batch_idx * batch_size_;
            size_t end = std::min(start + batch_size_, dataset_.size());
            std::vector<torch::Tensor> inputs_vec;
            std::vector<torch::Tensor> targets_vec;
            for (size_t i = start; i < end; ++i) {
                size_t idx = indices_[i];
                inputs_vec.push_back(dataset_[idx].first);
                targets_vec.push_back(dataset_[idx].second);
            }
            return {torch::stack(inputs_vec), torch::stack(targets_vec)};
        } else {
            // Multi-threaded: fetch from queue
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cond_.wait(lock, [this] {
                return !batch_queue_.empty() || stop_workers_ || current_batch_idx_ >= num_batches_;
            });
            if (batch_queue_.empty()) {
                return {}; // Return empty batch to signal end
            }
            Batch batch = std::move(batch_queue_.front());
            batch_queue_.pop();
            return batch;
        }
    }

    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset_; // Reference to the dataset
    int batch_size_;                                                     // Size of each batch
    bool shuffle_;                                                       // Whether to shuffle data
    int num_workers_;                                                    // Number of worker threads
    std::vector<size_t> indices_;                                        // Shuffled or ordered indices
    size_t num_batches_;                                                 // Total number of batches
    std::vector<std::thread> workers_;                                   // Worker threads
    std::queue<Batch> batch_queue_;                                      // Thread-safe batch queue
    std::mutex queue_mutex_;                                             // Mutex for queue access
    std::condition_variable queue_cond_;                                 // Condition variable for queue
    std::atomic<bool> stop_workers_;                                     // Flag to stop workers
    std::atomic<size_t> current_batch_idx_;                              // Current batch index
};

// Iterator class for DataLoader
class DataLoaderIterator {
public:
    DataLoaderIterator(DataLoader* loader, size_t batch_idx)
        : loader_(loader), batch_idx_(batch_idx) {
        // Pre-fetch the first batch
        if (batch_idx_ < loader_->num_batches_) {
            current_batch_ = loader_->get_next_batch();
        }
    }

    Batch operator*() const {
        return current_batch_;
    }

    DataLoaderIterator& operator++() {
        if (batch_idx_ < loader_->num_batches_) {
            ++batch_idx_;
            if (batch_idx_ < loader_->num_batches_) {
                current_batch_ = loader_->get_next_batch();
            }
        }
        return *this;
    }

    bool operator!=(const DataLoaderIterator& other) const {
        return batch_idx_ != other.batch_idx_;
    }

private:
    DataLoader* loader_;
    size_t batch_idx_;
    Batch current_batch_;
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

    // Create the custom DataLoader with batch size 64, shuffling, and 4 workers
    int batch_size = 64;
    bool shuffle = true;
    int num_workers = 32; // Adjust based on your CPU
    DataLoader loader(data, batch_size, shuffle, num_workers);

    auto start = std::chrono::high_resolution_clock::now();
    // Iterate over batches and print the shape of the first batch
    int count = 0;
    for (const auto& batch : loader) {
        torch::Tensor inputs = batch.inputs;
        torch::Tensor targets = batch.targets;
        std::cout << "Batch inputs shape: " << inputs.sizes() << "\n";
        std::cout << "Batch targets shape: " << targets.sizes() << "\n";
        // if (++count >= 1) break; // Print only the first batch
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << duration << " s\n";
    return 0;
}