#pragma once

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <random>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>


// Struct to hold a batch of data: input tensors and target tensors
struct Batch
{
    torch::Tensor inputs; // Batched images, e.g., shape [batch_size, 1, 28, 28]
    torch::Tensor targets; // Batched labels, e.g., shape [batch_size]
};

// Forward declaration of the iterator class
class DataLoaderIterator;

// Main DataLoader class
class CustomDataLoader
{
public:
    // Constructor: takes a dataset, batch size, shuffle flag, and number of workers
    CustomDataLoader(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset,
               int batch_size,
               bool shuffle = false,
               int num_workers = 0);

    // Destructor: stop workers and join threads
    ~CustomDataLoader();

    DataLoaderIterator begin();
    DataLoaderIterator end();

private:
    friend class DataLoaderIterator;
    // Worker thread function to pre-fetch batches
    void worker_thread();
    // Get the next batch (used by iterator)
    Batch get_next_batch();
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset_; // Reference to the dataset
    int batch_size_; // Size of each batch
    bool shuffle_; // Whether to shuffle data
    int num_workers_; // Number of worker threads
    std::vector<size_t> indices_; // Shuffled or ordered indices
    size_t num_batches_; // Total number of batches
    std::vector<std::thread> workers_; // Worker threads
    std::queue<Batch> batch_queue_; // Thread-safe batch queue
    std::mutex queue_mutex_; // Mutex for queue access
    std::condition_variable queue_cond_; // Condition variable for queue
    std::atomic<bool> stop_workers_; // Flag to stop workers
    std::atomic<size_t> current_batch_idx_; // Current batch index
};

// Iterator class for DataLoader
class DataLoaderIterator
{
public:
    DataLoaderIterator(CustomDataLoader* loader, size_t batch_idx);

    Batch operator*() const;

    DataLoaderIterator& operator++();

    bool operator!=(const DataLoaderIterator& other) const;

private:
    CustomDataLoader* loader_;
    size_t batch_idx_;
    Batch current_batch_;
};
