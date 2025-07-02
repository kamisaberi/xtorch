#pragma once

#include <iostream>
#include <torch/torch.h>
#include <torch/data/datasets.h> // For torch::data::Dataset
#include <torch/data/example.h>  // For torch::data::Example
#include <vector>
#include <string>
#include <algorithm> // For std::shuffle, std::iota, std::min, std::max
#include <random>    // For std::default_random_engine, std::random_device
#include <iostream>
#include <thread>
#include <future>    // Not directly used in V2, but good to keep in mind for other patterns
#include <mutex>
#include <condition_variable>
#include <deque>      // For the batch queue
#include <atomic>     // For atomic counters and flags
#include <optional>   // For torch::optional and std::optional
#include <stdexcept>  // For std::out_of_range, std::runtime_error, std::invalid_argument
#include <chrono>
#include "../base/base.h"


namespace xt::dataloaders
{
    using BatchData = std::pair<torch::Tensor, torch::Tensor>;

    class ExtendedDataLoader
    {
    public:
        ExtendedDataLoader(xt::datasets::Dataset& dataset, size_t batch_size, bool shuffle, size_t num_workers,
                             size_t prefetch_factor = 2); // How many batches per worker to aim for in queue

        ~ExtendedDataLoader();
        void shutdown();
        void reset_epoch();
        std::optional<BatchData> next_batch();

    private:
        // This is the function executed by each worker thread
        void worker_loop(size_t worker_id);

        // Helper to fetch samples for a given batch_overall_idx and collate them.
        std::optional<BatchData> produce_batch(size_t batch_overall_idx);

    public: // Iterator support
        class Iterator
        {
        public:
            Iterator(ExtendedDataLoader* loader, bool end = false);
            const BatchData& operator*() const;
            BatchData& operator*();
            Iterator& operator++();
            bool operator!=(const Iterator& other) const;

        private:
            ExtendedDataLoader* loader_;
            bool is_end_;
            std::optional<BatchData> current_batch_opt_; // Cache for current batch
        };

        Iterator begin();

        Iterator end();

    private:
        xt::datasets::Dataset& dataset_;
        size_t batch_size_;
        bool shuffle_;
        size_t num_workers_;
        size_t prefetch_queue_max_size_;
        size_t current_dataset_size_; // Cache dataset size for the epoch

        std::vector<size_t> indices_; // Shuffled indices for the current epoch

        std::deque<BatchData> prefetched_batch_queue_;
        std::mutex queue_mutex_;
        std::condition_variable data_available_cv_; // Main thread waits on this
        std::condition_variable space_available_cv_; // Workers wait on this

        std::vector<std::thread> worker_threads_;
        std::atomic<bool> shutdown_workers_;
        std::atomic<bool> epoch_ended_for_workers_; // True when all batches are produced/claimed by workers

        std::atomic<size_t> next_batch_idx_to_produce_;
        size_t total_batches_in_epoch_;
        size_t batches_consumed_in_epoch_; // For debugging/tracking
    };
}
