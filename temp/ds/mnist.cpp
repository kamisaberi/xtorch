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
#include <chrono>     // For timing in main

#include "../../include/datasets/computer_vision/image_classification/mnist.h"

using namespace std;


using BatchData = std::pair<torch::Tensor, torch::Tensor>;

// --- MyCustomDataLoaderV2 (Multi-worker Prefetching DataLoader) ---
class MyCustomDataLoaderV2
{
public:
    MyCustomDataLoaderV2(xt::datasets::Dataset& dataset,
                         size_t batch_size,
                         bool shuffle,
                         size_t num_workers,
                         size_t prefetch_factor = 2) // How many batches per worker to aim for in queue
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          num_workers_(num_workers),
          prefetch_queue_max_size_(std::max(size_t(1), num_workers * prefetch_factor)),
          // Ensure queue size is at least 1
          shutdown_workers_(false),
          epoch_ended_for_workers_(false), // True when workers have processed/claimed all batches for current epoch
          current_dataset_size_(0), // Will be initialized in constructor
          total_batches_in_epoch_(0),
          next_batch_idx_to_produce_(0), // Atomic counter for workers to claim batch tasks
          batches_consumed_in_epoch_(0)
    {
        if (batch_size_ == 0)
        {
            throw std::invalid_argument("Batch size cannot be zero.");
        }

        auto dataset_size_opt = dataset_.size();
        if (!dataset_size_opt.has_value())
        {
            // This DataLoader design requires a dataset with a known size.
            throw std::runtime_error(
                "Dataset size is unknown (optional is nullopt). This DataLoader requires a known size.");
        }
        current_dataset_size_ = dataset_size_opt.value();

        if (current_dataset_size_ > 0)
        {
            indices_.resize(current_dataset_size_);
            std::iota(indices_.begin(), indices_.end(), 0); // Fill with 0, 1, 2, ...
        }
        else
        {
            indices_.clear(); // Handle empty dataset case
        }
        // reset_epoch() will be called by begin() to shuffle (if needed) and start workers.
    }

    ~MyCustomDataLoaderV2()
    {
        shutdown();
    }

    void shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            shutdown_workers_ = true; // Signal workers to stop
            epoch_ended_for_workers_ = true; // Ensure workers don't get stuck waiting for epoch end
        }
        // Notify all potentially waiting threads (workers or main thread)
        space_available_cv_.notify_all();
        data_available_cv_.notify_all();

        for (auto& worker : worker_threads_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        worker_threads_.clear();

        // Clear the queue to release tensor resources
        std::lock_guard<std::mutex> lock(queue_mutex_);
        prefetched_batch_queue_.clear(); // std::deque::clear calls destructors of elements
    }

    void reset_epoch()
    {
        // 1. Stop and join existing workers if any (e.g., from a previous epoch)
        shutdown(); // This sets shutdown_workers_ and joins threads

        // 2. Reset internal state for the new epoch
        shutdown_workers_ = false; // Allow new workers to run
        epoch_ended_for_workers_ = false;
        batches_consumed_in_epoch_ = 0;
        next_batch_idx_to_produce_ = 0; // Reset for workers to pick tasks from batch 0

        // Re-fetch dataset size in case it could change (not typical for this example)
        auto dataset_size_opt = dataset_.size();
        if (!dataset_size_opt.has_value())
        {
            throw std::runtime_error("Dataset size became unknown during reset_epoch.");
        }
        current_dataset_size_ = dataset_size_opt.value();

        if (current_dataset_size_ == 0)
        {
            total_batches_in_epoch_ = 0;
            indices_.clear();
            return; // No data, nothing for workers to do
        }

        total_batches_in_epoch_ = (current_dataset_size_ + batch_size_ - 1) / batch_size_;

        // Ensure indices_ vector is correctly sized and initialized
        if (indices_.size() != current_dataset_size_)
        {
            indices_.resize(current_dataset_size_);
            std::iota(indices_.begin(), indices_.end(), 0);
        }

        if (shuffle_)
        {
            std::random_device rd;
            std::mt19937 g(rd()); // Standard mersenne_twister_engine seeded with rd()
            std::shuffle(indices_.begin(), indices_.end(), g);
        }

        // Clear any stale batches from the queue (should be empty after shutdown, but good practice)
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            prefetched_batch_queue_.clear();
        }

        // 3. Start new worker threads if num_workers > 0 and there's work to do
        if (num_workers_ > 0 && total_batches_in_epoch_ > 0)
        {
            for (size_t i = 0; i < num_workers_; ++i)
            {
                worker_threads_.emplace_back(&MyCustomDataLoaderV2::worker_loop, this, i);
            }
        }
    }

    std::optional<BatchData> next_batch()
    {
        if (current_dataset_size_ == 0)
        {
            // Handle empty dataset upfront
            return std::nullopt;
        }

        // Synchronous path for num_workers = 0
        if (num_workers_ == 0)
        {
            if (next_batch_idx_to_produce_ >= total_batches_in_epoch_)
            {
                return std::nullopt; // End of epoch
            }
            size_t current_batch_overall_idx = next_batch_idx_to_produce_++;
            auto batch = produce_batch(current_batch_overall_idx);
            if (batch) batches_consumed_in_epoch_++;
            return batch;
        }

        // Asynchronous path for num_workers > 0
        std::unique_lock<std::mutex> lock(queue_mutex_);
        // Wait until data is available OR all batches for the epoch are produced AND the queue is empty
        data_available_cv_.wait(lock, [this]
        {
            return !prefetched_batch_queue_.empty() ||
                (epoch_ended_for_workers_ && prefetched_batch_queue_.empty());
        });

        if (prefetched_batch_queue_.empty())
        {
            // This means epoch_ended_for_workers_ is true and queue is empty.
            // All batches have been produced and consumed for this epoch.
            return std::nullopt;
        }

        BatchData batch = prefetched_batch_queue_.front();
        prefetched_batch_queue_.pop_front();
        batches_consumed_in_epoch_++;

        lock.unlock();
        space_available_cv_.notify_one(); // Notify one worker that space is available in the queue

        return batch;
    }

private:
    // This is the function executed by each worker thread
    void worker_loop(size_t worker_id)
    {
        // std::cout << "Worker " << worker_id << " started." << std::endl;
        try
        {
            while (true)
            {
                if (shutdown_workers_.load(std::memory_order_relaxed))
                {
                    break; // Exit if global shutdown is requested
                }

                // Atomically get the index of the next batch to produce
                size_t current_batch_overall_idx = next_batch_idx_to_produce_.fetch_add(1, std::memory_order_relaxed);

                if (current_batch_overall_idx >= total_batches_in_epoch_)
                {
                    // This worker (and possibly others) have claimed all batches.
                    next_batch_idx_to_produce_.fetch_sub(1, std::memory_order_relaxed); // Correct over-increment

                    std::unique_lock<std::mutex> epoch_lock(queue_mutex_); // Lock to safely check/set epoch_ended
                    // Check if all batches are truly accounted for by next_batch_idx_to_produce_
                    if (next_batch_idx_to_produce_.load(std::memory_order_relaxed) >= total_batches_in_epoch_)
                    {
                        epoch_ended_for_workers_ = true;
                    }

                    if (epoch_ended_for_workers_)
                    {
                        data_available_cv_.notify_all(); // Wake main thread if it's waiting for the last batch
                    }
                    // Wait for a new epoch (epoch_ended_for_workers_ becomes false) or shutdown
                    space_available_cv_.wait(epoch_lock, [this]
                    {
                        return shutdown_workers_.load() || !epoch_ended_for_workers_.load();
                    });

                    if (shutdown_workers_.load()) break; // Exit if shutdown during wait
                    if (!epoch_ended_for_workers_.load())
                    {
                        // New epoch has started
                        // std::cout << "Worker " << worker_id << " detected new epoch." << std::endl;
                        continue; // Loop again to get a new batch index for the new epoch
                    }
                    // If still here, it might be a spurious wakeup or still end of epoch with no shutdown
                    continue;
                }

                // Produce the batch (this can take time, e.g., disk I/O, augmentations)
                auto batch_opt = produce_batch(current_batch_overall_idx);

                if (batch_opt)
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    // Wait if the prefetch queue is full
                    space_available_cv_.wait(lock, [this, worker_id]
                    {
                        return prefetched_batch_queue_.size() < prefetch_queue_max_size_ || shutdown_workers_.load();
                    });

                    if (shutdown_workers_.load())
                    {
                        break; // Exit if shutdown while waiting to push
                    }

                    prefetched_batch_queue_.push_back(*batch_opt);
                    // std::cout << "Worker " << worker_id << " pushed batch " << current_batch_overall_idx << ". Queue size: " << prefetched_batch_queue_.size() << std::endl;
                    lock.unlock();
                    data_available_cv_.notify_one(); // Notify main thread that a new batch is available
                }
                else
                {
                    // This case (batch_opt is nullopt) might occur if produce_batch failed
                    // or if total_batches_in_epoch was miscalculated (should be rare with current logic).
                    std::cerr << "Worker " << worker_id << " failed to produce batch " << current_batch_overall_idx <<
                        " or it was empty." << std::endl;
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "!!! Exception in worker " << worker_id << ": " << e.what() << std::endl;
            // Critical error in a worker. Signal global shutdown to prevent hangs.
            std::unique_lock<std::mutex> lock(queue_mutex_);
            shutdown_workers_ = true;
            epoch_ended_for_workers_ = true; // Treat epoch as done to unblock
            data_available_cv_.notify_all();
            space_available_cv_.notify_all();
        }
        // std::cout << "Worker " << worker_id << " finished." << std::endl;
        // Ensure main thread isn't stuck if this was the last active worker and epoch ended
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (next_batch_idx_to_produce_.load(std::memory_order_relaxed) >= total_batches_in_epoch_)
        {
            epoch_ended_for_workers_ = true;
        }
        if (epoch_ended_for_workers_ && prefetched_batch_queue_.empty())
        {
            data_available_cv_.notify_all();
        }
    }

    // Helper to fetch samples for a given batch_overall_idx and collate them.
    std::optional<BatchData> produce_batch(size_t batch_overall_idx)
    {
        size_t start_sample_idx_in_indices_vec = batch_overall_idx * batch_size_;

        // This check should ideally be redundant if total_batches_in_epoch_ is correct
        if (start_sample_idx_in_indices_vec >= current_dataset_size_ && current_dataset_size_ > 0)
        {
            std::cerr << "Warning: produce_batch called for batch_overall_idx " << batch_overall_idx
                << " which is out of bounds for current_dataset_size_ " << current_dataset_size_ << std::endl;
            return std::nullopt;
        }
        if (current_dataset_size_ == 0) return std::nullopt;


        size_t actual_batch_size = std::min(batch_size_, current_dataset_size_ - start_sample_idx_in_indices_vec);
        if (actual_batch_size == 0)
        {
            // Can happen if dataset_size is 0 or for last batch if miscalculated
            return std::nullopt;
        }

        std::vector<torch::Tensor> batch_features_list;
        std::vector<torch::Tensor> batch_labels_list;
        batch_features_list.reserve(actual_batch_size);
        batch_labels_list.reserve(actual_batch_size);

        for (size_t i = 0; i < actual_batch_size; ++i)
        {
            // Get the true index into the dataset from our (potentially shuffled) indices_ vector
            size_t dataset_true_idx = indices_[start_sample_idx_in_indices_vec + i];
            try
            {
                torch::data::Example<> example = dataset_.get(dataset_true_idx); // Calls MyTorchDataset::get()
                batch_features_list.push_back(example.data);
                batch_labels_list.push_back(example.target);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error in produce_batch (worker/sync) getting item with dataset_true_idx "
                    << dataset_true_idx << " (original index from shuffled list): " << e.what() << std::endl;
                return std::nullopt; // Fail the entire batch if one item fails
            }
        }

        if (batch_features_list.empty())
        {
            // Should only happen if actual_batch_size was 0 initially
            return std::nullopt;
        }

        // Collate list of tensors into single batch tensors
        torch::Tensor features_batch = torch::stack(batch_features_list, 0);
        torch::Tensor labels_batch = torch::stack(batch_labels_list, 0);
        // .squeeze_() can be useful if labels are [1] shaped from dataset.get but you want flat labels.
        // Be careful if your labels are intentionally multi-dimensional.
        if (labels_batch.dim() > 1 && labels_batch.size(1) == 1)
        {
            labels_batch = labels_batch.squeeze_();
        }


        return {{features_batch, labels_batch}};
    }

public: // Iterator support
    class Iterator
    {
    public:
        Iterator(MyCustomDataLoaderV2* loader, bool end = false)
            : loader_(loader), is_end_(end)
        {
            if (loader_ && !is_end_)
            {
                // Prime the first batch when the iterator is created (for begin())
                current_batch_opt_ = loader_->next_batch();
                if (!current_batch_opt_)
                {
                    // If next_batch() returns nullopt immediately, we are at the end
                    is_end_ = true;
                }
            }
        }

        const BatchData& operator*() const
        {
            if (!current_batch_opt_)
            {
                throw std::runtime_error("Attempting to dereference an end iterator or uninitialized iterator.");
            }
            return *current_batch_opt_;
        }

        BatchData& operator*()
        {
            // Non-const version
            if (!current_batch_opt_)
            {
                throw std::runtime_error("Attempting to dereference an end iterator or uninitialized iterator.");
            }
            return *current_batch_opt_;
        }

        Iterator& operator++()
        {
            if (loader_ && !is_end_)
            {
                // Only advance if not already at the end
                current_batch_opt_ = loader_->next_batch();
                if (!current_batch_opt_)
                {
                    is_end_ = true; // Reached the end
                }
            }
            else
            {
                // If loader is null or already at end, make sure is_end_ is true
                is_end_ = true;
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const
        {
            // Common iterator comparison:
            // 1. If both are "end" iterators, they are equal (so not unequal).
            if (is_end_ && other.is_end_) return false;
            // 2. If one is "end" and the other is not, they are unequal.
            if (is_end_ != other.is_end_) return true;
            // 3. If neither are "end", compare based on loader and potentially current value.
            //    For this simple iterator, if they point to the same loader and both are not end,
            //    they are considered "at the same position" for the purpose of range-for.
            //    A more robust iterator might compare actual batch indices if available.
            return loader_ != other.loader_ || current_batch_opt_.has_value() != other.current_batch_opt_.has_value();
        }

    private:
        MyCustomDataLoaderV2* loader_;
        bool is_end_;
        std::optional<BatchData> current_batch_opt_; // Cache for current batch
    };

    Iterator begin()
    {
        reset_epoch(); // Prepare for a new iteration: (re)shuffle, (re)start workers
        if (current_dataset_size_ == 0 || total_batches_in_epoch_ == 0)
        {
            // Handle empty dataset immediately
            return Iterator(this, true); // Return an end iterator
        }
        return Iterator(this, false); // Creates an iterator that primes the first batch
    }

    Iterator end()
    {
        return Iterator(this, true); // Represents the end
    }

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









int main()
{
    std::cout.precision(10);

    // xt::transforms::Compose compose({
    //     xt::transforms::image::Resize({32, 32}),
    //     xt::transforms::general::Normalize({0.5}, {0.5})
    // });


    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}));
    transform_list.push_back(std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5}));

    auto compose= std::make_unique<xt::transforms::Compose>(transform_list);

    auto dataset = xt::datasets::MNIST(
        "/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false, std::move(compose));

    auto datum = dataset.get(0);
    cout << datum.data.sizes() << endl;
    // xt::DataLoader<decltype(dataset)> loader(
    //     std::move(dataset),
    //     torch::data::DataLoaderOptions().batch_size(64).drop_last(false),
    //     true);

    return 0;
}
