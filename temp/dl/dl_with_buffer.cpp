#include <torch/torch.h>
#include <vector>
#include <algorithm> // For std::shuffle, std::iota
#include <random>    // For std::default_random_engine
#include <iostream>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>      // For the batch queue
#include <atomic>     // For atomic counters and flags
#include <optional>

// (MyCustomDataset and Sample definition remain the same as the previous example)
using Sample = std::pair<torch::Tensor, torch::Tensor>; // {feature, label}

struct MyCustomDataset
{
    std::vector<torch::Tensor> features;
    std::vector<torch::Tensor> labels;
    std::string name_ = "Dataset"; // Just for logging

    MyCustomDataset(std::string name = "Dataset") : name_(name)
    {
    }

    void load_data(int num_samples, int feature_dim, int num_classes)
    {
        features.reserve(num_samples);
        labels.reserve(num_samples);
        for (int i = 0; i < num_samples; ++i)
        {
            features.push_back(torch::randn({(long)feature_dim}));
            // Simulate some work in get_item
            // labels.push_back(torch::randint(0, num_classes, {1}, torch::kLong));
            labels.push_back(torch::tensor(i % num_classes, torch::kLong)); // More predictable for debugging
        }
        std::cout << name_ << ": Loaded " << num_samples << " samples.\n";
    }

    Sample get_item(size_t index) const
    {
        if (index >= features.size())
        {
            // std::cerr << name_ << " Error: Index " << index << " out of range. Size: " << features.size() << std::endl;
            throw std::out_of_range("Index out of range in MyCustomDataset::get_item for " + name_);
        }
        // Simulate some work if data loading was I/O bound
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return {features[index].clone(), labels[index].clone()};
        // Clone to ensure no accidental sharing if tensors were modified
    }

    size_t size() const
    {
        return features.size();
    }
};


class MyCustomDataLoaderV2
{
public:
    MyCustomDataLoaderV2(const MyCustomDataset& dataset,
                         size_t batch_size,
                         bool shuffle,
                         size_t num_workers,
                         size_t prefetch_factor = 2) // Number of batches to prefetch (total queue size)
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          num_workers_(num_workers),
          prefetch_queue_max_size_(std::max(size_t(1), num_workers * prefetch_factor)),
          // Ensure queue size is at least 1
          shutdown_workers_(false),
          epoch_ended_for_workers_(false),
          total_batches_in_epoch_(0),
          next_batch_idx_to_produce_(0),
          batches_consumed_in_epoch_(0)
    {
        if (batch_size_ == 0)
        {
            throw std::invalid_argument("Batch size cannot be zero.");
        }
        if (num_workers_ == 0)
        {
            // For num_workers == 0, this advanced prefetching logic isn't strictly necessary,
            // a simpler sequential loader would suffice. However, we can make it work
            // by having the main thread act as the 'worker' when next_batch is called.
            // Or, for this example, let's mandate num_workers >= 1 for the prefetching part.
            // If you want a num_workers = 0 path, it would bypass the worker threads.
            // For now, let's allow it and handle it mostly sequentially in next_batch()
            // if the queue is empty and workers aren't running (or aren't supposed to).
            std::cout << "Warning: num_workers=0. Prefetching will be minimal or done by main thread." << std::endl;
        }

        indices_.resize(dataset_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        // reset_epoch() will be called by begin()
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
            epoch_ended_for_workers_ = true; // Ensure they don't get stuck waiting for new epoch
        }
        space_available_cv_.notify_all(); // Wake up any workers waiting for space
        data_available_cv_.notify_all(); // Wake up main thread if it's waiting

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
        while (!prefetched_batch_queue_.empty())
        {
            prefetched_batch_queue_.pop_front();
        }
    }


    void reset_epoch()
    {
        // 1. Stop and join existing workers if any
        shutdown(); // This sets shutdown_workers_ and joins

        // 2. Reset internal state for the new epoch
        shutdown_workers_ = false; // Allow new workers to run
        epoch_ended_for_workers_ = false;
        batches_consumed_in_epoch_ = 0;
        next_batch_idx_to_produce_ = 0; // Reset for workers

        if (dataset_.size() == 0)
        {
            total_batches_in_epoch_ = 0;
            return; // No data, nothing to do
        }

        total_batches_in_epoch_ = (dataset_.size() + batch_size_ - 1) / batch_size_;

        if (shuffle_)
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }

        // Clear any stale batches from the queue (should be empty after shutdown)
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            while (!prefetched_batch_queue_.empty())
            {
                prefetched_batch_queue_.pop_front();
            }
        }


        // 3. Start new worker threads if num_workers > 0
        if (num_workers_ > 0)
        {
            for (size_t i = 0; i < num_workers_; ++i)
            {
                worker_threads_.emplace_back(&MyCustomDataLoaderV2::worker_loop, this, i);
            }
        }
    }


    std::optional<Sample> next_batch()
    {
        if (dataset_.size() == 0) return std::nullopt;

        if (num_workers_ == 0)
        {
            // Handle num_workers = 0 synchronously
            if (next_batch_idx_to_produce_ >= total_batches_in_epoch_)
            {
                return std::nullopt; // End of epoch
            }
            size_t current_batch_overall_idx = next_batch_idx_to_produce_++;
            auto batch = produce_batch(current_batch_overall_idx);
            if (batch) batches_consumed_in_epoch_++;
            return batch;
        }

        std::unique_lock<std::mutex> lock(queue_mutex_);
        // Wait until data is available OR all batches for the epoch are produced AND consumed
        data_available_cv_.wait(lock, [this]
        {
            return !prefetched_batch_queue_.empty() ||
                (epoch_ended_for_workers_ && prefetched_batch_queue_.empty());
        });

        if (prefetched_batch_queue_.empty())
        {
            // This means epoch_ended_for_workers_ is true and queue is empty.
            // All batches have been produced and consumed.
            return std::nullopt;
        }

        Sample batch = prefetched_batch_queue_.front();
        prefetched_batch_queue_.pop_front();
        batches_consumed_in_epoch_++;

        lock.unlock();
        space_available_cv_.notify_one(); // Notify one worker that space is available

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
                    // std::cout << "Worker " << worker_id << " shutting down." << std::endl;
                    break;
                }

                size_t current_batch_overall_idx = next_batch_idx_to_produce_.fetch_add(1, std::memory_order_relaxed);

                if (current_batch_overall_idx >= total_batches_in_epoch_)
                {
                    // All batches for this epoch have been claimed or are being processed
                    next_batch_idx_to_produce_.fetch_sub(1, std::memory_order_relaxed); // Correct over-increment
                    // Signal that this worker believes the epoch is done from production side
                    // std::cout << "Worker " << worker_id << ": No more batches to produce for this epoch (idx " << current_batch_overall_idx << " >= " << total_batches_in_epoch_ << ")." << std::endl;

                    // Wait for epoch reset or shutdown
                    std::unique_lock<std::mutex> epoch_lock(queue_mutex_); // Use same mutex for consistency
                    epoch_ended_for_workers_ = (next_batch_idx_to_produce_.load() >= total_batches_in_epoch_);
                    if (epoch_ended_for_workers_) data_available_cv_.notify_all();
                    // Wake main thread if it's waiting for last batches

                    space_available_cv_.wait(epoch_lock, [this, worker_id]
                    {
                        // Wait for space or shutdown or new epoch
                        // std::cout << "Worker " << worker_id << " waiting for new epoch or shutdown. Shutdown: " << shutdown_workers_.load() << " Epoch Ended: " << epoch_ended_for_workers_.load() << std::endl;
                        return shutdown_workers_.load() || !epoch_ended_for_workers_.load();
                    });
                    // std::cout << "Worker " << worker_id << " woke up. Shutdown: " << shutdown_workers_.load() << " Epoch Ended: " << epoch_ended_for_workers_.load() << std::endl;

                    if (shutdown_workers_.load()) break;
                    if (!epoch_ended_for_workers_.load())
                    {
                        // New epoch started
                        // std::cout << "Worker " << worker_id << " detected new epoch, continuing." << std::endl;
                        continue; // Re-evaluate next_batch_idx_to_produce_
                    }
                    // If still epoch_ended and not shutdown, something is wrong or it's a spurious wakeup
                    // std::cout << "Worker " << worker_id << " spurious wakeup? Continuing to check shutdown." << std::endl;
                    continue;
                }

                // Produce the batch
                auto batch_opt = produce_batch(current_batch_overall_idx);

                if (batch_opt)
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    space_available_cv_.wait(lock, [this, worker_id]
                    {
                        // if (prefetched_batch_queue_.size() >= prefetch_queue_max_size_ && !shutdown_workers_.load()) {
                        //     std::cout << "Worker " << worker_id << " waiting for space (queue size " << prefetched_batch_queue_.size() << ")" << std::endl;
                        // }
                        return prefetched_batch_queue_.size() < prefetch_queue_max_size_ || shutdown_workers_.load();
                    });

                    if (shutdown_workers_.load())
                    {
                        // std::cout << "Worker " << worker_id << " shutting down while trying to push to queue." << std::endl;
                        break;
                    }

                    prefetched_batch_queue_.push_back(*batch_opt);
                    // std::cout << "Worker " << worker_id << " pushed batch " << current_batch_overall_idx << ". Queue size: " << prefetched_batch_queue_.size() << std::endl;
                    lock.unlock();
                    data_available_cv_.notify_one(); // Notify main thread that data is available
                }
                else
                {
                    // Should not happen if current_batch_overall_idx was valid,
                    // but good to handle defensively.
                    // std::cerr << "Worker " << worker_id << " failed to produce batch " << current_batch_overall_idx << std::endl;
                    // This could happen if dataset size changed or logic error.
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Exception in worker " << worker_id << ": " << e.what() << std::endl;
            // Optionally, set a global error flag that the main thread can check.
            // For now, just log and terminate the worker. The DataLoader might hang if other workers also fail.
            std::unique_lock<std::mutex> lock(queue_mutex_);
            shutdown_workers_ = true; // Signal others to stop too on critical error
            epoch_ended_for_workers_ = true;
            data_available_cv_.notify_all();
            space_available_cv_.notify_all();
        }
        // std::cout << "Worker " << worker_id << " finished." << std::endl;
        // Ensure main thread is not stuck if this was the last active worker
        std::unique_lock<std::mutex> lock(queue_mutex_);
        epoch_ended_for_workers_ = (next_batch_idx_to_produce_.load() >= total_batches_in_epoch_); // Re-check
        if (epoch_ended_for_workers_ && prefetched_batch_queue_.empty())
        {
            data_available_cv_.notify_all();
        }
    }

    // Helper to create a batch given its overall index in the epoch
    std::optional<Sample> produce_batch(size_t batch_overall_idx)
    {
        size_t start_sample_idx_in_indices = batch_overall_idx * batch_size_;
        if (start_sample_idx_in_indices >= dataset_.size())
        {
            return std::nullopt; // Should be caught by total_batches_in_epoch check by caller
        }

        size_t actual_batch_size = std::min(batch_size_, dataset_.size() - start_sample_idx_in_indices);
        if (actual_batch_size == 0) return std::nullopt;

        std::vector<torch::Tensor> batch_features;
        std::vector<torch::Tensor> batch_labels;
        batch_features.reserve(actual_batch_size);
        batch_labels.reserve(actual_batch_size);

        for (size_t i = 0; i < actual_batch_size; ++i)
        {
            size_t dataset_idx = indices_[start_sample_idx_in_indices + i];
            try
            {
                Sample sample = dataset_.get_item(dataset_idx);
                batch_features.push_back(sample.first);
                batch_labels.push_back(sample.second);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error in produce_batch getting item " << dataset_idx << " (original index): " << e.what()
                    << std::endl;
                // Decide how to handle: skip item, return partial batch, or rethrow/return nullopt
                return std::nullopt; // Fail the batch
            }
        }

        if (batch_features.empty())
        {
            return std::nullopt;
        }

        return {{torch::stack(batch_features, 0), torch::stack(batch_labels, 0).squeeze_()}};
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
                // Prime the first batch
                current_batch_opt_ = loader_->next_batch();
                if (!current_batch_opt_)
                {
                    is_end_ = true;
                }
            }
        }

        const Sample& operator*() const
        {
            if (!current_batch_opt_)
            {
                throw std::runtime_error("Attempting to dereference an end iterator or uninitialized iterator.");
            }
            return *current_batch_opt_;
        }

        Sample& operator*()
        {
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
                current_batch_opt_ = loader_->next_batch();
                if (!current_batch_opt_)
                {
                    is_end_ = true;
                }
            }
            else
            {
                is_end_ = true; // Cannot increment past end
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const
        {
            if (is_end_ && other.is_end_) return false;
            if (is_end_ != other.is_end_) return true;
            // If both are not end, they are different if they point to different loaders
            // or if one is primed and the other isn't (though this setup avoids that)
            return loader_ != other.loader_ || current_batch_opt_.has_value() != other.current_batch_opt_.has_value();
        }

    private:
        MyCustomDataLoaderV2* loader_;
        bool is_end_;
        std::optional<Sample> current_batch_opt_;
    };

    Iterator begin()
    {
        reset_epoch(); // Prepare for a new iteration, (re)start workers
        if (dataset_.size() == 0 || total_batches_in_epoch_ == 0)
        {
            // Handle empty dataset immediately
            return Iterator(this, true); // Return an end iterator
        }
        return Iterator(this, false);
    }

    Iterator end()
    {
        return Iterator(this, true);
    }

private:
    const MyCustomDataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;
    size_t prefetch_queue_max_size_;

    std::vector<size_t> indices_; // Shuffled indices for the current epoch

    std::deque<Sample> prefetched_batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable data_available_cv_;
    std::condition_variable space_available_cv_;

    std::vector<std::thread> worker_threads_;
    std::atomic<bool> shutdown_workers_; // Global shutdown signal for all workers
    std::atomic<bool> epoch_ended_for_workers_; // True when workers have processed all batches for current epoch

    std::atomic<size_t> next_batch_idx_to_produce_; // Index of the next batch to be produced by any worker
    size_t total_batches_in_epoch_;
    size_t batches_consumed_in_epoch_; // For debugging/tracking
};


int main()
{
    torch::manual_seed(0);

    MyCustomDataset dataset("MainDataset");
    dataset.load_data(/*num_samples=*/107, /*feature_dim=*/5, /*num_classes=*/3);
    // MyCustomDataset dataset("MainDataset"); dataset.load_data(10, 2, 2); // Small dataset for testing

    int num_epochs = 2;
    size_t batch_size = 16; // 32;
    size_t num_workers = 2; // 4;
    bool shuffle = true;

    std::cout << "\n--- MyCustomDataLoaderV2 (num_workers=" << num_workers
        << ", shuffle=" << shuffle << ", batch_size=" << batch_size << ") ---" << std::endl;

    MyCustomDataLoaderV2 data_loader(dataset, batch_size, shuffle, num_workers, /*prefetch_factor=*/2);

    for (int epoch = 1; epoch <= num_epochs; ++epoch)
    {
        std::cout << "\nEpoch: " << epoch << std::endl;
        int batch_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (const auto& batch : data_loader)
        {
            // data_loader.begin() will call reset_epoch()
            torch::Tensor features = batch.first;
            torch::Tensor labels = batch.second;
            // Simulate some training work
            // std::this_thread::sleep_for(std::chrono::milliseconds(5));

            std::cout << "Batch " << ++batch_count << " received. Features: " << features.sizes()
                << ", Labels: " << labels.sizes() << " First label: " << labels[0].item<long>() << std::endl;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Epoch " << epoch << " completed in " << duration.count() << "ms. Total batches: " << batch_count
            << std::endl;
        if (batch_count == 0 && dataset.size() > 0)
        {
            std::cerr << "Error: No batches processed for a non-empty dataset in epoch " << epoch << std::endl;
        }
    }

    std::cout << "\n--- Testing with num_workers = 0 ---\n";
    MyCustomDataset dataset_nw0("DatasetNW0");
    dataset_nw0.load_data(20, 5, 2);
    MyCustomDataLoaderV2 loader_nw0(dataset_nw0, 5, false, 0);
    int batch_count_nw0 = 0;
    for (const auto& batch : loader_nw0)
    {
        std::cout << "NW0 Batch " << ++batch_count_nw0 << ": F:" << batch.first.sizes() << " L:" << batch.second.sizes()
            << std::endl;
    }
    std::cout << "NW0 Total batches: " << batch_count_nw0 << std::endl;


    std::cout << "\n--- Testing with empty dataset ---\n";
    MyCustomDataset empty_ds("EmptyDataset");
    // empty_ds.load_data(0,5,2); // already empty
    MyCustomDataLoaderV2 empty_loader(empty_ds, 5, false, 2);
    int empty_batch_count = 0;
    for (const auto& batch : empty_loader)
    {
        empty_batch_count++;
    }
    std::cout << "Empty dataset total batches: " << empty_batch_count << std::endl;


    std::cout << "\nDataLoader shutting down explicitly (though destructor would do it too)." << std::endl;
    // data_loader.shutdown(); // Already done by destructor when data_loader goes out of scope.
    // If data_loader was dynamically allocated, you'd call shutdown() before delete.

    std::cout << "Main finished." << std::endl;
    return 0;
}
