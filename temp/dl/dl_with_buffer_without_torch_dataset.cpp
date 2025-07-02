#include <torch/torch.h>
// No torch/data/datasets.h or torch/data/example.h needed for this version's dataset
#include <vector>
#include <string>
#include <algorithm> // For std::shuffle, std::iota, std::min, std::max
#include <random>    // For std::default_random_engine, std::random_device
#include <iostream>
#include <thread>
// #include <future> // Not directly used in V2 worker loop
#include <mutex>
#include <condition_variable>
#include <deque>      // For the batch queue
#include <atomic>     // For atomic counters and flags
#include <optional>   // For std::optional (used for return types)
#include <stdexcept>  // For std::out_of_range, std::runtime_error, std::invalid_argument
#include <chrono>     // For timing in main

// --- MyCustomDataset (Plain struct, NO inheritance) ---
// Define a type for our data sample from the dataset
using Sample = std::pair<torch::Tensor, torch::Tensor>; // {feature, label}

struct MyCustomDataset {
    std::vector<torch::Tensor> features;
    std::vector<torch::Tensor> labels;
    std::string name_ = "MyCustomDataset";

    MyCustomDataset(std::string name = "MyCustomDataset") : name_(std::move(name)) {}

    void load_data(int num_samples, int feature_dim, int num_classes) {
        features.reserve(num_samples);
        labels.reserve(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            features.push_back(torch::randn({(long)feature_dim}));
            labels.push_back(torch::tensor(i % num_classes, torch::kLong));
        }
        std::cout << name_ << ": Loaded " << num_samples << " samples.\n";
    }

    // Get a single sample by index (returns the Sample pair directly)
    Sample get_item(size_t index) const { // Marked const
        if (index >= features.size()) {
            throw std::out_of_range("Index out of range in MyCustomDataset::get_item for " + name_ +
                                    ". Requested index: " + std::to_string(index) +
                                    ", Dataset size: " + std::to_string(features.size()));
        }
        // .clone() is good practice
        return {features[index].clone(), labels[index].clone()};
    }

    // Get the size of the dataset
    size_t size() const { // Marked const
        return features.size();
    }
};

// Alias for the type of batch data our DataLoader will produce.
// This matches the Sample type but represents a batch of them.
using BatchData = std::pair<torch::Tensor, torch::Tensor>; // {batched_features, batched_labels}


// --- MyCustomDataLoaderV2 (Works with plain MyCustomDataset) ---
class MyCustomDataLoaderV2 {
public:
    MyCustomDataLoaderV2(const MyCustomDataset& dataset, // Takes MyCustomDataset
                         size_t batch_size,
                         bool shuffle,
                         size_t num_workers,
                         size_t prefetch_factor = 2)
        : dataset_(dataset), // Store reference to MyCustomDataset
          batch_size_(batch_size),
          shuffle_(shuffle),
          num_workers_(num_workers),
          prefetch_queue_max_size_(std::max(size_t(1), num_workers * prefetch_factor)),
          shutdown_workers_(false),
          epoch_ended_for_workers_(false),
          current_dataset_size_(0),
          total_batches_in_epoch_(0),
          next_batch_idx_to_produce_(0),
          batches_consumed_in_epoch_(0) {

        if (batch_size_ == 0) {
            throw std::invalid_argument("Batch size cannot be zero.");
        }

        current_dataset_size_ = dataset_.size(); // Use direct .size()

        if (current_dataset_size_ > 0) {
            indices_.resize(current_dataset_size_);
            std::iota(indices_.begin(), indices_.end(), 0);
        } else {
            indices_.clear();
        }
    }

    ~MyCustomDataLoaderV2() {
        shutdown();
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            shutdown_workers_ = true;
            epoch_ended_for_workers_ = true;
        }
        space_available_cv_.notify_all();
        data_available_cv_.notify_all();

        for (auto& worker : worker_threads_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        worker_threads_.clear();
        std::lock_guard<std::mutex> lock(queue_mutex_);
        prefetched_batch_queue_.clear();
    }

    void reset_epoch() {
        shutdown();

        shutdown_workers_ = false;
        epoch_ended_for_workers_ = false;
        batches_consumed_in_epoch_ = 0;
        next_batch_idx_to_produce_ = 0;

        current_dataset_size_ = dataset_.size(); // Use direct .size()

        if (current_dataset_size_ == 0) {
            total_batches_in_epoch_ = 0;
            indices_.clear();
            return;
        }

        total_batches_in_epoch_ = (current_dataset_size_ + batch_size_ - 1) / batch_size_;

        if (indices_.size() != current_dataset_size_) {
            indices_.resize(current_dataset_size_);
            std::iota(indices_.begin(), indices_.end(), 0);
        }

        if (shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            prefetched_batch_queue_.clear();
        }

        if (num_workers_ > 0 && total_batches_in_epoch_ > 0) {
            for (size_t i = 0; i < num_workers_; ++i) {
                worker_threads_.emplace_back(&MyCustomDataLoaderV2::worker_loop, this, i);
            }
        }
    }

    std::optional<BatchData> next_batch() {
        if (current_dataset_size_ == 0) {
            return std::nullopt;
        }

        if (num_workers_ == 0) {
            if (next_batch_idx_to_produce_ >= total_batches_in_epoch_) {
                return std::nullopt;
            }
            size_t current_batch_overall_idx = next_batch_idx_to_produce_++;
            auto batch = produce_batch(current_batch_overall_idx);
            if (batch) batches_consumed_in_epoch_++;
            return batch;
        }

        std::unique_lock<std::mutex> lock(queue_mutex_);
        data_available_cv_.wait(lock, [this] {
            return !prefetched_batch_queue_.empty() ||
                   (epoch_ended_for_workers_ && prefetched_batch_queue_.empty());
        });

        if (prefetched_batch_queue_.empty()) {
            return std::nullopt;
        }

        BatchData batch = prefetched_batch_queue_.front();
        prefetched_batch_queue_.pop_front();
        batches_consumed_in_epoch_++;

        lock.unlock();
        space_available_cv_.notify_one();

        return batch;
    }

private:
    void worker_loop(size_t worker_id) {
        try {
            while (true) {
                if (shutdown_workers_.load(std::memory_order_relaxed)) {
                    break;
                }

                size_t current_batch_overall_idx = next_batch_idx_to_produce_.fetch_add(1, std::memory_order_relaxed);

                if (current_batch_overall_idx >= total_batches_in_epoch_) {
                    next_batch_idx_to_produce_.fetch_sub(1, std::memory_order_relaxed);
                    std::unique_lock<std::mutex> epoch_lock(queue_mutex_);
                    if (next_batch_idx_to_produce_.load(std::memory_order_relaxed) >= total_batches_in_epoch_) {
                         epoch_ended_for_workers_ = true;
                    }
                    if (epoch_ended_for_workers_) {
                        data_available_cv_.notify_all();
                    }
                    space_available_cv_.wait(epoch_lock, [this] {
                        return shutdown_workers_.load() || !epoch_ended_for_workers_.load();
                    });
                    if (shutdown_workers_.load()) break;
                    if (!epoch_ended_for_workers_.load()) continue;
                    continue;
                }

                auto batch_opt = produce_batch(current_batch_overall_idx);

                if (batch_opt) {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    space_available_cv_.wait(lock, [this] {
                        return prefetched_batch_queue_.size() < prefetch_queue_max_size_ || shutdown_workers_.load();
                    });
                    if (shutdown_workers_.load()) break;
                    prefetched_batch_queue_.push_back(*batch_opt);
                    lock.unlock();
                    data_available_cv_.notify_one();
                } else {
                     std::cerr << "Worker " << worker_id << " failed to produce batch " << current_batch_overall_idx << " or it was empty." << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "!!! Exception in worker " << worker_id << ": " << e.what() << std::endl;
            std::unique_lock<std::mutex> lock(queue_mutex_);
            shutdown_workers_ = true;
            epoch_ended_for_workers_ = true;
            data_available_cv_.notify_all();
            space_available_cv_.notify_all();
        }
    }

    std::optional<BatchData> produce_batch(size_t batch_overall_idx) {
        size_t start_sample_idx_in_indices_vec = batch_overall_idx * batch_size_;

        if (start_sample_idx_in_indices_vec >= current_dataset_size_ && current_dataset_size_ > 0) {
             std::cerr << "Warning: produce_batch called for batch_overall_idx " << batch_overall_idx
                      << " which is out of bounds for current_dataset_size_ " << current_dataset_size_ << std::endl;
            return std::nullopt;
        }
        if (current_dataset_size_ == 0) return std::nullopt;

        size_t actual_batch_size = std::min(batch_size_, current_dataset_size_ - start_sample_idx_in_indices_vec);
        if (actual_batch_size == 0) {
            return std::nullopt;
        }

        std::vector<torch::Tensor> batch_features_list;
        std::vector<torch::Tensor> batch_labels_list;
        batch_features_list.reserve(actual_batch_size);
        batch_labels_list.reserve(actual_batch_size);

        for (size_t i = 0; i < actual_batch_size; ++i) {
            size_t dataset_true_idx = indices_[start_sample_idx_in_indices_vec + i];
            try {
                // Use dataset_.get_item() which returns Sample (std::pair<Tensor, Tensor>)
                Sample sample = dataset_.get_item(dataset_true_idx);
                batch_features_list.push_back(sample.first);  // The feature tensor
                batch_labels_list.push_back(sample.second); // The label tensor
            } catch (const std::exception& e) {
                 std::cerr << "Error in produce_batch (worker/sync) getting item with dataset_true_idx "
                           << dataset_true_idx << ": " << e.what() << std::endl;
                 return std::nullopt;
            }
        }

        if (batch_features_list.empty()) {
            return std::nullopt;
        }

        torch::Tensor features_batch = torch::stack(batch_features_list, 0);
        torch::Tensor labels_batch = torch::stack(batch_labels_list, 0);
        if (labels_batch.dim() > 1 && labels_batch.size(1) == 1) {
             labels_batch = labels_batch.squeeze_();
        }

        return {{features_batch, labels_batch}};
    }

public: // Iterator support
    class Iterator {
    public:
        Iterator(MyCustomDataLoaderV2* loader, bool end = false)
            : loader_(loader), is_end_(end) {
            if (loader_ && !is_end_) {
                current_batch_opt_ = loader_->next_batch();
                if (!current_batch_opt_) {
                    is_end_ = true;
                }
            }
        }

        const BatchData& operator*() const {
            if (!current_batch_opt_) throw std::runtime_error("Dereferencing end/uninitialized iterator");
            return *current_batch_opt_;
        }
        BatchData& operator*() {
            if (!current_batch_opt_) throw std::runtime_error("Dereferencing end/uninitialized iterator");
            return *current_batch_opt_;
        }

        Iterator& operator++() {
            if (loader_ && !is_end_) {
                current_batch_opt_ = loader_->next_batch();
                if (!current_batch_opt_) is_end_ = true;
            } else {
                is_end_ = true;
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            if (is_end_ && other.is_end_) return false;
            if (is_end_ != other.is_end_) return true;
            return loader_ != other.loader_ || current_batch_opt_.has_value() != other.current_batch_opt_.has_value();
        }

    private:
        MyCustomDataLoaderV2* loader_;
        bool is_end_;
        std::optional<BatchData> current_batch_opt_;
    };

    Iterator begin() {
        reset_epoch();
        if (current_dataset_size_ == 0 || total_batches_in_epoch_ == 0) {
            return Iterator(this, true);
        }
        return Iterator(this, false);
    }

    Iterator end() {
        return Iterator(this, true);
    }

private:
    const MyCustomDataset& dataset_; // Using MyCustomDataset
    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;
    size_t prefetch_queue_max_size_;
    size_t current_dataset_size_;

    std::vector<size_t> indices_;

    std::deque<BatchData> prefetched_batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable data_available_cv_;
    std::condition_variable space_available_cv_;

    std::vector<std::thread> worker_threads_;
    std::atomic<bool> shutdown_workers_;
    std::atomic<bool> epoch_ended_for_workers_;

    std::atomic<size_t> next_batch_idx_to_produce_;
    size_t total_batches_in_epoch_;
    size_t batches_consumed_in_epoch_;
};


// --- Main Function ---
int main() {
    torch::manual_seed(0);

    // --- Scenario 1: Standard usage with MyCustomDataset ---
    MyCustomDataset dataset("MainCustomDataset"); // Use the plain MyCustomDataset
    dataset.load_data(/*num_samples=*/107, /*feature_dim=*/5, /*num_classes=*/3);

    int num_epochs = 2;
    size_t batch_size = 16;
    size_t num_workers = 2;
    bool shuffle = true;

    std::cout << "\n--- MyCustomDataLoaderV2 with MyCustomDataset (num_workers=" << num_workers
              << ", shuffle=" << shuffle << ", batch_size=" << batch_size << ") ---" << std::endl;

    MyCustomDataLoaderV2 data_loader(dataset, batch_size, shuffle, num_workers);

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        std::cout << "\nEpoch: " << epoch << std::endl;
        int batch_count = 0;
        auto epoch_start_time = std::chrono::high_resolution_clock::now();

        for (const auto& batch : data_loader) {
            torch::Tensor features = batch.first;
            torch::Tensor labels = batch.second;
            // std::this_thread::sleep_for(std::chrono::milliseconds(20));

            std::cout << "  Batch " << ++batch_count << " received. Features: " << features.sizes()
                      << ", Labels: " << labels.sizes();
            if (labels.numel() > 0) {
                 std::cout << " First label: " << labels[0].item<long>();
            }
            std::cout << std::endl;
        }
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time);
        std::cout << "Epoch " << epoch << " completed in " << duration.count() << "ms. Total batches: " << batch_count << std::endl;
        if (batch_count == 0 && dataset.size() > 0) { // Use dataset.size() directly
            std::cerr << "Error: No batches processed for a non-empty dataset in epoch " << epoch << std::endl;
        }
    }

    // --- Scenario 2: Testing with num_workers = 0 ---
    std::cout << "\n--- Testing with num_workers = 0 (MyCustomDataset) ---\n";
    MyCustomDataset dataset_nw0("CustomDatasetNW0");
    dataset_nw0.load_data(20, 5, 2);
    MyCustomDataLoaderV2 loader_nw0(dataset_nw0, 5, false, 0);
    int batch_count_nw0 = 0;
    std::cout << "Epoch for NW0:" << std::endl;
    for (const auto& batch : loader_nw0) {
         std::cout << "  NW0 Batch " << ++batch_count_nw0 << ": Features:" << batch.first.sizes() << " Labels:" << batch.second.sizes() << std::endl;
    }
    std::cout << "NW0 Total batches: " << batch_count_nw0 << std::endl;


    // --- Scenario 3: Testing with an empty MyCustomDataset ---
    std::cout << "\n--- Testing with empty MyCustomDataset ---\n";
    MyCustomDataset empty_ds("EmptyCustomDataset");
    MyCustomDataLoaderV2 empty_loader(empty_ds, 5, false, 2);
    int empty_batch_count = 0;
    std::cout << "Epoch for Empty DS:" << std::endl;
    for (const auto& batch : empty_loader) {
        empty_batch_count++;
        std::cout << "  Empty DS Batch " << empty_batch_count << " (THIS SHOULD NOT PRINT)" << std::endl;
    }
    std::cout << "Empty dataset total batches: " << empty_batch_count << std::endl;

    // --- Scenario 4: Small MyCustomDataset, batch_size larger than dataset ---
    std::cout << "\n--- Testing with small MyCustomDataset, batch_size > dataset_size ---\n";
    MyCustomDataset small_ds("SmallCustomDS");
    small_ds.load_data(3, 5, 2);
    MyCustomDataLoaderV2 small_loader(small_ds, 10, false, 1);
    int small_batch_count = 0;
    std::cout << "Epoch for Small DS:" << std::endl;
    for (const auto& batch : small_loader) {
         std::cout << "  Small DS Batch " << ++small_batch_count << ": Features:" << batch.first.sizes() << " Labels:" << batch.second.sizes() << std::endl;
    }
    std::cout << "Small DS Total batches: " << small_batch_count << std::endl;


    std::cout << "\nMain function finished." << std::endl;
    return 0;
}