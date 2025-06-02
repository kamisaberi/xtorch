#include <torch/torch.h>
#include <torch/data/datasets.h> // For torch::data::Dataset
#include <torch/data/example.h>  // For torch::data::Example
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <optional>

// --- MyTorchDataset Definition ---
// Using CRTP (Curiously Recurring Template Pattern) with explicit ExampleType
struct MyTorchDataset : public torch::data::Dataset<MyTorchDataset, torch::data::Example<>> {
public:
    MyTorchDataset(std::string name = "MyTorchDataset") : name_(std::move(name)) {}

    void load_data(int num_samples, int feature_dim, int num_classes) {
        internal_features_.reserve(num_samples);
        internal_labels_.reserve(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            internal_features_.push_back(torch::randn({(long)feature_dim}));
            internal_labels_.push_back(torch::tensor(i % num_classes, torch::kLong));
        }
        std::cout << name_ << ": Loaded " << num_samples << " samples.\n";
    }

    // Implement the get method required by torch::data::Dataset
    torch::data::Example<> get(size_t index) override {
        if (index >= internal_features_.size()) {
            throw std::out_of_range("Index out of range in MyTorchDataset::get for " + name_ +
                                    ". Index: " + std::to_string(index) + ", Size: " + std::to_string(internal_features_.size()));
        }
        // Simulate some work if data loading was I/O bound
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return {internal_features_[index].clone(), internal_labels_[index].clone()}; // {data, target}
    }

    // Implement the size method required by torch::data::Dataset
    torch::optional<size_t> size() const override {
        return internal_features_.size();
    }

private:
    std::string name_;
    std::vector<torch::Tensor> internal_features_;
    std::vector<torch::Tensor> internal_labels_;
};

// For clarity, define the Sample type that our DataLoader will produce.
// It's convenient to keep it as a pair for the existing collate logic.
using BatchData = std::pair<torch::Tensor, torch::Tensor>; // {batched_features, batched_labels}


// --- MyCustomDataLoaderV2 (Modified to use MyTorchDataset) ---
class MyCustomDataLoaderV2 {
public:
    // Constructor now takes MyTorchDataset
    MyCustomDataLoaderV2(const MyTorchDataset& dataset, // Changed type here
                         size_t batch_size,
                         bool shuffle,
                         size_t num_workers,
                         size_t prefetch_factor = 2)
        : dataset_(dataset), // Store reference to MyTorchDataset
          batch_size_(batch_size),
          shuffle_(shuffle),
          num_workers_(num_workers),
          prefetch_queue_max_size_(std::max(size_t(1), num_workers * prefetch_factor)),
          shutdown_workers_(false),
          epoch_ended_for_workers_(false),
          total_batches_in_epoch_(0),
          next_batch_idx_to_produce_(0),
          batches_consumed_in_epoch_(0) {

        if (batch_size_ == 0) {
            throw std::invalid_argument("Batch size cannot be zero.");
        }

        auto dataset_size_opt = dataset_.size();
        if (!dataset_size_opt.has_value()) {
            throw std::runtime_error("Dataset size is unknown (optional is nullopt). This DataLoader requires a known size.");
        }
        current_dataset_size_ = dataset_size_opt.value();

        if (current_dataset_size_ > 0) {
            indices_.resize(current_dataset_size_);
            std::iota(indices_.begin(), indices_.end(), 0);
        } else {
            indices_.clear();
        }
        // reset_epoch() will be called by begin()
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
        shutdown(); // Stop existing workers and clear state

        shutdown_workers_ = false;
        epoch_ended_for_workers_ = false;
        batches_consumed_in_epoch_ = 0;
        next_batch_idx_to_produce_ = 0;

        auto dataset_size_opt = dataset_.size();
        // Re-fetch size in case dataset could change (though not in this example)
        if (!dataset_size_opt.has_value()) {
             throw std::runtime_error("Dataset size is unknown during reset_epoch.");
        }
        current_dataset_size_ = dataset_size_opt.value();


        if (current_dataset_size_ == 0) {
            total_batches_in_epoch_ = 0;
            indices_.clear();
            return;
        }

        total_batches_in_epoch_ = (current_dataset_size_ + batch_size_ - 1) / batch_size_;

        if (indices_.size() != current_dataset_size_) { // Resize if dataset changed (not typical mid-training)
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

    // Return type is now BatchData
    std::optional<BatchData> next_batch() {
        if (current_dataset_size_ == 0) return std::nullopt;

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
                    epoch_ended_for_workers_ = (next_batch_idx_to_produce_.load() >= total_batches_in_epoch_);
                    if (epoch_ended_for_workers_) data_available_cv_.notify_all();
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
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception in worker " << worker_id << ": " << e.what() << std::endl;
            std::unique_lock<std::mutex> lock(queue_mutex_);
            shutdown_workers_ = true;
            epoch_ended_for_workers_ = true;
            data_available_cv_.notify_all();
            space_available_cv_.notify_all();
        }
    }

    // produce_batch now uses dataset_.get() which returns torch::data::Example<>
    std::optional<BatchData> produce_batch(size_t batch_overall_idx) {
        size_t start_sample_idx_in_indices = batch_overall_idx * batch_size_;
        // Check against current_dataset_size_ which is known good
        if (start_sample_idx_in_indices >= current_dataset_size_) {
            return std::nullopt;
        }

        size_t actual_batch_size = std::min(batch_size_, current_dataset_size_ - start_sample_idx_in_indices);
        if (actual_batch_size == 0) return std::nullopt;

        std::vector<torch::Tensor> batch_features_list;
        std::vector<torch::Tensor> batch_labels_list;
        batch_features_list.reserve(actual_batch_size);
        batch_labels_list.reserve(actual_batch_size);

        for (size_t i = 0; i < actual_batch_size; ++i) {
            size_t dataset_true_idx = indices_[start_sample_idx_in_indices + i];
            try {
                // Use dataset_.get() which returns torch::data::Example<>
                torch::data::Example<> example = dataset_.get(dataset_true_idx);
                batch_features_list.push_back(example.data);
                batch_labels_list.push_back(example.target);
            } catch (const std::exception& e) {
                 std::cerr << "Error in produce_batch (worker) getting item with dataset_true_idx "
                           << dataset_true_idx << ": " << e.what() << std::endl;
                 return std::nullopt; // Fail the batch
            }
        }

        if (batch_features_list.empty()) {
            return std::nullopt;
        }

        // Collate into BatchData
        return {{torch::stack(batch_features_list, 0), torch::stack(batch_labels_list, 0).squeeze_()}};
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

        // Return type matches BatchData
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
        std::optional<BatchData> current_batch_opt_; // Stores BatchData
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
    const MyTorchDataset& dataset_; // Changed type
    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;
    size_t prefetch_queue_max_size_;
    size_t current_dataset_size_; // Store the known dataset size for the epoch

    std::vector<size_t> indices_;

    std::deque<BatchData> prefetched_batch_queue_; // Stores BatchData
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


// --- Main Function (Updated to use MyTorchDataset) ---
int main() {
    torch::manual_seed(0);

    // Use MyTorchDataset
    MyTorchDataset dataset("MainTorchDataset");
    dataset.load_data(/*num_samples=*/107, /*feature_dim=*/5, /*num_classes=*/3);

    int num_epochs = 2;
    size_t batch_size = 16;
    size_t num_workers = 2;
    bool shuffle = true;

    std::cout << "\n--- MyCustomDataLoaderV2 with MyTorchDataset (num_workers=" << num_workers
              << ", shuffle=" << shuffle << ", batch_size=" << batch_size << ") ---" << std::endl;

    // Pass MyTorchDataset to the DataLoader
    MyCustomDataLoaderV2 data_loader(dataset, batch_size, shuffle, num_workers);

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        std::cout << "\nEpoch: " << epoch << std::endl;
        int batch_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (const auto& batch : data_loader) { // batch is now BatchData (std::pair)
            torch::Tensor features = batch.first;
            torch::Tensor labels = batch.second;
            // std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Simulate work

            std::cout << "Batch " << ++batch_count << " received. Features: " << features.sizes()
                      << ", Labels: " << labels.sizes() << " First label: " << (labels.numel() > 0 ? labels[0].item<long>() : -1L) << std::endl;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Epoch " << epoch << " completed in " << duration.count() << "ms. Total batches: " << batch_count << std::endl;
         if (batch_count == 0 && dataset.size().value_or(0) > 0) {
            std::cerr << "Error: No batches processed for a non-empty dataset in epoch " << epoch << std::endl;
        }
    }

    std::cout << "\n--- Testing with num_workers = 0 ---\n";
    MyTorchDataset dataset_nw0("DatasetNW0");
    dataset_nw0.load_data(20, 5, 2);
    MyCustomDataLoaderV2 loader_nw0(dataset_nw0, 5, false, 0);
    int batch_count_nw0 = 0;
    for (const auto& batch : loader_nw0) {
         std::cout << "NW0 Batch " << ++batch_count_nw0 << ": F:" << batch.first.sizes() << " L:" << batch.second.sizes() << std::endl;
    }
    std::cout << "NW0 Total batches: " << batch_count_nw0 << std::endl;


    std::cout << "\n--- Testing with empty dataset ---\n";
    MyTorchDataset empty_ds("EmptyTorchDataset");
    // empty_ds.load_data(0,5,2); // already empty
    MyCustomDataLoaderV2 empty_loader(empty_ds, 5, false, 2);
    int empty_batch_count = 0;
    for (const auto& batch : empty_loader) {
        empty_batch_count++;
    }
    std::cout << "Empty dataset total batches: " << empty_batch_count << std::endl;


    std::cout << "\nDataLoader shutting down explicitly (though destructor would do it too)." << std::endl;
    // data_loader.shutdown(); // Destructor handles this
    std::cout << "Main finished." << std::endl;
    return 0;
}