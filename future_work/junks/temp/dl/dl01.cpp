#include <torch/torch.h>
#include <vector>
#include <algorithm> // For std::shuffle, std::iota
#include <random>    // For std::default_random_engine
#include <iostream>
#include <thread>     // For std::thread
#include <future>     // For std::async, std::future
#include <mutex>      // For std::mutex (if needed for more complex scenarios)

// Define a type for our data sample
using Sample = std::pair<torch::Tensor, torch::Tensor>; // {feature, label}

// 1. MyCustomDataset: Holds the actual data
struct MyCustomDataset {
    std::vector<torch::Tensor> features;
    std::vector<torch::Tensor> labels;

    MyCustomDataset() = default;

    // Example: Load data (in a real scenario, this would load from disk)
    void load_data(int num_samples, int feature_dim, int num_classes) {
        features.reserve(num_samples);
        labels.reserve(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            features.push_back(torch::randn({(long)feature_dim}));
            labels.push_back(torch::randint(0, num_classes, {1}, torch::kLong));
        }
        std::cout << "Dataset: Loaded " << num_samples << " samples.\n";
    }

    // Get a single sample by index
    Sample get_item(size_t index) const {
        if (index >= features.size()) {
            throw std::out_of_range("Index out of range in MyCustomDataset");
        }
        return {features[index], labels[index]};
    }

    // Get the size of the dataset
    size_t size() const {
        return features.size();
    }
};


// 2. MyCustomDataLoader: Manages batching, shuffling, and iteration
class MyCustomDataLoader {
public:
    MyCustomDataLoader(const MyCustomDataset& dataset,
                       size_t batch_size,
                       bool shuffle,
                       size_t num_workers = 0) // 0 means single-threaded for fetching this batch
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          num_workers_(num_workers),
          current_sample_idx_(0) {
        if (batch_size == 0) {
            throw std::invalid_argument("Batch size cannot be zero.");
        }
        indices_.resize(dataset_.size());
        std::iota(indices_.begin(), indices_.end(), 0); // Fill with 0, 1, 2, ...
        reset_iteration();
    }

    void reset_iteration() {
        current_sample_idx_ = 0;
        if (shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
    }

    bool has_next() const {
        return current_sample_idx_ < dataset_.size();
    }

    // Fetches the next batch of data
    std::optional<Sample> next_batch() {
        if (!has_next()) {
            return std::nullopt; // No more data
        }

        size_t actual_batch_size = std::min(batch_size_, dataset_.size() - current_sample_idx_);
        if (actual_batch_size == 0) {
             return std::nullopt;
        }

        std::vector<torch::Tensor> batch_features;
        std::vector<torch::Tensor> batch_labels;
        batch_features.reserve(actual_batch_size);
        batch_labels.reserve(actual_batch_size);

        std::vector<size_t> batch_item_indices;
        batch_item_indices.reserve(actual_batch_size);
        for(size_t i = 0; i < actual_batch_size; ++i) {
            batch_item_indices.push_back(indices_[current_sample_idx_ + i]);
        }

        // --- Data Fetching ---
        // This is a simplified multi-threading model:
        // It parallelizes fetching individual items for the *current* batch.
        // It does NOT prefetch subsequent batches in the background.
        std::vector<Sample> fetched_samples(actual_batch_size);

        if (num_workers_ > 0 && actual_batch_size > 1) {
            std::vector<std::future<Sample>> futures;
            futures.reserve(actual_batch_size);

            for (size_t i = 0; i < actual_batch_size; ++i) {
                size_t data_idx = batch_item_indices[i];
                // std::async can use a thread pool internally or launch new threads.
                // For true control, you'd manage your own std::thread pool.
                futures.emplace_back(
                    std::async(std::launch::async, [this, data_idx]() {
                        return dataset_.get_item(data_idx);
                    })
                );
            }
            for (size_t i = 0; i < actual_batch_size; ++i) {
                fetched_samples[i] = futures[i].get(); // This will block until the future is ready
            }
        } else { // Single-threaded fetching for this batch
            for (size_t i = 0; i < actual_batch_size; ++i) {
                size_t data_idx = batch_item_indices[i];
                fetched_samples[i] = dataset_.get_item(data_idx);
            }
        }
        // --- End Data Fetching ---


        // Collate fetched samples
        for (const auto& sample_pair : fetched_samples) {
            batch_features.push_back(sample_pair.first);
            batch_labels.push_back(sample_pair.second);
        }

        current_sample_idx_ += actual_batch_size;

        if (batch_features.empty()) { // Should not happen if actual_batch_size > 0
            return std::nullopt;
        }

        // Stack tensors to form a batch
        // Ensure all feature tensors have the same shape and all label tensors have the same shape
        // For simplicity, this example assumes they do.
        torch::Tensor features_batch = torch::stack(batch_features, 0);
        torch::Tensor labels_batch = torch::stack(batch_labels, 0).squeeze_(); // Squeeze if labels were [1]

        return {{features_batch, labels_batch}};
    }

    // Basic iterator support to allow range-based for loops
    // This is a very simplified iterator.
    class Iterator {
    public:
        Iterator(MyCustomDataLoader* loader, bool end = false)
            : loader_(loader), is_end_(end || (loader_ && !loader_->has_next())) {}

        Sample operator*() {
            if (!current_batch_val_) { // if not cached or first access
                current_batch_opt_ = loader_->next_batch();
                if (current_batch_opt_) {
                    current_batch_val_ = *current_batch_opt_;
                } else {
                    // This indicates we've truly reached the end if next_batch() returns nullopt
                    is_end_ = true;
                }
            }
            if (!current_batch_opt_) { // Should only happen if loader was empty initially
                 throw std::runtime_error("Attempting to dereference an end iterator or empty loader.");
            }
            return *current_batch_val_;
        }

        Iterator& operator++() {
            if (loader_) {
                 // current_batch_opt_ is populated by the first call to operator* or a previous ++
                 // Now, try to get the next one for the *next* iteration
                current_batch_opt_ = loader_->next_batch();
                if (current_batch_opt_) {
                    current_batch_val_ = *current_batch_opt_;
                } else {
                    is_end_ = true; // Reached the end
                }
            }
            return *this;
        }

        // For the first iteration of range-based for loop, it directly calls operator*
        // then it calls ++ then operator*. So we need to prime the first batch.
        // The `begin()` method will return an iterator that has `next_batch()` called once.
        void prime() {
            if (loader_ && !is_end_) {
                current_batch_opt_ = loader_->next_batch();
                if (current_batch_opt_) {
                    current_batch_val_ = *current_batch_opt_;
                } else {
                    is_end_ = true;
                }
            }
        }


        bool operator!=(const Iterator& other) const {
            // Simplified: check if both are end iterators or if they point to different loaders.
            // A more robust iterator would compare positions.
            if (is_end_ && other.is_end_) return false; // Both at end
            if (is_end_ != other.is_end_) return true;  // One is at end, other is not
            return loader_ != other.loader_; // Should not happen in typical range-for
        }

    private:
        MyCustomDataLoader* loader_;
        bool is_end_;
        std::optional<Sample> current_batch_opt_; // Cache for current batch from next_batch()
        std::optional<Sample> current_batch_val_; // Cache for the value of *current_batch_opt_
    };

    Iterator begin() {
        reset_iteration(); // Reset for a new epoch
        Iterator it(this);
        it.prime(); // Load the first batch
        return it;
    }

    Iterator end() {
        return Iterator(this, true); // Represents the end
    }


private:
    const MyCustomDataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;

    std::vector<size_t> indices_;      // Indices into the dataset_
    size_t current_sample_idx_;      // Current starting sample index for the next batch
};


int main() {
    torch::manual_seed(0);

    // 1. Create and load custom dataset
    MyCustomDataset dataset;
    dataset.load_data(/*num_samples=*/107, /*feature_dim=*/10, /*num_classes=*/5);

    // 2. Create Custom DataLoader
    std::cout << "\n--- Single-threaded DataLoader (shuffle=true) ---\n";
    MyCustomDataLoader data_loader(dataset, /*batch_size=*/32, /*shuffle=*/true, /*num_workers=*/0);

    int epoch = 1;
    std::cout << "Epoch: " << epoch << std::endl;
    int batch_count = 0;
    for (auto batch : data_loader) { // Uses begin() and end()
        torch::Tensor features = batch.first;
        torch::Tensor labels = batch.second;
        std::cout << "Batch " << ++batch_count << ": Features shape: " << features.sizes()
                  << ", Labels shape: " << labels.sizes() << std::endl;
        // Example: Check first label of first batch for shuffling (if reproducible)
        if (batch_count == 1) {
            std::cout << "First label of first batch: " << labels[0].item<long>() << std::endl;
        }
    }
    std::cout << "Total batches: " << batch_count << std::endl;

    std::cout << "\n--- Multi-threaded (num_workers=2) DataLoader (shuffle=false) ---\n";
    MyCustomDataLoader mt_data_loader(dataset, /*batch_size=*/20, /*shuffle=*/false, /*num_workers=*/2);

    epoch = 2;
    std::cout << "Epoch: " << epoch << std::endl;
    batch_count = 0;
    for (auto batch : mt_data_loader) {
        torch::Tensor features = batch.first;
        torch::Tensor labels = batch.second;
        std::cout << "Batch " << ++batch_count << ": Features shape: " << features.sizes()
                  << ", Labels shape: " << labels.sizes() << std::endl;
        if (batch_count == 1) {
             std::cout << "First label of first batch (no shuffle): " << labels[0].item<long>() << std::endl;
        }
    }
    std::cout << "Total batches: " << batch_count << std::endl;

    std::cout << "\n--- Testing edge case: batch_size > dataset_size ---\n";
    MyCustomDataset small_dataset;
    small_dataset.load_data(5, 5, 2);
    MyCustomDataLoader edge_loader(small_dataset, /*batch_size=*/10, /*shuffle=*/false);
    batch_count = 0;
    for (auto batch : edge_loader) {
         std::cout << "Batch " << ++batch_count << ": Features shape: " << batch.first.sizes()
                  << ", Labels shape: " << batch.second.sizes() << std::endl;
    }
     std::cout << "Total batches: " << batch_count << std::endl;


    std::cout << "\n--- Testing empty dataset ---\n";
    MyCustomDataset empty_dataset;
    // empty_dataset.load_data(0, 5, 2); // No data loaded
    MyCustomDataLoader empty_loader(empty_dataset, /*batch_size=*/10, /*shuffle=*/false);
    batch_count = 0;
    try {
        for (auto batch : empty_loader) { // This should not loop
            std::cout << "Batch " << ++batch_count << ": Features shape: " << batch.first.sizes()
                    << ", Labels shape: " << batch.second.sizes() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Caught exception with empty dataset: " << e.what() << std::endl;
    }
    std::cout << "Total batches with empty dataset: " << batch_count << std::endl;


    return 0;
}