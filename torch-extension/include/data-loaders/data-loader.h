#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <optional>

using namespace std;
namespace fs = std::filesystem;


namespace xt {


template <typename Dataset>
class CustomDataLoader : public torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<size_t>> {
    using BatchType        = typename Dataset::BatchType;          // e.g., Example<Tensor, Tensor>
    using BatchRequestType = std::vector<size_t>;                  // list of indices for one batch
    using Base = torch::data::DataLoaderBase<Dataset, BatchType, BatchRequestType>;

public:
    CustomDataLoader(Dataset dataset, const torch::data::DataLoaderOptions& options, bool shuffle = false)
        : Base(options, std::make_unique<Dataset>(std::move(dataset))), shuffle_(shuffle) {
        // Only single-thread (workers=0) is supported in this custom loader
        if (options.workers() != 0) {
            throw std::runtime_error("CustomDataLoader supports only workers=0 (single-threaded)");
        }
        dataset_ptr_ = Base::main_thread_dataset_.get();      // pointer to dataset (stored in base)
        batch_size_  = options.batch_size();                  // batch size per iteration
        drop_last_   = options.drop_last();                   // whether to drop last incomplete batch
        reset_indices();                                      // initialize index sequence
    }

//    // Iterator support for range-for loops
//    typename Base::iterator begin() {
//        this->reset();       // reset (and shuffle if needed) at start of epoch
//        return Base::begin();
//    }
//    typename Base::iterator end() {
//        return Base::end();
//    }

protected:
    // Provide the next batch of indices to fetch from the dataset
    std::optional<BatchRequestType> get_batch_request() override {
        if (current_index_ >= indices_.size()) {
            // No more indices -> signal end of data
            return std::nullopt;
        }
        // Determine the range [start_index, end_index) for the next batch of indices
        size_t start_index = current_index_;
        size_t end_index   = std::min(current_index_ + batch_size_, indices_.size());
        // If drop_last_ is true and the remaining indices are fewer than batch_size, stop here
        if (drop_last_ && (end_index - start_index) < batch_size_) {
            return std::nullopt;
        }
        // Collect indices for this batch and advance the pointer
        BatchRequestType batch_indices(indices_.begin() + start_index, indices_.begin() + end_index);
        current_index_ = end_index;
        return batch_indices;
    }

    // Reset and (optionally) shuffle indices for a new epoch
    void reset_indices() {
        const size_t N = dataset_ptr_->size().value();
        indices_.resize(N);
        std::iota(indices_.begin(), indices_.end(), 0);  // fill with 0,1,...,N-1
        if (shuffle_) {
            // Shuffle the indices to randomize batch order
            static std::mt19937 rng(std::random_device{}());  // fixed seeded RNG for reproducibility
            std::shuffle(indices_.begin(), indices_.end(), rng);
        }
        current_index_ = 0;
    }

    // Override base class reset() to shuffle indices each epoch (if enabled)
    void reset() override {
        reset_indices();
        Base::reset();  // let DataLoaderBase handle internal reset (e.g., for iterator state)
    }


private:
    Dataset* dataset_ptr_;           // raw pointer to the dataset (owned by base class)
    std::vector<size_t> indices_;    // sequence of indices to iterate over
    size_t current_index_ = 0;       // current position in indices_ vector
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
};



}