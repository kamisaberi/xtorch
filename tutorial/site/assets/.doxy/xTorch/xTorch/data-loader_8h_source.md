

# File data-loader.h

[**File List**](files.md) **>** [**data-loaders**](dir_83fc80326b80dce73692b3bd70d110c8.md) **>** [**data-loader.h**](data-loader_8h.md)

[Go to the documentation of this file](data-loader_8h.md)


```C++
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
#include "../datasets/base/base.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt {

    template <typename Dataset>
    bool is_transformed_dataset(const Dataset& dataset) {
        if constexpr (std::is_same_v<Dataset, torch::data::datasets::MapDataset<xt::data::datasets::BaseDataset, torch::data::transforms::Stack<>>>) {
            return true;
        } else {
            return false;
        }
    }


template <typename Dataset>
class DataLoader : public torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<size_t>> {

public:
    using BatchType        = typename Dataset::BatchType;          // e.g., Example<Tensor, Tensor>
    using BatchRequestType = std::vector<size_t>;                  // list of indices for one batch
    using Base = torch::data::DataLoaderBase<Dataset, BatchType, BatchRequestType>;
    DataLoader(Dataset dataset, const torch::data::DataLoaderOptions& options, bool shuffle = false);

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
    std::optional<BatchRequestType> get_batch_request() override ;
    // Reset and (optionally) shuffle indices for a new epoch
    void reset_indices();
    // Override base class reset() to shuffle indices each epoch (if enabled)
    void reset() override;
private:
    Dataset* dataset_ptr_;           // raw pointer to the dataset (owned by base class)
    std::vector<size_t> indices_;    // sequence of indices to iterate over
    size_t current_index_ = 0;       // current position in indices_ vector
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
};


}
#include "data-loader.tpp"

```


