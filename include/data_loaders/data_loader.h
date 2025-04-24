#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <random>
#include <optional>
#include "../datasets/base/base.h"
using namespace std;
namespace fs = std::filesystem;

/**
 * @namespace xt
 * @brief Namespace for custom data loading utilities and dataset handling in the xt framework.
 */
namespace xt {

    /**
     * @brief Checks if a dataset is a transformed dataset of type MapDataset with BaseDataset and Stack transform.
     *
     * @tparam Dataset The type of the dataset to check.
     * @param dataset The dataset instance to evaluate.
     * @return bool True if the dataset is a transformed MapDataset, false otherwise.
     */
    template <typename Dataset>
    bool is_transformed_dataset(const Dataset& dataset) {
        if constexpr (std::is_same_v<Dataset, torch::data::datasets::MapDataset<xt::data::datasets::BaseDataset, torch::data::transforms::Stack<>>>) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * @brief A custom data loader for iterating over datasets in batches, supporting shuffling and index-based batch requests.
     *
     * @tparam Dataset The type of the dataset to load.
     */
    template <typename Dataset>
    class DataLoader : public torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<size_t>> {

    public:
        using BatchType        = typename Dataset::BatchType;          ///< Type of a single batch (e.g., Example<Tensor, Tensor>).
        using BatchRequestType = std::vector<size_t>;                  ///< Type for batch index requests (list of indices).
        using Base = torch::data::DataLoaderBase<Dataset, BatchType, BatchRequestType>; ///< Base class type.

        /**
         * @brief Constructs a DataLoader for the given dataset with specified options.
         *
         * @param dataset The dataset to load.
         * @param options DataLoader options (e.g., batch size, number of workers).
         * @param shuffle Whether to shuffle the dataset indices each epoch.
         */
        DataLoader(Dataset dataset, const torch::data::DataLoaderOptions& options, bool shuffle = false);

        // Iterator support for range-for loops
        // typename Base::iterator begin() {
        //     this->reset();       // reset (and shuffle if needed) at start of epoch
        //     return Base::begin();
        // }
        // typename Base::iterator end() {
        //     return Base::end();
        // }

    protected:
        /**
         * @brief Provides the next batch of indices to fetch from the dataset.
         *
         * @return std::optional<BatchRequestType> A vector of indices for the next batch, or nullopt if no more batches are available.
         */
        std::optional<BatchRequestType> get_batch_request() override;

        /**
         * @brief Resets and optionally shuffles the indices for a new epoch.
         */
        void reset_indices();

        /**
         * @brief Resets the DataLoader for a new epoch, shuffling indices if enabled.
         */
        void reset() override;

    private:
        Dataset* dataset_ptr_;           ///< Raw pointer to the dataset (owned by base class).
        std::vector<size_t> indices_;    ///< Sequence of indices to iterate over.
        size_t current_index_ = 0;       ///< Current position in the indices_ vector.
        size_t batch_size_;              ///< Size of each batch.
        bool shuffle_;                   ///< Whether to shuffle indices each epoch.
        bool drop_last_;                 ///< Whether to drop the last incomplete batch.
    };

}

#include "data_loader.tpp"