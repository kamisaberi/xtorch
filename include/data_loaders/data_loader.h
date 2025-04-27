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
 * @brief Extended Tensor (xt) framework namespace providing enhanced data loading capabilities
 *
 * The xt namespace contains custom implementations of data loading utilities that extend
 * PyTorch's native functionality with additional features and optimizations while maintaining
 * compatibility with standard PyTorch datasets and transforms.
 */
namespace xt {

    /**
     * @brief Compile-time check for transformed MapDataset with Stack transform
     * @tparam Dataset Type of dataset to check
     * @param dataset Dataset instance (unused in evaluation)
     * @return constexpr bool True if Dataset is MapDataset<BaseDataset, Stack<>>, false otherwise
     *
     * This type trait is particularly useful for:
     * - Validating dataset transformations in template code
     * - Enabling different code paths based on dataset type
     * - Ensuring proper collation of samples via Stack transform
     *
     * @note The dataset parameter is unused in evaluation since this is a compile-time check
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
     * @class DataLoader
     * @brief Enhanced data loader with extended batch control and shuffling capabilities
     * @tparam Dataset Type of dataset to load, must satisfy torch::data::Dataset requirements
     *
     * This implementation provides:
     * - Fine-grained control over batch composition through index-based requests
     * - Optional epoch-level shuffling of samples
     * - Configurable handling of incomplete batches
     * - Thread-safe iteration when used with multiple workers
     * - Seamless integration with standard PyTorch datasets and transforms
     *
     * @note Inherits from torch::data::DataLoaderBase to maintain compatibility with
     *       PyTorch's data loading ecosystem while adding custom functionality.
     */
    template <typename Dataset>
    class DataLoader : public torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<size_t>> {

    public:
        /// Type representing a single batch of data (typically torch::data::Example<Tensor, Tensor>)
        using BatchType = typename Dataset::BatchType;

        /// Type used to request specific batches (vector of sample indices)
        using BatchRequestType = std::vector<size_t>;

        /// Base class type alias for cleaner inheritance
        using Base = torch::data::DataLoaderBase<Dataset, BatchType, BatchRequestType>;

        /**
         * @brief Constructs a DataLoader instance
         * @param dataset The dataset to load (moved into the loader)
         * @param options Configuration options including:
         *                - batch_size: Number of samples per batch
         *                - workers: Number of parallel data loading workers
         *                - timeout: Maximum time to wait for a batch
         *                - enforce_ordering: Whether to preserve sample order
         * @param shuffle If true, shuffles sample indices between epochs
         *
         * The constructor initializes the data loading pipeline and prepares
         * the initial index sequence based on the dataset size and options.
         */
        DataLoader(Dataset dataset, const torch::data::DataLoaderOptions& options, bool shuffle = false);

    protected:
        /**
         * @brief Generates the next batch request during iteration
         * @return std::optional<BatchRequestType> Indices for next batch, or nullopt if epoch complete
         *
         * Key responsibilities:
         * - Manages batch size enforcement
         * - Handles dataset boundary conditions
         * - Respects drop_last setting for incomplete batches
         * - Updates iteration state
         *
         * @note This override provides the core batch generation logic and is called
         *       automatically during iteration.
         */
        std::optional<BatchRequestType> get_batch_request() override;

        /**
         * @brief Rebuilds the index sequence for a new epoch
         *
         * Performs:
         * - Regeneration of complete index sequence
         * - Optional shuffling if enabled
         * - Reset of iteration state
         *
         * @note Called automatically at start of each epoch
         */
        void reset_indices();

        /**
         * @brief Prepares the loader for a new epoch
         *
         * This override:
         * - Ensures proper initialization/reset of base class state
         * - Delegates to reset_indices() for index management
         * - Maintains consistency with PyTorch's data loading semantics
         */
        void reset() override;

    private:
        /// Non-owning pointer to managed dataset (owned by base class)
        Dataset* dataset_ptr_;

        /// Current epoch's index sequence (may be shuffled)
        std::vector<size_t> indices_;

        /// Current position in index sequence
        size_t current_index_ = 0;

        /// Configured batch size
        size_t batch_size_;

        /// Shuffle flag (controls whether to shuffle between epochs)
        bool shuffle_;

        /// Whether to drop incomplete final batch
        bool drop_last_;
    };

}

#include "data_loader.tpp"