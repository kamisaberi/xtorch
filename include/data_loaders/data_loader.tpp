#pragma once

#include "data_loader.h"

/**
 * @namespace xt
 * @brief Implementation of custom data loading utilities for the xt framework.
 *
 * This namespace contains the template implementations of the DataLoader class
 * functionality declared in data_loader.h.
 */
namespace xt {

    /**
     * @brief Constructs a DataLoader instance with the specified dataset and options
     *
     * @tparam Dataset The dataset type (must satisfy torch::data::Dataset requirements)
     * @param dataset The dataset to load (will be moved into the loader)
     * @param options Configuration options including:
     *                - batch_size: Number of samples per batch
     *                - workers: Must be 0 (single-threaded only)
     *                - drop_last: Whether to discard incomplete final batches
     * @param shuffle If true, shuffles sample indices between epochs
     * @throws std::runtime_error If workers option is not 0
     *
     * Initializes the data loader with:
     * - Dataset ownership transfer to base class
     * - Batch size and drop_last configuration
     * - Initial index sequence generation
     *
     * @note This implementation currently only supports single-threaded operation
     */
    template <typename Dataset>
    DataLoader<Dataset>::DataLoader(Dataset dataset, const torch::data::DataLoaderOptions& options, bool shuffle)
        : Base(options, std::make_unique<Dataset>(std::move(dataset))), shuffle_(shuffle) {
        // Enforce single-threaded operation
        if (options.workers() != 0) {
            throw std::runtime_error("CustomDataLoader supports only workers=0 (single-threaded)");
        }

        // Initialize member variables
        dataset_ptr_ = Base::main_thread_dataset_.get();  // Get pointer to managed dataset
        batch_size_  = options.batch_size();              // Store configured batch size
        drop_last_   = options.drop_last();               // Store drop_last setting
        reset_indices();                                  // Generate initial index sequence
    }

    /**
     * @brief Generates the next batch request during iteration
     * @return std::optional<BatchRequestType> Batch indices or nullopt if iteration complete
     *
     * The method:
     * 1. Checks for remaining indices
     * 2. Calculates batch boundaries
     * 3. Handles incomplete batches according to drop_last_
     * 4. Advances the iteration position
     *
     * @note Called automatically by the base class during iteration
     */
    template <typename Dataset>
    std::optional<typename DataLoader<Dataset>::BatchRequestType> DataLoader<Dataset>::get_batch_request() {
        // Check for end of iteration
        if (current_index_ >= indices_.size()) {
            return std::nullopt;
        }

        // Calculate batch boundaries
        size_t start_index = current_index_;
        size_t end_index   = std::min(current_index_ + batch_size_, indices_.size());

        // Handle incomplete final batch
        if (drop_last_ && (end_index - start_index) < batch_size_) {
            return std::nullopt;
        }

        // Extract batch indices and advance position
        BatchRequestType batch_indices(indices_.begin() + start_index, indices_.begin() + end_index);
        current_index_ = end_index;

        return batch_indices;
    }

    /**
     * @brief Reinitializes the index sequence for a new epoch
     *
     * Creates a sequential index list (0 to N-1) and optionally shuffles it.
     * Resets the iteration position to start of sequence.
     *
     * @note Uses a fixed-seed RNG for reproducible shuffling when enabled
     */
    template <typename Dataset>
    void DataLoader<Dataset>::reset_indices() {
        const size_t N = dataset_ptr_->size().value();
        indices_.resize(N);

        // Fill with sequential indices
        std::iota(indices_.begin(), indices_.end(), 0);

        // Apply shuffling if enabled
        if (shuffle_) {
            static std::mt19937 rng(std::random_device{}());
            std::shuffle(indices_.begin(), indices_.end(), rng);
        }

        current_index_ = 0;
    }

    /**
     * @brief Prepares the loader for a new epoch
     *
     * Combines:
     * 1. Index sequence regeneration (with optional shuffling)
     * 2. Base class state reset
     *
     * @note Called automatically at start of each epoch
     */
    template <typename Dataset>
    void DataLoader<Dataset>::reset() {
        reset_indices();  // Regenerate index sequence
        Base::reset();    // Reset base class iterator state
    }

}