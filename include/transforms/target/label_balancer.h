#pragma once

#include "../common.h"


#include <torch/torch.h> // Assumes LibTorch is installed
#include <vector>
#include <any>
#include <random> // For random number generation

namespace xt::transforms::target {

    /**
     * @brief The strategy to use for balancing the dataset.
     */
    enum class BalancingStrategy {
        OVERSAMPLE,  // Randomly duplicate samples from minority classes.
        UNDERSAMPLE  // Randomly remove samples from majority classes.
    };


    /**
     * @class LabelBalancer
     * @brief A dataset-level transformation that balances class representation by
     *        oversampling minority classes or undersampling majority classes.
     *
     * This transform takes an entire dataset (features and labels) and outputs
     * a new dataset where each class has the same number of samples.
     */
    class LabelBalancer : public xt::Module {
    public:
        /**
         * @brief Constructs the LabelBalancer transform.
         * @param strategy The balancing strategy to use. Defaults to OVERSAMPLE.
         */
        explicit LabelBalancer(BalancingStrategy strategy = BalancingStrategy::OVERSAMPLE);

        /**
         * @brief Executes the balancing operation on a full dataset.
         * @param tensors An initializer list expected to contain exactly two items:
         *                1. A `torch::Tensor` of features (shape: [num_samples, ...features]).
         *                2. A `torch::Tensor` of labels (shape: [num_samples]).
         * @return An std::any containing a `std::pair<torch::Tensor, torch::Tensor>`
         *         with the new, balanced features and labels.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        BalancingStrategy strategy_;
        std::mt19937 random_engine_;
    };

} // namespace xt::transforms::target