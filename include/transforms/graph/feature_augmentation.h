#pragma once

#include "../common.h"

namespace xt::transforms::graph {

    /**
     * @enum FeatureAugmentationType
     * @brief Defines the type of augmentation to apply to node features.
     */
    enum class FeatureAugmentationType {
        Noise,  // Add Gaussian noise
        Mask    // Mask (zero out) entire feature dimensions
    };

    /**
     * @class FeatureAugmentation
     * @brief A graph transform that augments node features by adding noise or masking.
     *
     * This is a regularization technique for GNNs that helps prevent overfitting to
     * specific node feature values.
     */
    class FeatureAugmentation : public xt::Module {
    public:
        /**
         * @brief Constructs the FeatureAugmentation transform.
         *
         * @param aug_type The type of augmentation to perform ("noise" or "mask").
         * @param strength The strength of the augmentation.
         *                 - For "noise": The standard deviation of the Gaussian noise.
         *                 - For "mask": The fraction of feature dimensions to mask.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit FeatureAugmentation(
                const std::string& aug_type,
                double strength,
                double p = 1.0);

        /**
         * @brief Executes the node feature augmentation.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new graph `{new_node_features, edge_index}`. The
         *         edge_index is unchanged.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        FeatureAugmentationType aug_type_;
        double strength_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph