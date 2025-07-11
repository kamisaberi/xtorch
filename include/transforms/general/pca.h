#pragma once
#include "../common.h"


namespace xt::transforms::general {

    /**
     * @class PCA
     * @brief A stateful transform for Principal Component Analysis.
     *
     * This transform reduces the dimensionality of data by projecting it onto
     * a smaller set of principal components.
     *
     * USAGE WORKFLOW:
     * 1. Create a PCA object with the desired number of components.
     * 2. Call the `fit()` method on your entire training dataset (or a large batch)
     *    to learn the components.
     * 3. Use the fitted object as a regular transform in your pipeline. The `forward()`
     *    method will then apply the learned transformation to new data.
     */
    class PCA : public xt::Module {
    public:
        PCA();

        /**
         * @brief Constructor for PCA.
         * @param n_components The number of principal components to keep.
         * @param pre_transforms An optional vector of transforms to apply to the data
         *                       before fitting or transforming.
         */
        explicit PCA(int n_components, std::vector<xt::Module*> pre_transforms = {});

        // This constructor is kept to match your original request, interpreted as
        // a PCA with default components and a pre-processing chain.
        explicit PCA(std::vector<xt::Module*> transforms);

        /**
         * @brief Learns the principal components from the provided data.
         * @param data A 2D tensor of shape [n_samples, n_features].
         */
        void fit(torch::Tensor data);

        /**
         * @brief Projects the input data onto the learned principal components.
         * @param tensors An initializer list containing a tensor to be transformed.
         * @return A std::any containing the dimension-reduced torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        /**
         * @brief Returns true if the PCA model has been fitted.
         */
        bool is_fitted() const;

    private:
        // Helper to apply pre-processing transforms
        torch::Tensor _apply_pre_transforms(torch::Tensor data);

        // State learned during fitting
        int n_components_ = -1;
        bool is_fitted_ = false;
        torch::Tensor mean_;
        torch::Tensor components_; // Shape: [n_components, n_features]

        // Pre-processing transforms
        std::vector<xt::Module*> pre_transforms_;
    };

} // namespace xt::transforms::general