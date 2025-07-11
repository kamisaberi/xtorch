#pragma once

#include "../common.h"



namespace xt::transforms::general {

    /**
     * @class LatentProjection
     * @brief A transform that projects input data into a latent space using a given encoder model.
     *
     * This module wraps a neural network model (the "encoder") and uses it to
     * transform raw input data (like images) into their lower-dimensional
     * latent representations. This is useful for creating modular pipelines where
     * feature extraction is a distinct pre-processing step.
     */
    class LatentProjection : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an uninitialized (and unusable) transform.
         */
        LatentProjection();

        /**
         * @brief Constructs the LatentProjection transform.
         * @param encoder A shared pointer to a pre-trained model (or any torch::nn::Module)
         *                that will be used to encode the input data. The model is not
         *                copied; this transform holds a reference to it.
         */
        explicit LatentProjection(std::shared_ptr<xt::Module> encoder);

        /**
         * @brief Executes the projection by passing the input through the encoder.
         * @param tensors An initializer list expected to contain a single batch of
         *                data (e.g., an image tensor of shape [B, C, H, W]).
         * @return An std::any containing the resulting batch of latent vectors
         *         (e.g., a torch::Tensor of shape [B, LatentDim]).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        /**
         * @brief Sets the internal encoder model to evaluation mode.
         */
        void eval();

        /**
         * @brief Sets the internal encoder model to training mode.
         */
        void train();

    private:
        // Use a shared_ptr to manage the lifetime of the model.
        // The transform doesn't "own" the model, it just uses it.
        std::shared_ptr<xt::Module> encoder_;
    };

} // namespace xt::transforms::general