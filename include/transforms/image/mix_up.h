#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class Mixup
     * @brief An image and label transformation that applies the Mixup augmentation.
     *
     * Mixup creates composite training examples by performing a linear interpolation
     * between two random images and their corresponding labels from a batch. This
     * encourages the model to learn smoother decision boundaries and improves
     * generalization.
     *
     * This transform must be applied to an entire batch of images and labels.
     */
    class Mixup : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses alpha=0.4 and applies transform 50% of the time.
         */
        Mixup();

        /**
         * @brief Constructs the Mixup transform.
         * @param alpha The alpha parameter for the Beta(alpha, alpha) distribution,
         *              which determines the interpolation factor (lambda). A value
         *              of 0.4 is common.
         * @param p The probability of applying Mixup to a given batch.
         */
        Mixup(float alpha, float p = 0.5f);

        /**
         * @brief Executes the Mixup operation on a batch of data.
         * @param tensors An initializer list containing exactly two tensors:
         *                1. A batch of images (4D, [B, C, H, W])
         *                2. A batch of labels (1D for integer labels, or 2D for one-hot)
         * @return An std::any containing a std::pair<torch::Tensor, torch::Tensor>:
         *         {mixed_images_batch, mixed_labels_batch}.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float alpha_;
        float p_;
    };

} // namespace xt::transforms::image