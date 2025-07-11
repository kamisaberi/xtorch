#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class CutMix
     * @brief An image and label transformation that applies the CutMix augmentation.
     *
     * CutMix is a data augmentation strategy that creates composite images by
     * cutting a patch from one image and pasting it onto another. The labels
     * are mixed proportionally to the area of the patch.
     *
     * This transform must be applied to an entire batch of images and labels.
     */
    class CutMix : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses alpha=1.0 and applies transform 50% of the time.
         */
        CutMix();

        /**
         * @brief Constructs the CutMix transform.
         * @param alpha The alpha parameter for the Beta(alpha, alpha) distribution,
         *              which determines the patch size. A value of 1.0 is common.
         * @param p The probability of applying CutMix to a given batch.
         */
        CutMix(float alpha, float p = 0.5f);

        /**
         * @brief Executes the CutMix operation on a batch of data.
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

        // Helper to generate the random bounding box coordinates
        torch::Tensor _generate_bbox(int64_t H, int64_t W, double lambda);
    };

} // namespace xt::transforms::image