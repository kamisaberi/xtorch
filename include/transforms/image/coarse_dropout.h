#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class CoarseDropout
     * @brief An image transformation that randomly erases rectangular regions in an image.
     *
     * Also known as Cutout, this is a strong regularization technique that forces
     * the model to learn more robust features by preventing it from relying on any
     * single small region of the input.
     */
    class CoarseDropout : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters.
         */
        CoarseDropout();

        /**
         * @brief Constructs the CoarseDropout transform with specific parameters.
         *
         * @param max_holes The maximum number of rectangular holes to create.
         * @param max_height The maximum height of each hole, as a fraction of image height.
         * @param max_width The maximum width of each hole, as a fraction of image width.
         * @param min_holes The minimum number of holes to create. Defaults to 1.
         * @param min_height The minimum height of each hole. Defaults to max_height.
         * @param min_width The minimum width of each hole. Defaults to max_width.
         * @param fill_value The value to fill the erased regions with. Defaults to 0 (black).
         */
        CoarseDropout(
            int max_holes,
            float max_height,
            float max_width,
            int min_holes = 1,
            float min_height = -1.0f,
            float min_width = -1.0f,
            float fill_value = 0.0f
        );

        /**
         * @brief Executes the CoarseDropout operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with rectangular
         *         regions erased. The shape and type remain the same.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int max_holes_;
        float max_height_;
        float max_width_;
        int min_holes_;
        float min_height_;
        float min_width_;
        float fill_value_;
    };

} // namespace xt::transforms::image