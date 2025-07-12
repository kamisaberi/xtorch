#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomAugment
     * @brief An image transformation that applies a sequence of randomly selected
     *        augmentations from a predefined list.
     *
     * This transform implements the RandAugment policy. For each image, it
     * randomly samples `num_ops` transformations from a pool of augmentations
     * and applies them sequentially. The strength of each transformation is
     * controlled by a single `magnitude` parameter.
     */
    class RandomAugment : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies 2 operations with a magnitude of 9,
         *        which are common default values.
         */
        RandomAugment();

        /**
         * @brief Constructs the RandomAugment transform.
         *
         * @param num_ops The number of augmentation operations to apply sequentially.
         * @param magnitude The overall magnitude for all operations, an integer
         *                  typically in the range [0, 30].
         * @param fill A vector representing the color to fill new areas with during
         *             geometric transformations. Should be in the [0, 1] range for
         *             float tensors. E.g., `{0.0, 0.0, 0.0}` for black.
         * @param interpolation The interpolation method for geometric transforms.
         *                      Supported: "bilinear" (default), "nearest".
         */
        explicit RandomAugment(
            int num_ops,
            int magnitude,
            const std::vector<double>& fill = {0.0, 0.0, 0.0},
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the RandomAugment policy.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting augmented torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Populates the map of available augmentations and their implementations.
        void initialize_augmentation_space();

        int num_ops_;
        int magnitude_;
        cv::Scalar fill_color_;
        int interpolation_flag_;

        std::mt19937 gen_;
        std::vector<std::string> op_names_;
        // A map from operation name to a function that applies it.
        // The function takes the image tensor and a signed level [-1, 1].
        std::map<std::string, std::function<torch::Tensor(torch::Tensor, double)>> augmentation_space_;
    };

} // namespace xt::transforms::image