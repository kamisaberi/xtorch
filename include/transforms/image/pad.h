#pragma once


#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class Pad
     * @brief An image transformation that adds padding to the borders of an image.
     *
     * This transform can add a border of a specified size and value around an
     * image, using various modes like constant, reflection, or replication.
     * This implementation uses OpenCV for robust and stable padding operations.
     */
    class Pad : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        Pad();

        /**
         * @brief Constructs the Pad transform.
         * @param padding A vector defining the padding for each side. It can have:
         *                - 1 element: a single padding for all sides.
         *                - 2 elements: {left/right, top/bottom}.
         *                - 4 elements: {left, right, top, bottom}.
         * @param mode The padding mode. Can be "constant", "reflect", "replicate".
         * @param fill_value The value to use for padding if mode is "constant".
         */
        Pad(const std::vector<int>& padding, // Use int for OpenCV compatibility
            const std::string& mode = "constant",
            float fill_value = 0.0f);

        /**
         * @brief Executes the padding operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting padded torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Using int for OpenCV compatibility
        int top_, bottom_, left_, right_;
        int border_type_flag_; // OpenCV border type flag
        float fill_value_;
    };

} // namespace xt::transforms::image