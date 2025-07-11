#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class AdvancedBlur
     * @brief An advanced image transformation that applies adaptive thresholding to
     *        create a pure black-and-white (binary) image.
     *
     * Unlike global thresholding, this method calculates a different threshold for
     * each pixel based on the brightness of its local neighborhood. This is highly
     * effective for images with varying lighting conditions. The output contains
     * only values of 0 and 1.
     */
    class AdvancedBlur : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters.
         */
        AdvancedBlur();

        /**
         * @brief Constructs the AdvancedBlur transform.
         * @param block_size The size of the local neighborhood used to calculate
         *                   the threshold (must be an odd integer > 1).
         * @param c A constant subtracted from the mean or weighted mean. Normally,
         *          it is positive but may be zero or negative as well.
         * @param adaptive_method The adaptive thresholding algorithm to use.
         *                        Can be "mean" or "gaussian".
         */
        AdvancedBlur(int block_size, double c, const std::string& adaptive_method = "gaussian");

        /**
         * @brief Executes the adaptive thresholding operation.
         * @param tensors An initializer list expected to contain a single image tensor.
         *                If 3-channel, it's converted to grayscale first.
         * @return An std::any containing the resulting binary torch::Tensor of
         *         shape [1, H, W] with values of 0 or 1.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int block_size_;
        double c_;
        int adaptive_method_flag_; // OpenCV flag for the method
    };

} // namespace xt::transforms::image