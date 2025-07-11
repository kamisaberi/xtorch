#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class PadIfNeeded
     * @brief A transform that pads an image only if its dimensions are smaller
     *        than a specified minimum size.
     *
     * This is useful for ensuring all images in a dataset meet a minimum size
     * requirement before further processing (like cropping). Padding is applied
     * symmetrically to keep the content centered.
     */
    class PadIfNeeded : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        PadIfNeeded();

        /**
         * @brief Constructs the PadIfNeeded transform.
         * @param min_height The minimum required height for the image.
         * @param min_width The minimum required width for the image.
         * @param border_mode The padding mode. Can be "constant", "reflect", "replicate".
         * @param fill_value The value to use for padding if mode is "constant".
         */
        PadIfNeeded(
            int min_height,
            int min_width,
            const std::string& border_mode = "constant",
            float fill_value = 0.0f
        );

        /**
         * @brief Executes the conditional padding operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting padded (or original) torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int min_height_;
        int min_width_;
        int border_type_flag_; // OpenCV border type flag
        float fill_value_;
    };

} // namespace xt::transforms::image