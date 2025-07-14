#pragma once


#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Grayscale
     * @brief Converts an image to grayscale.
     *
     * This transform converts an RGB image to its single-channel luminance
     * representation. It can be configured to either return a single-channel
     * image or a 3-channel image where R, G, and B are all equal to the
     * calculated luminance.
     */
    class Grayscale : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates a grayscale transform that outputs a
         *        3-channel grayscale image.
         */
        Grayscale();

        /**
         * @brief Constructs the Grayscale transform.
         *
         * @param num_output_channels The number of channels in the output image.
         *                            Must be 1 or 3.
         */
        explicit Grayscale(int num_output_channels);

        /**
         * @brief Executes the grayscale conversion.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting grayscale torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_output_channels_;
    };

} // namespace xt::transforms::image