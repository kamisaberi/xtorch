#pragma once



#include "../headers/transforms.h"

namespace xt::data::transforms {



    /**
     * @struct GrayscaleToRGB
     * @brief A functor to convert a grayscale tensor to an RGB tensor.
     *
     * This struct provides a callable object that transforms a grayscale tensor, typically with a single
     * channel (e.g., [H, W] or [1, H, W]), into an RGB tensor with three channels (e.g., [3, H, W]).
     * The conversion is performed by replicating the grayscale channel across the RGB dimensions,
     * suitable for preprocessing grayscale images in machine learning workflows using LibTorch.
     */
    struct GrayscaleToRGB {
    public:
        /**
         * @brief Converts a grayscale tensor to an RGB tensor.
         * @param tensor The input grayscale tensor, expected in format [H, W] or [1, H, W].
         * @return A new tensor in RGB format [3, H, W], with the grayscale values replicated across channels.
         *
         * This operator takes a grayscale tensor and produces an RGB tensor by duplicating the single
         * channel’s values into three identical channels (red, green, blue). The input tensor must have
         * a single channel, either as a 2D tensor [H, W] or a 3D tensor with one channel [1, H, W].
         */
        torch::Tensor operator()(const torch::Tensor &tensor);
    };

    struct Grayscale {
    public:
        Grayscale();

        torch::Tensor operator()(torch::Tensor input);
    };

    struct ToGray {
        torch::Tensor operator()(const torch::Tensor& color_tensor) const;
    };


}