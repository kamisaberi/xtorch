#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class Solarize
     * @brief An image transformation that inverts pixel values above a given threshold.
     *
     * This transform creates a "solarized" effect, which is a common artistic
     * filter and can be used for data augmentation. Any pixel with a value
     * greater than the threshold will be inverted (e.g., value -> 1.0 - value).
     */
    class Solarize : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a threshold of 0.5.
         */
        Solarize();

        /**
         * @brief Constructs the Solarize transform.
         * @param threshold The intensity value (typically between 0.0 and 1.0)
         *                  above which pixels will be inverted.
         */
        explicit Solarize(float threshold);

        /**
         * @brief Executes the solarization operation.
         * @param tensors An initializer list expected to contain a single image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting solarized torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float threshold_;
    };

} // namespace xt::transforms::image