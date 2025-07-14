#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class ColorJitter
     * @brief Randomly changes the brightness, contrast, saturation, and hue of an image.
     *
     * This transform applies a sequence of random color adjustments. The order of
     * the adjustments (brightness, contrast, saturation, hue) is randomized for
     * each image.
     */
    class ColorJitter : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a moderate, commonly used set of jitters.
         */
        ColorJitter();

        /**
         * @brief Constructs the ColorJitter transform.
         *
         * @param brightness Optional pair `{min, max}` for the brightness adjustment factor.
         *                   Factors are chosen from this range. A factor of 1.0 is no change.
         * @param contrast Optional pair `{min, max}` for the contrast adjustment factor.
         * @param saturation Optional pair `{min, max}` for the saturation adjustment factor.
         * @param hue Optional float `h` in `[0, 0.5]`. The hue shift will be chosen
         *            randomly from `[-h, h]`.
         */
        explicit ColorJitter(
            std::optional<std::pair<double, double>> brightness = std::nullopt,
            std::optional<std::pair<double, double>> contrast = std::nullopt,
            std::optional<std::pair<double, double>> saturation = std::nullopt,
            std::optional<double> hue = std::nullopt
        );

        /**
         * @brief Executes the color jittering.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with float values in the range [0, 1].
         * @return An std::any containing the resulting jittered torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::optional<std::pair<double, double>> brightness_;
        std::optional<std::pair<double, double>> contrast_;
        std::optional<std::pair<double, double>> saturation_;
        std::optional<double> hue_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image