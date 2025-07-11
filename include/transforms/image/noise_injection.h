#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class NoiseInjection
     * @brief A flexible transform to add random noise from various distributions to an image.
     *
     * This data augmentation technique improves model robustness by simulating
     * various types of noise. It supports Gaussian and Uniform noise distributions.
     */
    class NoiseInjection : public xt::Module {
    public:
        /**
         * @brief Default constructor. Defaults to Gaussian noise with mean=0, sigma=0.1.
         */
        NoiseInjection();

        /**
         * @brief Constructs the NoiseInjection transform.
         * @param noise_type A string specifying the noise distribution.
         *                   Can be "gaussian" or "uniform".
         * @param params A vector of parameters for the chosen distribution.
         *               For "gaussian": {mean, sigma}.
         *               For "uniform": {low, high}.
         */
        NoiseInjection(const std::string& noise_type, const std::vector<double>& params);

        /**
         * @brief Executes the noise injection operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting noisy torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::string noise_type_;
        std::vector<double> params_;
    };

} // namespace xt::transforms::image