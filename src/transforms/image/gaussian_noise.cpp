#include "../../../include/transforms/image/gaussian_noise.h"

namespace xt::data::transforms {





    GaussianNoise::GaussianNoise(float mean, float std) : mean(mean), std(std) {
        if (std < 0) {
            throw std::invalid_argument("Standard deviation must be non-negative.");
        }
    }

    torch::Tensor GaussianNoise::operator()(torch::Tensor input) {
        // Generate noise ~ N(0, 1) with the same shape as input
        torch::Tensor noise = torch::randn_like(input, torch::TensorOptions()
                                                .dtype(input.dtype())
                                                .device(input.device()));

        // Scale by std and shift by mean, then add to input
        return input + (noise * std + mean);
    }





}