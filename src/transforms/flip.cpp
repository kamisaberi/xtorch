#include "../../include/transforms/flip.h"

namespace xt::data::transforms {


    HorizontalFlip::HorizontalFlip() {
    }

    torch::Tensor HorizontalFlip::operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 2) {
            throw std::runtime_error("Input tensor must have at least 2 dimensions (e.g., [H, W]).");
        }

        // Flip along the last dimension (width)
        return torch::flip(input, {-1});
    }


    VerticalFlip::VerticalFlip() {
    }

    torch::Tensor VerticalFlip::operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 2) {
            throw std::runtime_error("Input tensor must have at least 2 dimensions (e.g., [H, W]).");
        }

        // Flip along the second-to-last dimension (height)
        return torch::flip(input, {-2});
    }





}