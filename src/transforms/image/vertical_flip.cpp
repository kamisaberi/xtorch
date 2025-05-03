#include "../../../include/transforms/image/vertical_flip.h"

namespace xt::transforms::image {



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