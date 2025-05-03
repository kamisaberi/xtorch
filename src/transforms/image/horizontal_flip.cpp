#include "../../../include/transforms/image/horizontal_flip.h"

namespace xt::transforms::image
{
    HorizontalFlip::HorizontalFlip()
    {
    }

    torch::Tensor HorizontalFlip::operator()(torch::Tensor input)
    {
        int64_t input_dims = input.dim();
        if (input_dims < 2)
        {
            throw std::runtime_error("Input tensor must have at least 2 dimensions (e.g., [H, W]).");
        }

        // Flip along the last dimension (width)
        return torch::flip(input, {-1});
    }
}
