#include "include/transforms/image/resize.h"

namespace xt::transforms::image {

    Resize::Resize(std::vector<int64_t> size) : size(size) {
    }
    torch::Tensor Resize::operator()(torch::Tensor img) {
        img = img.unsqueeze(0); // Add batch dimension
        img = torch::nn::functional::interpolate(
            img,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({size[0], size[1]}))
            .mode(torch::kBilinear)
            .align_corners(false)
        );
        return img.squeeze(0); // Remove batch dimension
    }

}