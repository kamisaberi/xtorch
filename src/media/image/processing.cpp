#include "../../../include/media/image/processing.h"

namespace temi = torch::ext::media::image;


torch::Tensor temi::resize(const torch::Tensor &tensor, const std::vector<int64_t> &size) {
    return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(
                    torch::kBilinear).align_corners(false)
    ).squeeze(0);
}

