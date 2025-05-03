#include "../../include/transforms/image/normalize.h"

namespace xt::data::transforms {

    Normalize::Normalize(std::vector<float> mean_, std::vector<float> std_)
            : mean(std::move(mean_)), std(std::move(std_)) {
        if (mean.size() != std.size()) {
            throw std::invalid_argument("Mean and std must have the same number of channels");
        }
    }

    torch::Tensor Normalize::operator()(const torch::Tensor &tensor) const {
        if (tensor.dim() != 3 || tensor.size(0) != static_cast<long>(mean.size())) {
            throw std::invalid_argument("Input tensor must be CHW with matching number of channels");
        }

        torch::Tensor out = tensor.clone();
        for (size_t c = 0; c < mean.size(); ++c) {
            out[c] = (out[c] - mean[c]) / std[c];
        }

        return out;
    }



}