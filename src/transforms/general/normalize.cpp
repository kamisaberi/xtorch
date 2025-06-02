#include "include/transforms/general/normalize.h"

namespace xt::transforms::general
{
    Normalize::Normalize(std::vector<float> mean_, std::vector<float> std_)
        : mean(std::move(mean_)), std(std::move(std_))
    {
        if (mean.size() != std.size())
        {
            throw std::invalid_argument("Mean and std must have the same number of channels");
        }
    }

    auto Normalize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        const torch::Tensor input = tensor_vec[0];

        if (input.dim() != 3 || input.size(0) != static_cast<long>(mean.size()))
        {
            throw std::invalid_argument("Input tensor must be CHW with matching number of channels");
        }

        torch::Tensor out = input.clone();
        for (size_t c = 0; c < mean.size(); ++c)
        {
            out[c] = (out[c] - mean[c]) / std[c];
        }

        return out;
    }
}
