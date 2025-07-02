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
        // if (tensors.size() != 1) throw std::runtime_error("Normalize expects one input");
        // torch::Tensor tensor = std::any_cast<torch::Tensor>(*tensors.begin());
        // // Basic normalization, make sure types and shapes are compatible
        // tensor = tensor.to(torch::kFloat32);
        // auto mean = mean.to(tensor.device()).to(tensor.scalar_type());
        // auto std = std.to(tensor.device()).to(tensor.scalar_type());
        // // Ensure mean/std are broadcastable, e.g., reshape for image channels
        // if (tensor.dim() == 3 && mean.numel() == tensor.size(0)) { // CHW
        //     mean = mean.reshape({-1, 1, 1});
        //     std = std.reshape({-1, 1, 1});
        // } else if (tensor.dim() == 4 && mean.numel() == tensor.size(1)) { // NCHW
        //     mean = mean.reshape({1, -1, 1, 1});
        //     std = std.reshape({1, -1, 1, 1});
        // }
        // return (tensor - mean) / std;
    }
}
