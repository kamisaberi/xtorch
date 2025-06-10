#include "include/activations/evonorm_s0.h"

namespace xt::activations
{
    torch::Tensor evonorm_s0(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta,
                             const torch::Tensor& v_param, int64_t num_groups,
                             double eps)
    {
        TORCH_CHECK(x.dim() == 4, "Input tensor x must be 4D (N, C, H, W)");
        int64_t num_channels = x.size(1);
        TORCH_CHECK(num_channels > 0, "Number of channels C must be greater than 0");
        TORCH_CHECK(num_groups > 0, "Number of groups must be greater than 0");
        TORCH_CHECK(num_channels % num_groups == 0, "Number of channels must be divisible by num_groups");
        TORCH_CHECK(gamma.numel() == num_channels, "gamma must have C elements");
        TORCH_CHECK(beta.numel() == num_channels, "beta must have C elements");
        TORCH_CHECK(v_param.numel() == num_channels, "v_param must have C elements");

        auto gamma_r = gamma.view({1, num_channels, 1, 1});
        auto beta_r = beta.view({1, num_channels, 1, 1});
        auto v_param_r = v_param.view({1, num_channels, 1, 1});

        torch::Tensor numerator_term = x * torch::sigmoid(v_param_r * x);

        int64_t N = x.size(0);
        int64_t C = x.size(1);
        int64_t H = x.size(2);
        int64_t W = x.size(3);
        int64_t num_channels_per_group = C / num_groups;

        torch::Tensor x_reshaped_for_stats = x.view({N, num_groups, num_channels_per_group, H, W});

        std::vector<int64_t> dims_to_reduce = {2, 3, 4}; // Dims for (num_channels_per_group, H, W)
        torch::Tensor mean_g = torch::mean(x_reshaped_for_stats, dims_to_reduce, /*keepdim=*/true);
        torch::Tensor variance_g = torch::mean(torch::pow(x_reshaped_for_stats - mean_g, 2), dims_to_reduce,
                                               /*keepdim=*/true);

        torch::Tensor denominator_n_x = torch::sqrt(variance_g + eps);

        torch::Tensor numerator_reshaped = numerator_term.view({N, num_groups, num_channels_per_group, H, W});

        torch::Tensor y_normalized_reshaped = numerator_reshaped / denominator_n_x;
        torch::Tensor y_normalized = y_normalized_reshaped.view_as(x);

        return y_normalized * gamma_r + beta_r;
    }


    auto EvonormS0::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::evonorm_s0(torch::zeros(10), torch::zeros(10), torch::zeros(10) < torch::zeros(10), 0);
    }
}
