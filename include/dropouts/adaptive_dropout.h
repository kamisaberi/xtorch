#pragma once

#include "common.h"
#include <torch/torch.h>
#include <vector>
#include <cmath>     // For std::log
#include <ostream>   // For std::ostream


namespace
{
    // Anonymous namespace for helper utility
    double calculate_initial_log_alpha_value(double initial_dropout_rate)
    {
        // Clamp initial_dropout_rate to avoid log(0) or log( división by zero )
        double epsilon = 1e-7;
        if (initial_dropout_rate < epsilon)
        {
            initial_dropout_rate = epsilon;
        }
        if (initial_dropout_rate > 1.0 - epsilon)
        {
            initial_dropout_rate = 1.0 - epsilon;
        }
        return std::log(initial_dropout_rate / (1.0 - initial_dropout_rate));
    }
} // namespace


struct AdaptiveDropoutImpl : torch::nn::Module {
    torch::Tensor log_alpha_;

    AdaptiveDropoutImpl(c10::IntArrayRef probability_shape = {}, double initial_dropout_rate = 0.05) {
        double initial_log_alpha_val = calculate_initial_log_alpha_value(initial_dropout_rate);

        torch::Tensor log_alpha_init;
        if (probability_shape.empty()) { // Scalar probability
            log_alpha_init = torch::tensor(initial_log_alpha_val, torch::kFloat32);
        } else {
            log_alpha_init = torch::full(probability_shape, initial_log_alpha_val, torch::kFloat32);
        }
        log_alpha_ = register_parameter("log_alpha", log_alpha_init);
    }

    torch::Tensor forward(const torch::Tensor& input) {
        if (!this->is_training()) {
            return input;
        }

        torch::Tensor p = torch::sigmoid(log_alpha_);

        // Clamp p to prevent keep_prob from being too close to 0 or 1, ensuring numerical stability.
        // sigmoid output is (0,1), so p is already >0 and <1 if log_alpha_ is finite.
        // Clamping mainly for extreme log_alpha_ values or if desired for stricter bounds.
        // Let's ensure keep_prob is not zero for division.
        p = torch::clamp(p, 0.0, 1.0 - 1e-7);

        torch::Tensor keep_prob = 1.0 - p;
        torch::Tensor kp_broadcastable = keep_prob;

        // Heuristic: if keep_prob is 1D (e.g. shape [C]) and input is multi-dimensional (e.g. [N, C, H, W]),
        // and its size matches input's dim 1 (channel dimension), reshape for broadcasting.
        if (keep_prob.dim() == 1 && input.dim() > 1 && keep_prob.size(0) == input.size(1)) {
            std::vector<int64_t> view_shape(input.dim(), 1L); // Create shape like [1, 1, ..., 1]
            view_shape[1] = keep_prob.size(0); // Set channel dimension, e.g., [1, C, 1, 1]
            kp_broadcastable = keep_prob.view(view_shape);
        }
        // If shapes are otherwise, rely on standard PyTorch broadcasting rules.
        // If incompatible, PyTorch will raise an error.

        torch::Tensor random_tensor = torch::rand_like(input);
        torch::Tensor mask = (random_tensor < kp_broadcastable).to(input.dtype());

        // Scale the output by 1/keep_prob.
        // kp_broadcastable is guaranteed to be >= 1e-7 due to p clamping.
        torch::Tensor output = (input * mask) / kp_broadcastable;

        return output;
    }




    void pretty_print(std::ostream& stream) const override {
        stream << "AdaptiveDropout(probability_shape=" << log_alpha_.sizes() << ")";
    }
};

TORCH_MODULE(AdaptiveDropout); // Creates the AdaptiveDropout module "class"


namespace xt::dropouts
{
    torch::Tensor adaptive_dropout(torch::Tensor x);

    struct AdaptiveDropout : xt::Module
    {
    public:
        AdaptiveDropout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
