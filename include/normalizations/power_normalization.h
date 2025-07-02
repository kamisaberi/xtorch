#pragma once

#include "common.h"

namespace xt::norm
{
    struct PowerNorm : xt::Module
    {
    public:
        PowerNorm(double initial_power = 0.5, // e.g., 0.5 for sqrt normalization
                  bool learnable_power = false,
                  bool apply_l2_norm = false,
                  int64_t l2_norm_dim = -1, // -1 for global L2 norm
                  bool signed_power = true,
                  double eps_l2 = 1e-8);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double power_val_; // The exponent 'p'
        bool learnable_power_; // Whether 'p' is learnable
        bool apply_l2_norm_; // Whether to apply L2 norm before power transform
        bool signed_power_; // If true, use sgn(x) * |x|^p, else x^p (risks NaN for x<0, non-integer p)
        double eps_l2_; // Epsilon for L2 normalization (if applied)
        int64_t l2_norm_dim_; // Dimension for L2 norm (-1 for global, or specific axis)

        // Learnable parameter for power (if learnable_power_ is true)
        torch::Tensor power_param_;
    };
}
