#pragma once

#include "common.h"

namespace xt::dropouts
{
    torch::Tensor curriculum_dropout(torch::Tensor x);

    struct CurriculumDropout : xt::Module
    {
    public:
        CurriculumDropout(
            double initial_dropout_rate = 0.5,
            double final_dropout_rate = 0.1,
            int64_t total_curriculum_steps = 10000, // e.g., total training steps or epochs for curriculum
            bool increase_rate = false // Default: decrease rate from initial to final
        );
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double initial_dropout_rate_;
        double final_dropout_rate_;
        int64_t total_curriculum_steps_; // Number of steps/epochs over which the rate changes
        bool increase_rate_; // If true, rate goes from initial to final. If false, from final to initial.

        // A transient member to store the most recently calculated dropout rate for pretty_print or inspection
        // Not a registered parameter or buffer.
        mutable double current_p_ = 0.0;
    };
}
