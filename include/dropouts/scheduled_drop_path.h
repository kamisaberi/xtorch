#pragma once

#include "common.h"

namespace xt::dropouts
{

    struct ScheduledDropPath : xt::Module
    {
    public:
        ScheduledDropPath(
            double final_drop_rate = 0.2, // The target drop rate at the end of the schedule
            int64_t total_curriculum_steps = 100000, // Total steps for the schedule
            double initial_drop_rate = 0.0 // Starting drop rate (often 0 for DropPath)
        );
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double initial_drop_rate_;
        double final_drop_rate_;
        int64_t total_curriculum_steps_; // Number of steps/epochs over which the rate changes
        bool increase_rate_; // If true, rate goes from initial to final.
        // Common for DropPath: start low (or 0), increase to final.
        double epsilon_ = 1e-7; // For numerical stability

        // Transient member to store the most recently calculated drop rate
        mutable double current_p_drop_ = 0.0;
    };
}
