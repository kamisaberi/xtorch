#pragma once

#include "common.h"

namespace xt::dropouts
{
    struct SensorDropout : xt::Module
    {
    public:
        SensorDropout(const std::vector<int64_t>& sensor_splits, double p_drop_sensor = 0.1);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_drop_sensor_; // Probability of dropping any given sensor
        std::vector<int64_t> sensor_splits_; // Number of features for each sensor, e.g., {10, 5, 15} for 3 sensors
        int64_t total_features_;
        double epsilon_ = 1e-7;
    };
}
