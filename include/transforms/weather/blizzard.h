#pragma once

#include "../common.h"



namespace xt::transforms::weather {

    /**
     * @class Blizzard
     * @brief A weather transformation to identify blizzard conditions based on thresholds.
     *
     * A blizzard is defined by sustained strong winds and intense snowfall, causing
     * very low visibility. This transform takes tensors for wind speed and snowfall rate
     * and produces a binary tensor identifying the locations/times where blizzard
     * conditions are met.
     *
     * The conditions are checked element-wise:
     * `is_blizzard = (wind_speed > wind_threshold) AND (snowfall > snow_threshold)`
     */
    class Blizzard : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Uses common meteorological thresholds for a blizzard:
         * - Wind speed > 56 km/h (approx. 35 mph)
         * - Snowfall rate > 0.5 cm/hr (indicative of moderate to heavy snow)
         */
        Blizzard();

        /**
         * @brief Constructs the Blizzard transform with custom thresholds.
         * @param wind_speed_threshold The minimum wind speed to be considered for a blizzard.
         * @param snowfall_rate_threshold The minimum snowfall rate to be considered.
         */
        explicit Blizzard(float wind_speed_threshold, float snowfall_rate_threshold);

        /**
         * @brief Executes the blizzard detection operation.
         * @param tensors An initializer list expected to contain two tensors of the same shape:
         *                1. Wind Speed Tensor
         *                2. Snowfall Rate Tensor
         * @return An std::any containing the resulting torch::Tensor of the same shape, with
         *         a value of 1 where blizzard conditions are met and 0 otherwise. The output
         *         tensor will have the same data type as the input tensors.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float wind_speed_threshold_;
        float snowfall_rate_threshold_;
    };

} // namespace xt::transforms::weather