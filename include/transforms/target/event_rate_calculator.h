#pragma once

#include "../common.h"



#include "../common.h"
#include <vector>
#include <any>

namespace xt::transforms::target {

    /**
     * @class EventRateCalculator
     * @brief A target transformation that calculates the rate of events over a
     *        given duration.
     *
     * This transform takes a list of event timestamps and a total observation
     * duration and computes a single feature: the event rate. The rate can be
     * scaled to a desired time unit (e.g., events per second, per minute).
     */
    class EventRateCalculator : public xt::Module {
    public:
        /**
         * @brief Constructs the EventRateCalculator.
         *
         * @param per_time_unit A scaling factor for the output. If timestamps and
         *                      duration are in seconds, a `per_time_unit` of 1.0
         *                      gives rate in events/sec, while 60.0 gives
         *                      events/minute. Defaults to 1.0.
         */
        explicit EventRateCalculator(double per_time_unit = 1.0);

        /**
         * @brief Executes the rate calculation.
         * @param tensors An initializer list expected to contain exactly two items:
         *                1. A `std::vector<double>` of event timestamps.
         *                2. A `double` for the total observation duration.
         * @return An std::any containing the calculated rate as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double per_time_unit_;
    };

} // namespace xt::transforms::target