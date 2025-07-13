#pragma once

#include "../common.h"


#include "../common.h"
#include <vector>
#include <any>

namespace xt::transforms::target {

    /**
     * @class EventToIntervalConverter
     * @brief A target transformation that converts a sequence of event timestamps
     *        into a sequence of inter-event intervals (durations).
     *
     * Given a sorted list of N timestamps, this transform produces a list of
     * N-1 durations, where each duration is the time between one event and the next.
     */
    class EventToIntervalConverter : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        EventToIntervalConverter();

        /**
         * @brief Executes the interval calculation.
         * @param tensors An initializer list expected to contain a single item:
         *                a `std::vector<double>` of sorted event timestamps.
         * @return An std::any containing the calculated intervals as a
         *         `std::vector<double>`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::target