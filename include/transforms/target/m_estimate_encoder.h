#pragma once

#include "../common.h"

#include <string>
#include <vector>
#include <any>
#include <unordered_map>

namespace xt::transforms::target {

    /**
     * @struct TargetStats
     * @brief Holds the pre-calculated statistics for a single category needed
     *        for M-Estimate encoding.
     */
    struct TargetStatsM {
        double mean; // The mean of the target variable for this category.
        long count;  // The number of times this category appeared.
    };


    /**
     * @class MEstimateEncoder
     * @brief A target transformation that replaces a categorical label with a
     *        smoothed version of its target mean using M-Estimate smoothing.
     *
     * This is a robust target encoding method that shrinks the estimate for rare
     * categories towards the global mean, preventing overfitting.
     */
    class MEstimateEncoder : public xt::Module {
    public:
        /**
         * @brief Constructs the MEstimateEncoder.
         *
         * @param category_stats A map from string labels to their pre-calculated
         *                       TargetStats (mean and count) from the training data.
         * @param global_mean The global mean of the target variable across the
         *                    entire training dataset.
         * @param m The smoothing factor `m`. A higher value applies more
         *          smoothing and shrinks rare categories more strongly towards
         *          the global mean. Defaults to 1.0.
         */
        explicit MEstimateEncoder(
            const std::unordered_map<std::string, TargetStatsM>& category_stats,
            double global_mean,
            double m = 1.0
        );

        /**
         * @brief Executes the M-Estimate encoding operation.
         * @param tensors An initializer list expected to contain a single label.
         *                The label can be a `std::string`, `int`, `long`, etc.
         * @return An std::any containing the smoothed, encoded value as a `double`.
         *         Returns the global mean for labels not seen during fitting.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        const std::unordered_map<std::string, TargetStatsM>& category_stats_;
        double global_mean_;
        double m_;
    };

} // namespace xt::transforms::target