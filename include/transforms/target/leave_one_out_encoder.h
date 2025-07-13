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
     *        for leave-one-out encoding.
     */
    struct TargetStats {
        double sum;  // The total sum of the target variable for this category.
        long count;  // The number of times this category appeared.
    };


    /**
     * @class LeaveOneOutEncoder
     * @brief A target transformation that replaces a categorical label with the
     *        mean of the target variable for that category, calculated by
     *        "leaving out" the current sample's own target value.
     *
     * This is a powerful regularization technique for target encoding that
     * prevents data leakage from the sample's own label.
     */
    class LeaveOneOutEncoder : public xt::Module {
    public:
        /**
         * @brief Constructs the LeaveOneOutEncoder.
         *
         * @param target_stats A map from string labels to their pre-calculated
         *                     TargetStats (sum and count) from the training data.
         * @param global_mean The global mean of the target variable across the
         *                    entire training dataset. This is used as a fallback.
         */
        explicit LeaveOneOutEncoder(
            const std::unordered_map<std::string, TargetStats>& target_stats,
            double global_mean
        );

        /**
         * @brief Executes the leave-one-out encoding operation.
         * @param tensors An initializer list expected to contain exactly two items:
         *                1. The categorical label (e.g., std::string, int).
         *                2. The target value for this specific sample (e.g., double, float).
         * @return An std::any containing the calculated leave-one-out mean as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        const std::unordered_map<std::string, TargetStats>& target_stats_;
        double global_mean_;
    };

} // namespace xt::transforms::target