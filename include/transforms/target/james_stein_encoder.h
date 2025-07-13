#pragma once

#include "../common.h"


#include <string>
#include <vector>
#include <any>
#include <unordered_map>

namespace xt::transforms::target {

    /**
     * @struct CategoryStats
     * @brief Holds the pre-calculated statistics for a single category needed
     *        for James-Stein encoding.
     */
    struct CategoryStats {
        double mean; // The mean of the target variable for this category.
        long count;  // The number of times this category appeared.
    };


    /**
     * @class JamesSteinEncoder
     * @brief A target transformation that replaces a categorical label with a
     *        smoothed (shrinkage) version of its target mean.
     *
     * This advanced encoder regularizes the target encoding by shrinking the
     * mean of rare categories towards the global target mean, preventing
     * overfitting and making the encoding more robust.
     */
    class JamesSteinEncoder : public xt::Module {
    public:
        /**
         * @brief Constructs the JamesSteinEncoder.
         *
         * @param category_stats A map from string labels to their pre-calculated
         *                       CategoryStats (mean and count).
         * @param global_mean The global mean of the target variable across the
         *                    entire training dataset.
         * @param smoothing A tunable parameter `k` that controls the amount of
         *                  shrinkage. Higher values cause more shrinkage.
         *                  Defaults to 10.0.
         */
        explicit JamesSteinEncoder(
            const std::unordered_map<std::string, CategoryStats>& category_stats,
            double global_mean,
            double smoothing = 10.0
        );

        /**
         * @brief Executes the James-Stein encoding operation.
         * @param tensors An initializer list expected to contain a single label.
         *                The label can be a `std::string`, `int`, `long`, etc.
         * @return An std::any containing the smoothed, encoded value as a `double`.
         *         Returns the global mean for labels not seen during fitting.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        const std::unordered_map<std::string, CategoryStats>& category_stats_;
        double global_mean_;
        double smoothing_;
    };

} // namespace xt::transforms::target