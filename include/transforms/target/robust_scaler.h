#pragma once

#include "../common.h"


#include <any>

namespace xt::transforms::target {

    /**
     * @class RobustScaler
     * @brief A target transformation that scales numerical data using statistics
     *        that are robust to outliers.
     *
     * This scaler removes the median and scales the data according to a quantile
     * range (usually the Interquartile Range, IQR: 75th quantile - 25th quantile).
     * The formula is:
     *   `scaled_value = (value - median) / (quantile_3 - quantile_1)`
     */
    class RobustScaler : public xt::Module {
    public:
        /**
         * @brief Constructs the RobustScaler.
         *
         * @param median The median of the feature in the training dataset.
         * @param q1 The first quantile (25th percentile) from the training data.
         * @param q3 The third quantile (75th percentile) from the training data.
         */
        explicit RobustScaler(double median, double q1, double q3);

        /**
         * @brief Executes the scaling operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                numerical value (e.g., float, double, int).
         * @return An std::any containing the scaled value as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double median_;
        double interquartile_range_;
    };

} // namespace xt::transforms::target