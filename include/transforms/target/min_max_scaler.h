#pragma once

#include "../common.h"


#include <any>

namespace xt::transforms::target {

    /**
     * @class MinMaxScaler
     * @brief A target transformation that scales a numerical label to the [0, 1] range.
     *
     * This scaler is "fitted" by providing the minimum and maximum values of the
     * training data to its constructor. It then applies the formula:
     *   `scaled_value = (value - data_min) / (data_max - data_min)`
     */
    class MinMaxScaler : public xt::Module {
    public:
        /**
         * @brief Constructs the MinMaxScaler.
         *
         * @param data_min The minimum value of the feature/label in the training dataset.
         * @param data_max The maximum value of the feature/label in the training dataset.
         */
        explicit MinMaxScaler(double data_min, double data_max);

        /**
         * @brief Executes the scaling operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                numerical value (e.g., float, double, int).
         * @return An std::any containing the scaled value as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double data_min_;
        double data_range_;
    };

} // namespace xt::transforms::target