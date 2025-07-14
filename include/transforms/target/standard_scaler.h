#pragma once

#include "../common.h"


#include <any>

namespace xt::transforms::target {

    /**
     * @class StandardScaler
     * @brief A target transformation that standardizes a numerical label by
     *        removing the mean and scaling to unit variance.
     *
     * This is a very common preprocessing step. The scaler is "fitted" by
     * providing the mean and standard deviation of the training data. It then
     * applies the formula:
     *   `z = (x - mean) / std_dev`
     */
    class StandardScaler : public xt::Module {
    public:
        /**
         * @brief Constructs the StandardScaler.
         *
         * @param mean The mean (μ) of the feature in the training dataset.
         * @param std_dev The standard deviation (σ) of the feature in the training dataset.
         */
        explicit StandardScaler(double mean, double std_dev);

        /**
         * @brief Executes the standardization operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                numerical value (e.g., float, double, int).
         * @return An std::any containing the standardized value as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double mean_;
        double std_dev_;
    };

} // namespace xt::transforms::target