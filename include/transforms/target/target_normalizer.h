#pragma once

#include "../common.h"

#include <any>

namespace xt::transforms::target {

    /**
     * @class TargetNormalizer
     * @brief A target transformation that standardizes a numerical label by
     *        removing the mean and scaling to unit variance.
     *
     * This is a very common preprocessing step, also known as z-score normalization.
     * The normalizer is "fitted" by providing the mean and standard deviation of
     * the training data. It then applies the formula:
     *   `normalized_value = (value - mean) / std_dev`
     */
    class TargetNormalizer : public xt::Module {
    public:
        /**
         * @brief Constructs the TargetNormalizer.
         *
         * @param mean The mean (μ) of the target variable in the training dataset.
         * @param std_dev The standard deviation (σ) of the target variable in the
         *                training dataset.
         */
        explicit TargetNormalizer(double mean, double std_dev);

        /**
         * @brief Executes the normalization operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                numerical value (e.g., float, double, int).
         * @return An std::any containing the normalized value as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double mean_;
        double std_dev_;
    };

} // namespace xt::transforms::target