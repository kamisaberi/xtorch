#pragma once

#include "../common.h"

#include <any>

namespace xt::transforms::target {

    /**
     * @class LogTransformer
     * @brief A target transformation that applies a natural logarithm (log(1+x))
     *        to a numerical label.
     *
     * This is a variance-stabilizing transformation, often used to handle
     * right-skewed data (e.g., prices, counts) and make its distribution
     * more normal. The log(1+x) form is used to gracefully handle inputs of 0.
     */
    class LogTransformer : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        LogTransformer();

        /**
         * @brief Executes the log transformation.
         * @param tensors An initializer list expected to contain a single scalar
         *                numerical value (e.g., float, double, int).
         * @return An std::any containing the log-transformed value as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::target