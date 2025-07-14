#pragma once

#include "../common.h"


#include <any>
#include <vector>

namespace xt::transforms::target {

    /**
     * @brief The target probability distribution for the output data.
     */
    enum class QuantileOutputType {
        UNIFORM,
        NORMAL
    };


    /**
     * @class QuantileTransformer
     * @brief A target transformation that maps numerical data to a specified
     *        probability distribution using quantile information.
     *
     * This is a non-linear transform that is robust to outliers. It is "fitted"
     * by providing the learned quantiles of the training data. It then maps
     * new data points to their rank and uses the inverse CDF of the target
     * distribution (Uniform or Normal) to produce the output.
     */
    class QuantileTransformer : public xt::Module {
    public:
        /**
         * @brief Constructs the QuantileTransformer.
         *
         * @param learned_quantiles A sorted vector of floats representing the
         *                          empirical quantiles learned from the training data.
         * @param output_type The desired output distribution. Defaults to UNIFORM.
         */
        explicit QuantileTransformer(
                const std::vector<double>& learned_quantiles,
                QuantileOutputType output_type = QuantileOutputType::UNIFORM
        );

        /**
         * @brief Executes the quantile transformation.
         * @param tensors An initializer list expected to contain a single scalar
         *                numerical value (e.g., float, double, int).
         * @return An std::any containing the transformed value as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<double> learned_quantiles_;
        QuantileOutputType output_type_;

        /**
         * @brief Calculates the value of the inverse normal CDF (probit function).
         * This is an approximation. For higher precision, a dedicated stats
         * library would be needed.
         */
        static double inverse_normal_cdf(double p);
    };

} // namespace xt::transforms::target