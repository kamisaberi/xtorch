#pragma once

#include "../common.h"

#include <any>

namespace xt::transforms::target {

    /**
     * @brief The type of power transformation to apply.
     */
    enum class PowerTransformType {
        YEO_JOHNSON,
        BOX_COX
    };


    /**
     * @class PowerTransformer
     * @brief A target transformation that applies a power function to make the
     *        data more Gaussian-like (normal).
     *
     * This class can perform both Yeo-Johnson and Box-Cox transformations. It must
     * be "fitted" by providing the lambda (λ) parameter, which is typically
     * found via maximum likelihood estimation on the training data.
     */
    class PowerTransformer : public xt::Module {
    public:
        /**
         * @brief Constructs the PowerTransformer.
         *
         * @param lambda The power parameter λ, pre-calculated from the training data.
         * @param type The type of transformation to apply (Yeo-Johnson or Box-Cox).
         *             Defaults to YEO_JOHNSON as it's more general.
         */
        explicit PowerTransformer(
                double lambda,
                PowerTransformType type = PowerTransformType::YEO_JOHNSON
        );

        /**
         * @brief Executes the power transformation.
         * @param tensors An initializer list expected to contain a single scalar
         *                numerical value (e.g., float, double, int).
         * @return An std::any containing the transformed value as a `double`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double lambda_;
        PowerTransformType type_;
    };

} // namespace xt::transforms::target