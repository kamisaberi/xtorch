#pragma once

#include "../common.h"


namespace xt::transforms::weather {

    /**
     * @class AccumulatedSnow
     * @brief A weather transformation that calculates the cumulative snow over a dimension (usually time).
     *
     * This transform takes a tensor representing snowfall at discrete intervals (e.g., hourly or daily)
     * and calculates the accumulated total at each step. It is essentially a wrapper around
     * torch::cumsum for a specific domain.
     */
    class AccumulatedSnow : public xt::Module {
    public:
        /**
         * @brief Default constructor. Accumulates along dimension 0 by default.
         */
        AccumulatedSnow();

        /**
         * @brief Constructs the AccumulatedSnow transform.
         * @param dim The dimension along which to accumulate the snowfall.
         *            For a 1D tensor of daily snowfall, this would be 0.
         *            For a 3D weather grid (time, lat, lon), this would likely be 0.
         */
        explicit AccumulatedSnow(int dim);

        /**
         * @brief Executes the snow accumulation operation.
         * @param tensors An initializer list expected to contain a single tensor
         *                representing snowfall rates or amounts.
         * @return An std::any containing the resulting torch::Tensor with the same
         *         shape and type, where each element is the cumulative sum up to that point.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int dim_; // The dimension to accumulate along.
    };

} // namespace xt::transforms::weather