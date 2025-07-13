#pragma once

#include "../common.h"


#include <vector>
#include <any>

namespace xt::transforms::target {

    /**
     * @class Discretizer
     * @brief A target transformation that converts a continuous numerical label
     *        into a discrete class index based on a set of bin boundaries.
     *
     * Given a set of `k` boundaries `{b_1, b_2, ..., b_k}`, this transform
     * creates `k+1` bins. A value `v` is placed into bin `i` according to:
     * - bin 0: if v < b_1
     * - bin i: if b_i <= v < b_{i+1}
     * - bin k: if v >= b_k
     */
    class Discretizer : public xt::Module {
    public:
        /**
         * @brief Constructs the Discretizer transform.
         *
         * @param bin_boundaries A sorted vector of floats representing the
         *                       boundaries of the bins. For `k` boundaries, `k+1`
         *                       bins will be created.
         */
        explicit Discretizer(const std::vector<float>& bin_boundaries);

        /**
         * @brief Executes the discretization operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                numerical value (e.g., float, double, int).
         * @return An std::any containing the resulting class index as a `long`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<float> bin_boundaries_;
    };

} // namespace xt::transforms::target