#pragma once

#include "../common.h"
#include <initializer_list>
#include <any>

namespace xt::transforms::target {

    /**
     * @class Clipper
     * @brief A target transformation that clamps or "clips" a numerical label
     *        to be within a specified range [min, max].
     *
     * If the input label is less than the minimum value, it is set to the minimum.
     * If it is greater than the maximum value, it is set to the maximum. Otherwise,
     * it remains unchanged.
     */
    class Clipper : public xt::Module {
    public:
        /**
         * @brief Constructs the Clipper transform.
         *
         * @param min_val The minimum allowed value (inclusive).
         * @param max_val The maximum allowed value (inclusive).
         */
        explicit Clipper(long min_val, long max_val);

        /**
         * @brief Executes the clipping operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                integer value (e.g., int, long) representing the label.
         * @return An std::any containing the clipped label as a `long`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        long min_val_;
        long max_val_;
    };

} // namespace xt::transforms::target