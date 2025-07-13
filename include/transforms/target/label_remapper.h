#pragma once

#include "../common.h"

#include <vector>
#include <any>
#include <unordered_map>

namespace xt::transforms::target {

    /**
     * @class LabelRemapper
     * @brief A target transformation that changes specific class labels to new
     *        values based on a provided mapping.
     *
     * This transform is useful for merging classes or correcting label schemes.
     * If an input label is a key in the remapping map, its value is replaced
     * with the corresponding value from the map. If the label is not in the map,
     * it is passed through unchanged.
     */
    class LabelRemapper : public xt::Module {
    public:
        /**
         * @brief Constructs the LabelRemapper.
         *
         * @param remapping_map An unordered map where the key is the original
         *                      label ID and the value is the new label ID.
         */
        explicit LabelRemapper(const std::unordered_map<long, long>& remapping_map);

        /**
         * @brief Executes the remapping operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                integer value (e.g., int, long) representing the label.
         * @return An std::any containing the remapped label as a `long`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::unordered_map<long, long> remapping_map_;
    };

} // namespace xt::transforms::target