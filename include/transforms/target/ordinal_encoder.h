#pragma once

#include "../common.h"



#include <string>
#include <vector>
#include <any>
#include <unordered_map>

namespace xt::transforms::target {

    /**
     * @class OrdinalEncoder
     * @brief A target transformation that converts categorical labels into integer
     *        IDs based on a specified, explicit order.
     *
     * This is used for features where the categories have an intrinsic ranking
     * (e.g., ["low", "medium", "high"]). The encoder maps the first category in
     * the specified list to 0, the second to 1, and so on.
     */
    class OrdinalEncoder : public xt::Module {
    public:
        /**
         * @brief Constructs the OrdinalEncoder.
         *
         * @param ordered_categories A vector of strings containing all possible
         *                           categories in their desired ordinal ranking.
         *                           The first item will be mapped to 0, the second to 1, etc.
         */
        explicit OrdinalEncoder(const std::vector<std::string>& ordered_categories);

        /**
         * @brief Executes the ordinal encoding operation.
         * @param tensors An initializer list expected to contain a single label
         *                as a `std::string`.
         * @return An std::any containing the assigned ordinal integer as a `long`.
         * @throws std::invalid_argument if the label is not found in the initial ordering.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::unordered_map<std::string, long> category_to_index_;
    };

} // namespace xt::transforms::target