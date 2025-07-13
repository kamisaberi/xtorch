#pragma once

#include "../common.h"



#include <string>
#include <vector>
#include <any>
#include <unordered_map>

namespace xt::transforms::target {

    /**
     * @class FrequencyEncoder
     * @brief A target transformation that replaces a categorical label with its
     *        pre-calculated frequency.
     *
     * This is a powerful encoding technique for categorical data, where each
     * category is mapped to its frequency (percentage of occurrence) in the
     * training dataset.
     */
    class FrequencyEncoder : public xt::Module {
    public:
        /**
         * @brief Constructs the FrequencyEncoder.
         *
         * @param frequency_map A map where keys are the string representations
         *                      of labels and values are their pre-calculated
         *                      frequencies (e.g., from 0.0 to 1.0).
         */
        explicit FrequencyEncoder(const std::unordered_map<std::string, float>& frequency_map);

        /**
         * @brief Executes the frequency lookup operation.
         * @param tensors An initializer list expected to contain a single label.
         *                The label can be a `std::string`, `int`, `long`, etc.
         *                If it's a number, it will be converted to a string for lookup.
         * @return An std::any containing the frequency of the label as a `double`.
         *         Returns 0.0 for labels not found in the frequency map.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::unordered_map<std::string, float> frequency_map_;
    };

} // namespace xt::transforms::target