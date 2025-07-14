#pragma once

#include "../common.h"


#include <string>
#include <vector>
#include <any>
#include <unordered_map>

namespace xt::transforms::target {

    /**
     * @class TargetEncoder
     * @brief A target transformation that replaces a categorical label with the
     *        mean of the target variable for that category.
     *
     * This is a powerful encoding method that directly incorporates information
     * from the target variable. It must be "fitted" on a training set to
     * calculate the means for each category.
     */
    class TargetEncoder : public xt::Module {
    public:
        /**
         * @brief Constructs the TargetEncoder.
         *
         * @param category_means A map from string labels to the pre-calculated
         *                       mean of the target variable for that category.
         * @param global_mean The global mean of the target variable across the
         *                    entire training dataset. This is used as a fallback
         *                    for unseen categories.
         */
        explicit TargetEncoder(
                const std::unordered_map<std::string, double>& category_means,
                double global_mean
        );

        /**
         * @brief Executes the target encoding operation.
         * @param tensors An initializer list expected to contain a single label.
         *                The label can be a `std::string`, `int`, `long`, etc.
         * @return An std::any containing the corresponding target mean as a `double`.
         *         Returns the global mean for labels not seen during fitting.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        const std::unordered_map<std::string, double>& category_means_;
        double global_mean_;
    };

} // namespace xt::transforms::target