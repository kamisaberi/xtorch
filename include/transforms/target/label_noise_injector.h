#pragma once

#include "../common.h"

#include <vector>
#include <any>
#include <random> // For random number generation

namespace xt::transforms::target {

    /**
     * @class LabelNoiseInjector
     * @brief A target transformation that randomly replaces a label with another
     *        incorrect label to simulate a noisy dataset.
     *
     * This is a regularization technique. With a given probability, the input
     * label is swapped for a different, randomly chosen label from the set of
     * all possible classes.
     */
    class LabelNoiseInjector : public xt::Module {
    public:
        /**
         * @brief Constructs the LabelNoiseInjector.
         *
         * @param num_classes The total number of unique classes. This is needed
         *                    to know the range of possible incorrect labels.
         * @param noise_probability The probability (from 0.0 to 1.0) that any given
         *                          label will be replaced. Defaults to 0.1 (10%).
         */
        explicit LabelNoiseInjector(int num_classes, float noise_probability = 0.1f);

        /**
         * @brief Executes the noise injection operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                integer value (e.g., int, long) representing the correct label.
         * @return An std::any containing a label as a `long`. This will be the
         *         original label most of the time, but sometimes a random incorrect one.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_classes_;
        float noise_prob_;
        std::mt19937 random_engine_;
        std::uniform_real_distribution<float> prob_dist_;
        std::uniform_int_distribution<long> label_dist_;
    };

} // namespace xt::transforms::target