#pragma once

#include "../common.h"


#include <torch/torch.h> // Assumes LibTorch is installed
#include <vector>

namespace xt::transforms::target {

    /**
     * @class LabelSmoother
     * @brief A target transformation that converts a class index into a smoothed,
     *        one-hot-like floating point vector.
     *
     * This is a regularization technique that prevents a model from becoming
     * overconfident. Instead of a hard target (e.g., [0, 1, 0]), it uses a
     * soft target (e.g., [0.05, 0.9, 0.05]).
     */
    class LabelSmoother : public xt::Module {
    public:
        /**
         * @brief Constructs the LabelSmoother transform.
         *
         * @param num_classes The total number of classes in the dataset.
         * @param epsilon The smoothing factor (0.0 <= epsilon < 1.0). A larger
         *                value means more smoothing. Defaults to 0.1.
         */
        explicit LabelSmoother(int num_classes, float epsilon = 0.1f);

        /**
         * @brief Executes the label smoothing operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                integer value (e.g., int, long) representing the true class index.
         * @return An std::any containing the smoothed label vector as a 1D
         *         `torch::Tensor` of shape (num_classes,).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_classes_;
        float epsilon_;

        // Pre-calculated values for efficiency
        float value_for_correct_class_;
        float value_for_incorrect_class_;
    };

} // namespace xt::transforms::target