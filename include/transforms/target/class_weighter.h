#pragma once

#include "../common.h"


namespace xt::transforms::target {

    /**
     * @class ClassWeighter
     * @brief A target transformation that assigns a pre-defined weight to a class label.
     *
     * This is commonly used in training on imbalanced datasets. It takes a
     * class index and returns a corresponding scalar weight, which can then be
     * used to scale the loss for that specific sample.
     */
    class ClassWeighter : public xt::Module {
    public:
        /**
         * @brief Constructs the ClassWeighter transform.
         *
         * @param class_weights A vector of floats where `class_weights[i]` is the
         *                      weight to be assigned to class `i`. The size of this
         *                      vector determines the number of classes handled.
         */
        explicit ClassWeighter(const std::vector<float>& class_weights);

        /**
         * @brief Executes the weight lookup operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                value (e.g., int, long) representing the class index.
         * @return An std::any containing a scalar torch::Tensor (0-dimensional)
         *         with the corresponding class weight.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<float> class_weights_;
        int num_classes_;
    };

} // namespace xt::transforms::target