#pragma once

#include "../common.h"


namespace xt::transforms::target {

    /**
     * @class Binarizer
     * @brief A target transformation that converts a class index into a one-hot
     *        encoded binary tensor.
     *
     * For a given number of classes N, this transform takes a scalar label `i`
     * and converts it into a 1D tensor of size N which is all zeros except for
     * a 1 at index `i`. This is the standard format for labels in multi-class
     * classification problems.
     */
    class Binarizer : public xt::Module {
    public:
        /**
         * @brief Constructs the Binarizer transform.
         *
         * @param num_classes The total number of classes in the dataset. The
         *                    output tensor will have this length.
         */
        explicit Binarizer(int num_classes);

        /**
         * @brief Executes the one-hot encoding operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                value (e.g., int, long) representing the class index.
         * @return An std::any containing the one-hot encoded torch::Tensor of
         *         shape (num_classes,).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_classes_;
    };

} // namespace xt::transforms::target