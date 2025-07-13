#pragma once

#include "../common.h"



namespace xt::transforms::target {

    /**
     * @class BinaryEncoder
     * @brief A target transformation that converts a class index into its dense
     *        binary representation.
     *
     * For a fixed number of bits `N`, this transform takes a scalar label `i`
     * and converts it into a 1D tensor of size `N` representing the binary
     * digits of `i`. For example, with N=8, the label 42 would become the
     * tensor `[0, 0, 1, 0, 1, 0, 1, 0]`.
     */
    class BinaryEncoder : public xt::Module {
    public:
        /**
         * @brief Constructs the BinaryEncoder transform.
         *
         * @param num_bits The fixed number of bits for the output vector. This
         *                 determines the length of the output tensor and the
         *                 maximum class index that can be represented (2^num_bits - 1).
         */
        explicit BinaryEncoder(int num_bits);

        /**
         * @brief Executes the binary encoding operation.
         * @param tensors An initializer list expected to contain a single scalar
         *                value (e.g., int, long) representing the class index.
         * @return An std::any containing the binary encoded torch::Tensor of
         *         shape (num_bits,).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_bits_;
    };

} // namespace xt::transforms::target