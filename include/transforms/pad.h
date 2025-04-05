#pragma once


#include "../headers/transforms.h"

namespace xt::data::transforms {

    /**
     * @struct Pad
     * @brief A functor to pad a tensor with a specified padding configuration.
     *
     * This struct provides a callable object that applies padding to a `torch::Tensor` based on a
     * vector of padding sizes. It is designed to extend the dimensions of a tensor (e.g., images
     * in machine learning workflows) by adding values (typically zeros) around its boundaries,
     * using the padding amounts specified during construction. The padding is applied using
     * LibTorch's functional padding utilities.
     */
    struct Pad {
    public:
        /**
         * @brief Constructs a Pad object with the specified padding sizes.
         * @param padding A vector of 64-bit integers defining the padding amounts, in pairs (e.g., {left, right, top, bottom}).
         *
         * Initializes the Pad object with a vector specifying the padding to be applied to the tensor’s
         * dimensions. The vector must contain an even number of elements, where each pair corresponds
         * to the left and right padding for a dimension, applied to the tensor’s last dimensions in
         * reverse order (e.g., width, then height for a 2D tensor).
         */
        Pad(std::vector<int64_t> padding);

        /**
         * @brief Applies padding to the input tensor.
         * @param input The input tensor to be padded, typically in format [N, C, H, W] or [H, W].
         * @return A new tensor with padded dimensions according to the stored padding configuration.
         *
         * This operator pads the input tensor using the padding sizes provided at construction,
         * typically with zeros using constant mode padding. For a 4D tensor [N, C, H, W] and padding
         * {p_left, p_right, p_top, p_bottom}, it pads width (W) and height (H), resulting in
         * [N, C, H + p_top + p_bottom, W + p_left + p_right]. The padding is applied to the last
         * dimensions corresponding to the number of pairs in the padding vector.
         */
        torch::Tensor operator()(torch::Tensor input);

    private:
        /**
         * @brief Vector storing the padding sizes.
         *
         * This member holds the padding configuration, where each pair of values specifies the
         * left and right padding for a dimension of the tensor.
         */
        std::vector<int64_t> padding;
    };



}