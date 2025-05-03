
#include "../../../include/transforms/image/pad.h"

namespace xt::transforms::image {


    /**
     * @brief Constructs a Pad object with the specified padding sizes.
     * @param padding A vector of 64-bit integers defining the padding amounts, in pairs (e.g., {left, right, top, bottom}).
     *
     * Initializes the Pad object by storing the provided padding vector, which will be used to pad
     * input tensors in subsequent calls to the operator() function. The vector must contain an even
     * number of elements, where each pair specifies the left and right padding for a dimension.
     * No validation is performed in this implementation; invalid padding sizes may result in runtime
     * errors when applied.
     */
    Pad::Pad(std::vector<int64_t> padding) : padding(padding) {
    }

    /**
     * @brief Applies padding to the input tensor using the stored padding configuration.
     * @param input The input tensor to be padded, typically in format [N, C, H, W] or [H, W].
     * @return A new tensor with padded dimensions according to the stored padding configuration.
     *
     * This function pads the input tensor using LibTorch’s torch::nn::functional::pad utility with
     * the padding sizes specified during construction. The padding is applied with constant mode
     * (defaulting to zeros) to the last dimensions of the tensor, as determined by the number of
     * pairs in the padding vector. For example, for a 4D tensor [N, C, H, W] with padding {p_left,
     * p_right, p_top, p_bottom}, it pads width (W) and height (H), resulting in [N, C, H + p_top +
     * p_bottom, W + p_left + p_right]. The number of padding values must be even and compatible
     * with the tensor’s dimensions, or a runtime error will occur.
     */
    torch::Tensor Pad::operator()(torch::Tensor input) {
        return torch::nn::functional::pad(input, padding);
    }

}