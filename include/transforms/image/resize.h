#pragma once


#include "../../headers/transforms.h"

namespace xt::data::transforms {


    /**
     * @struct Resize
     * @brief A functor to resize a tensor image to a specified size.
     *
     * This struct provides a callable object that resizes a `torch::Tensor` representing an image
     * to a target size specified as a vector of 64-bit integers. It uses the call operator to
     * perform the resizing operation, making it suitable for use in functional pipelines or
     * transformations.
     */
    struct Resize {
    public:
        /**
         * @brief Constructs a Resize object with the target size.
         * @param size A vector of 64-bit integers specifying the target dimensions (e.g., {height, width}).
         */
        Resize(std::vector<int64_t> size);

        /**
         * @brief Resizes the input tensor image to the target size.
         * @param img The input tensor image to be resized.
         * @return A new tensor with the resized dimensions.
         */
        torch::Tensor operator()(torch::Tensor img);

    private:
        std::vector<int64_t> size; ///< The target size for resizing (e.g., {height, width}).
    };




}