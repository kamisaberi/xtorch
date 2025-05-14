#pragma once
#include "transforms/common.h"

namespace xt::transforms::image {


    struct Rotation {
    public:
        /**
         * @brief Constructs a Resize object with the target size.
         * @param size A vector of 64-bit integers specifying the target dimensions (e.g., {height, width}).
         */
        Rotation(double angle_deg);

        /**
         * @brief Resizes the input tensor image to the target size.
         * @param img The input tensor image to be resized.
         * @return A new tensor with the resized dimensions.
         */
        torch::Tensor operator()(const torch::Tensor &input_tensor);

    private:
        double angle;
    };




}