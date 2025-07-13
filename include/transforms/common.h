#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include <torch/data/transforms/base.h>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>
#include "../base/base.h"
#include  "../utils/utils.h"
#include <vector>
#include <any>
#include "../math/math.h"



// namespace xt::transforms::text
// {
//     /**
//     * @brief Specifies the direction for padding.
//     */
//     enum class PaddingDirection {
//         RIGHT, // Add padding to the end of the sequence.
//         LEFT   // Add padding to the beginning of the sequence.
//     };
//
//     /**
//      * @brief Specifies the direction for truncation.
//      */
//     enum class TruncationDirection {
//         RIGHT, // Remove elements from the end of the sequence.
//         LEFT   // Remove elements from the beginning of the sequence.
//     };
//
//
// }