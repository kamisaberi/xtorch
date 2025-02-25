#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include <torch/data/transforms/base.h>
#include <functional>

namespace torch::ext::data::transforms {


    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size) ;

    torch::data::transforms::Lambda<torch::data::Example<>> resize(std::vector<int64_t> size) ;

    torch::data::transforms::Lambda<torch::data::Example<>> normalize(double mean , double stddev) ;




}