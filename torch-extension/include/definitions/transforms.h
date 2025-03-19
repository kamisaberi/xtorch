#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include <torch/data/transforms/base.h>
#include <functional>

namespace torch::ext::data::transforms {
    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size);
    torch::Tensor pad_tensor(const torch::Tensor &tensor, const int size);
    torch::Tensor grayscale_image(const torch::Tensor &tensor);
    torch::Tensor  grayscale_to_rgb(const torch::Tensor &tensor);


    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size);
    torch::data::transforms::Lambda<torch::data::Example<> > pad(int size);
    torch::data::transforms::Lambda<torch::data::Example<> > grayscale();

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev);

}
