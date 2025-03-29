#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include <torch/data/transforms/base.h>
#include <functional>

namespace xt::data::transforms {
    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size);

    torch::Tensor pad_tensor(const torch::Tensor &tensor, const int size);

    torch::Tensor grayscale_image(const torch::Tensor &tensor);

    torch::Tensor grayscale_to_rgb(const torch::Tensor &tensor);


    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size);

    torch::data::transforms::Lambda<torch::data::Example<> > pad(int size);

    torch::data::transforms::Lambda<torch::data::Example<> > grayscale();

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev);

    torch::data::transforms::Lambda<torch::data::Example<> > grayscaleToRGB();

    class Compose {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        Compose();
        Compose(std::vector<TransformFunc> transforms);

        torch::Tensor operator()(torch::Tensor input) const;

    private:
        std::vector<TransformFunc> transforms;
    };


    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size);


}
