#include <iostream>
#include "../../include/datasets/computer_vision/image_classification/mnist.h"

using namespace std;

int main()
{
    std::cout.precision(10);

    auto dataset = xt::datasets::MNIST(
        "/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, true );

    // xt::DataLoader<decltype(dataset)> loader(
    //     std::move(dataset),
    //     torch::data::DataLoaderOptions().batch_size(64).drop_last(false),
    //     true);

    return 0;
}
