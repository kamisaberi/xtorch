#include <iostream>
#include "../../include/datasets/computer_vision/image_classification/mnist.h"

using namespace std;

int main()
{
    std::cout.precision(10);

    // xt::transforms::Compose compose({
    //     xt::transforms::image::Resize({32, 32}),
    //     xt::transforms::general::Normalize({0.5}, {0.5})
    // });


    auto dataset = xt::datasets::MNIST(
        "/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false);


    auto datum = dataset.get(0);
    cout << datum.data.sizes() << endl;
    // xt::DataLoader<decltype(dataset)> loader(
    //     std::move(dataset),
    //     torch::data::DataLoaderOptions().batch_size(64).drop_last(false),
    //     true);

    return 0;
}
