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


    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}));
    transform_list.push_back(std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5}));

    auto compose= std::make_unique<xt::transforms::Compose>(transform_list);

    auto dataset = xt::datasets::MNIST(
        "/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false, std::move(compose));


    auto datum = dataset.get(0);
    cout << datum.data.sizes() << endl;
    // xt::DataLoader<decltype(dataset)> loader(
    //     std::move(dataset),
    //     torch::data::DataLoaderOptions().batch_size(64).drop_last(false),
    //     true);

    return 0;
}
