#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <functional>
#include "../../include/datasets/image-classification/mnist.h"
#include "../../include/models/cnn/lenet5.h"
#include "../../include/definitions/transforms.h"
#include "../../include/data-loaders/data-loader.h"
#include "../../include/trainers/trainer.h"

using namespace std;

int main() {
    std::cout.precision(10);
    auto dataset = xt::data::datasets::MNIST(
        "/home/kami/Documents/temp/", DataMode::TRAIN, true,
        {
            xt::data::transforms::Resize({32, 32}),
            torch::data::transforms::Normalize<>(0.5, 0.5)
        }).map(torch::data::transforms::Stack<>());


    xt::DataLoader<decltype(dataset)> loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(64).drop_last(false),
        true);

    xt::models::LeNet5 model(10);
    model.to(torch::Device(torch::kCPU));
    model.train();

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    xt::Trainer trainer;
    trainer.set_optimizer(&optimizer)
            .set_max_epochs(5)
            .set_loss_fn([](auto output, auto target) {
                return torch::nll_loss(output, target);
            });

    trainer.fit<decltype(dataset)>(&model, loader);

    return 0;
}
