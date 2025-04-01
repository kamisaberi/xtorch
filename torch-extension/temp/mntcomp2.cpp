#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include "../include/datasets/mnist.h"
#include "../include/models/cnn/lenet5.h"
#include <torch/data/transforms/base.h>
#include <functional>
#include "../include/definitions/transforms.h"

#include "../include/datasets/mnist.h"
#include "../include/datasets/cifar.h"
#include "../include/datasets/imagenette.h"
#include "../include/models/cnn/lenet5.h"
#include "../include/definitions/transforms.h"
#include "../include/data-loaders/data-loader.h"
#include "../include/trainers/trainer.h"
#include <type_traits>
#include <iostream>

using namespace std;
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

template <typename Dataset>
void check_dataset_type(const Dataset& dataset) {
    if constexpr (std::is_same_v<Dataset, xt::data::datasets::BaseDataset>) {
        std::cout << "The object is a MNIST dataset" << std::endl;
    } else if constexpr (std::is_same_v<Dataset, torch::data::datasets::MapDataset<xt::data::datasets::BaseDataset, torch::data::transforms::Stack<>>>) {
        std::cout << "The object is a transformed MNIST dataset" << std::endl;
    } else {
        std::cout << "The object is of an unknown type" << std::endl;
    }
}

int main() {
    std::vector<int64_t> size = {32, 32};
    std::cout.precision(10);
    torch::Device device(torch::kCPU);

    auto dataset = xt::data::datasets::MNIST("/home/kami/Documents/temp/", DataMode::TRAIN, true,
                                             {
                                                 xt::data::transforms::create_resize_transform(size),
                                                 torch::data::transforms::Normalize<>(0.5, 0.5)
                                             });
    auto transformed_dataset =  dataset.map(torch::data::transforms::Stack<>());


    check_dataset_type(dataset);
    check_dataset_type(transformed_dataset);

    auto dataseti = xt::data::datasets::Imagenette("/home/kami/Documents/temp/", DataMode::TRAIN, true , xt::data::datasets::ImageType::PX160);

    cout << "MNIST dataset size: " << endl;
    auto transformed_dataseti = dataseti.map(torch::data::transforms::Stack<>());

    cout << "MNIST dataset size: " << endl;
    check_dataset_type(dataseti);
    check_dataset_type(transformed_dataseti);


    return 0;

    xt::DataLoader<decltype(transformed_dataset)> loader(std::move(transformed_dataset), torch::data::DataLoaderOptions().batch_size(64).drop_last(false), /*shuffle=*/true);
//    xt::DataLoader<decltype(dataset)> loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(64).drop_last(false), /*shuffle=*/true);



    xt::models::LeNet5 model(10);
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    xt::Trainer trainer;
    trainer.set_optimizer(&optimizer)
        .set_max_epochs(5)
        .set_loss_fn([](auto output, auto target) {return torch::nll_loss(output, target);});
//    trainer.set

    trainer.fit<decltype(transformed_dataset)>(&model ,loader );

//    for (size_t epoch = 0; epoch != 10; ++epoch) {
//        cout << "epoch: " << epoch << endl;
//        for (auto& batch : loader) {
//
//            torch::Tensor data, targets;
//            data = batch.data;
//            targets = batch.target;
//            optimizer.zero_grad();
//            torch::Tensor output;
//            output = model.forward(data);
//            torch::Tensor loss;
//            loss = torch::nll_loss(output, targets);
//            loss.backward();
//            optimizer.step();
//            //                std::cout << "Epoch: " << epoch << " | Batch: " <<  " | Loss: " << loss.item<float>() <<                            std::endl;
//
//            //            }
//
//        }
//    }
    return 0;
}
