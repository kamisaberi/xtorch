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
#include "../include/models/cnn/lenet5.h"
#include "../include/definitions/transforms.h"
#include "../include/data-loaders/data-loader.h"


using namespace std;
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

int main() {
    std::vector<int64_t> size = {32, 32};

    std::cout.precision(10);
    torch::Device device(torch::kCPU);


    auto normalize_fn = torch::data::transforms::Normalize<>(0.5, 0.5);
    auto resize_fn  = xt::data::transforms::create_resize_transform({32,32});
    auto compose = xt::data::transforms::Compose({resize_fn, normalize_fn});
    auto dataset = xt::data::datasets::MNIST("/home/kami/Documents/temp/", DataMode::TRAIN, true,std::make_shared<xt::data::transforms::Compose>(compose) );
    auto transformed_dataset = dataset.map(torch::data::transforms::Stack<>());
//    auto transformed_dataset2 = dataset.map(torch::data::transforms::Stack<>());


    auto options = torch::data::DataLoaderOptions().batch_size(64).drop_last(false);
    // Instantiate CustomDataLoader with shuffle enabled
    xt::DataLoader loader(std::move(transformed_dataset), options, /*shuffle=*/true);

    torch::ext::models::LeNet5 model(10);
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    for (size_t epoch = 0; epoch != 10; ++epoch) {
        cout << "epoch: " << epoch << endl;
        for (auto& batch : loader) {
            // Each `batch` is an Example<Tensor, Tensor> with shapes [batch_size, ...]
            //                std::cout << "Batch data size: "    << batch.data.sizes() << ", Batch target size: " << batch.target.sizes() << "\n";
            //        std::cout << "Data:\n"    << batch.data << "\n";
            //        std::cout << "Targets:\n" << batch.target << "\n";
            //                std::cout << "-------------------------\n";
            torch::Tensor data, targets;
            data = batch.data;
            targets = batch.target;
            optimizer.zero_grad();
            torch::Tensor output;
            output = model.forward(data);
            torch::Tensor loss;
            loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();
            //                std::cout << "Epoch: " << epoch << " | Batch: " <<  " | Loss: " << loss.item<float>() <<                            std::endl;

            //            }

        }
    }



//    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
//        std::move(transformed_dataset2), 64);
//
//    torch::ext::models::LeNet5 model(10);
//    model.to(device);
//    model.train();
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
//    for (size_t epoch = 0; epoch != 10; ++epoch) {
//        size_t batch_index = 0;
//        auto train_loader_iterator = train_loader->begin();
//        auto train_loader_end = train_loader->end();
//        while (train_loader_iterator != train_loader_end) {
//            torch::Tensor data, targets;
//            auto batch = *train_loader_iterator;
//            data = batch.data;
//            targets = batch.target;
//            optimizer.zero_grad();
//            torch::Tensor output;
//            output = model.forward(data);
//            torch::Tensor loss;
//            loss = torch::nll_loss(output, targets);
//            loss.backward();
//            optimizer.step();
//            if (++batch_index % 100 == 0) {
//                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() <<
//                        std::endl;
//            }
//            ++train_loader_iterator;
//        }
//    }

    return 0;
}
