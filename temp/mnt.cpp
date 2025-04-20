#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include "../include/datasets/mnist.h"
#include "../include/models/cnn/lenet/lenet5.h"
#include <torch/data/transforms/base.h>
#include <functional>
#include "../include/definitions/transforms.h"


using namespace std;
//template <typename Input, typename Output>
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

void set_random() {
    torch::manual_seed(1);
    torch::cuda::manual_seed_all(1);
    srand(1);
}

// Function to resize a single tensor
// torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size) {
//     return torch::nn::functional::interpolate(
//         tensor.unsqueeze(0),
//         torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
//     ).squeeze(0);
// }
//
// torch::data::transforms::Lambda<torch::data::Example<>> resize(std::vector<int64_t> size) {
//     return torch::data::transforms::Lambda<torch::data::Example<> >(
//         [size](torch::data::Example<> example) {
//             example.data = resize_tensor(example.data, size);
//             return example;
//         }
//     );
// }


// auto normalize_transform = torch::data::transforms::Lambda<torch::data::Example<> >(
//     [](torch::data::Example<> example) {
//         example.data = torch::data::transforms::Normalize<>(0.5, 0.5)(example.data);
//         return example;
//     }
// );
//
//
// torch::data::transforms::Lambda<torch::data::Example<>> normalize(double mean , double stddev) {
//     return torch::data::transforms::Lambda<torch::data::Example<> >(
//         [mean, stddev](torch::data::Example<> example) {
//             example.data = torch::data::transforms::Normalize<>(mean, stddev)(example.data);
//             return example;
//         }
//     );
// }


int main() {
    cout << "mnt 01\n";
    std::vector<int64_t> size = {32, 32};

    std::cout.precision(10);
    torch::Device device(torch::kCPU);

    auto dataset = xt::data::datasets::MNIST("/home/kami/Documents/temp/", DataMode::TRAIN, true);

    cout << "DATASET\n";
    cout << dataset.get(0).data << "  " << dataset.get(0).target << "\n";

    //
    auto transformed_dataset = dataset
            .map(xt::data::transforms::resize({32, 32}))
            .map(xt::data::transforms::normalize(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());

    cout << "TRANSFORMED DATASET\n";
    cout << transformed_dataset.get_batch(0).data << "  " << transformed_dataset.get_batch(0).target << "\n";


    //TODO Custom Dataset START
    vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms = {
        xt::data::transforms::resize({32, 32}),
        xt::data::transforms::normalize(0.5, 0.5)
    };
    auto dataset2 = xt::data::datasets::MNIST("/home/kami/Documents/temp/",
                                                      {
                                                          .mode = DataMode::TRAIN, .download = true,
                                                          .transforms = transforms
                                                      });
    // cout << "mnt 03:" << dataset2.size().value() << "\n";
    // auto dataset3 =dataset2.map(torch::data::transforms::Stack<>());
    //TODO Custom Dataset FINISH


    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(transformed_dataset), 64);

    cout << "mnt 04\n";

    xt::models::LeNet5 model(10);
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    cout << "mnt 05\n";
    for (size_t epoch = 0; epoch != 10; ++epoch) {
        size_t batch_index = 0;
        auto train_loader_interator = train_loader->begin();
        auto train_loader_end = train_loader->end();
        while (train_loader_interator != train_loader_end) {
            torch::Tensor data, targets;
            auto batch = *train_loader_interator;
            data = batch.data;
            targets = batch.target;
            optimizer.zero_grad();
            torch::Tensor output;
            output = model.forward(data);
            torch::Tensor loss;
            cout << "output: " << output.sizes() << endl;
            cout << "targets: " << targets.sizes() << endl;
            loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() <<
                        std::endl;
            }
            ++train_loader_interator;
        }
    }

    cout << "mnt 06\n";


    return 0;
}
