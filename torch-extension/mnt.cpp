#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include "include/datasets/mnist.h"
#include "include/models/cnn/lenet5.h"
#include <torch/data/transforms/base.h>
#include <functional>
#include "include/definitions/transforms.h"


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
    std::vector<int64_t> size = {32, 32};


    // auto resize_transform = torch::data::transforms::Lambda<torch::data::Example<> >(
    //     [size](torch::data::Example<> example) {
    //         example.data = resize_tensor(example.data, size);
    //         return example;
    //     }
    // );


    std::cout.precision(10);
    torch::Device device(torch::kCPU);
    // Load the MNIST dataset


    // std::vector<std::shared_ptr<torch::data::transforms::Lambda<torch::data::Example<> > > > transforms;
    // transforms.push_back(std::make_shared<normalize_transform>());
    //    std::vector<std::function<torch::data::Example<>>> transforms;
    //    std::vector<torch::data::transforms::Transform<Input, Output>> transforms;
    //    transforms.push_back(torch::data::transforms::Normalize<>(0.5,0.5));
    //    transforms.push_back(resize_transform);

    // auto dataset = torch::ext::data::datasets::MNIST("/home/kami/Documents/temp/",
    //                                                  {.mode = DataMode::TRAIN, .download = true});
    auto dataset = torch::ext::data::datasets::MNIST("/home/kami/Documents/temp/", DataMode::TRAIN, true);

    // cout << typeid(dataset).name() << endl;

    // Apply the resize transform to the dataset
    auto transformed_dataset = dataset.map(torch::ext::data::transforms::resize({32, 32}))
            .map(torch::ext::data::transforms::normalize(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());


    // auto sssss = transformed_dataset.size();
    // cout << sssss.value() << endl;

    // cout << typeid(transformed_dataset).name() << endl;
    // auto ss =  transformed_dataset.get_batch(0).data.sizes();
    // cout << "SIZES : " << ss << endl;
    // auto sss = transformed_dataset.dataset().size();
    // cout << "SIZES : " << sss.value() << endl;


    vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms = {
        torch::ext::data::transforms::resize({32, 32}),
        torch::ext::data::transforms::normalize(0.5, 0.5)

    };

    // cout << "MNIST training data size: "  << endl;

    auto dataset2 = torch::ext::data::datasets::MNIST("/home/kami/Documents/temp/",
                                                 {.mode = DataMode::TRAIN, .download = true , .transforms = transforms});


    cout << "r1" << endl;

    // for (const auto &transform: transforms) {
    //     auto t  = dataset.map(transform);
    //     cout << t.get_batch(0).data()->data[0] << endl;
    // }
    // return 0;


    // cout << dataset.get(0).data << endl;
    // dataset = dataset.map(torch::data::transforms::Stack<>());
    // cout << dataset.get(0).data << endl;

    // auto ld = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), 64);
    // for (size_t epoch = 0; epoch != 10; ++epoch) {
    //     size_t batch_index = 0;
    //     auto ld_iter = ld->begin();
    //     auto ld_end = ld->end();
    //     while (ld_iter != ld_end) {
    //         torch::Tensor data, targets;
    //         auto batch = *ld_iter;
    //         data = batch[0].data;
    //         targets = batch[0].target;
    //         cout << data << endl;
    //         return 0;
    //
    //         // auto targets = batch.target;
    //     }
    // }


    // auto rs = torch::ext::data::transforms::resize({32, 32});
    // torch::data::datasets::MapDataset<torch::ext::data::datasets::MNIST, torch::data::transforms::Lambda<
    //     torch::data::Example<> > > dataset1 = dataset.map(transforms[0]);
    // for (int i = 1 ; i < transforms.size();i++) {
    //     dataset1 = dataset.map(transforms[i]);
    // }
    // cout << dataset1.size().value() << endl;

    // cout << transformed_dataset2.get_batch(0).data << endl;

    cout << "r2" << endl;
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(transformed_dataset), 64);

    cout << "r3" << endl;

    torch::ext::models::LeNet5 model(10);
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    cout << "r4" << endl;

    for (size_t epoch = 0; epoch != 10; ++epoch) {
        cout << "r5.1" << endl;
        size_t batch_index = 0;
        cout << "r5.2" << endl;
        auto train_loader_interator = train_loader->begin();
        cout << "r5.3" << endl;
        auto train_loader_end = train_loader->end();
        cout << "r5.4" << endl;
        while (train_loader_interator != train_loader_end) {
            cout << "r5.4.1" << endl;
            torch::Tensor data, targets;
            auto batch = *train_loader_interator;
            data = batch.data;
            targets = batch.target;
            cout << "r5.4.2" << endl;
            optimizer.zero_grad();
            torch::Tensor output;
            output = model.forward(data);
            cout << "r5.4.4" << endl;
            torch::Tensor loss;
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
    return 0;
}
