#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>

#define DEBUG_MODE true


using namespace std;

struct Net : torch::nn::Module {
    torch::nn::Sequential layer1 = nullptr, layer2 = nullptr;
    torch::nn::Linear fc1 = nullptr, fc2 = nullptr, fc3 = nullptr;

    Net(int num_classes) {
        layer1 = torch::nn::Sequential();
        torch::nn::Conv2d conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5).stride(1).padding(0));
        layer1->push_back(conv1);
        torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(6);
        layer1->push_back(batch1);
        torch::nn::ReLU relu1 = torch::nn::ReLU();
        layer1->push_back(relu1);
        torch::nn::MaxPool2d pool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer1->push_back(pool1);
        register_module("layer1", layer1);

        layer2 = torch::nn::Sequential();
        torch::nn::Conv2d conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5).stride(1).padding(0));
        layer2->push_back(conv2);
        torch::nn::BatchNorm2d batch2 = torch::nn::BatchNorm2d(16);
        layer2->push_back(batch2);
        torch::nn::ReLU relu2 = torch::nn::ReLU();
        layer2->push_back(relu2);
        torch::nn::MaxPool2d pool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer2->push_back(pool2);
        register_module("layer2", layer2);

        fc1 = torch::nn::Linear(400, 120);
        fc2 = torch::nn::Linear(120, 84);
        fc3 = torch::nn::Linear(84, num_classes);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = torch::relu(fc1->forward(x.view({-1, 400})));
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

};



void set_random()
{
    torch::manual_seed(1);
    torch::cuda::manual_seed_all(1);
    srand(1);
}



// Function to resize a single tensor
torch::Tensor resize_tensor(const torch::Tensor& tensor, const std::vector<int64_t>& size) {
    return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
    ).squeeze(0);
}

int main() {


    std::cout.precision(10);
    torch::Device device(torch::kCPU);

    // Load the MNIST dataset
    auto dataset = torch::data::datasets::MNIST("/home/kami/datasets/MNIST/raw");

    // Define the target size
    // std::vector<int64_t> size = {32, 32};

    // Create a lambda function for resizing
    auto resize_transform = torch::data::transforms::Lambda<torch::data::Example<>>(
            [](torch::data::Example<> example) {
                example.data = resize_tensor(example.data, {32, 32});
                return example;
            }
    );

    // Apply the resize transform to the dataset
    auto transformed_dataset = dataset.map(resize_transform).map(torch::data::transforms::Normalize<>(0.5,0.5)).map(torch::data::transforms::Stack<>());


    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(transformed_dataset), 64);


    Net model(10);
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    for (size_t epoch = 0; epoch != 10; ++epoch) {
        size_t batch_index = 0;

        auto train_loader_interator = train_loader->begin();
        auto train_loader_end = train_loader->end();

        while(train_loader_interator != train_loader_end) {
            torch::Tensor  data,targets;
            auto batch = *train_loader_interator;
            data = batch.data;
            targets = batch.target;
            optimizer.zero_grad();

            torch::Tensor output;
            output = model.forward(data);

            torch::Tensor loss;
            cout << output.sizes() << " " << targets.sizes() << endl;
            cout << targets << endl << endl << endl << endl;
            loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();

            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
            }
            ++train_loader_interator;

        }
    }




    // Print the size of the original and resized images
//    std::cout << "Original image size: " << train_loader[0].data.sizes() << std::endl;
//    std::cout << "Resized image size: " << transformed_dataset[0].data.sizes() << std::endl;

    return 0;
}
