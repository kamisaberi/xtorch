#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#define DEBUG_MODE true

using namespace std;

class CIFAR10 : public torch::data::Dataset<CIFAR10> {
public:
    CIFAR10(const std::string &root) {
        // Load data from the specified root directory
        load_data(root);
    }

    // Override the get method to return a sample
    torch::data::Example<> get(size_t index) override {
        // Return the tensor image and its corresponding label
        return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    }

    // Override the size method to return the number of samples
    torch::optional<size_t> size() const override {
        return data.size();
    }

private:
    std::vector<torch::Tensor> data; // Store image data as tensors
    std::vector<int64_t> labels;      // Store labels

    void load_data(const std::string &root) {
        const int num_files = 5;
        for (int i = 1; i <= num_files; ++i) {
            std::string file_path = root + "/data_batch_" + std::to_string(i) + ".bin";
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                continue;
            }

            for (int j = 0; j < 10000; ++j) {
                uint8_t label;
                file.read(reinterpret_cast<char *>(&label), sizeof(label));
                labels.push_back(static_cast<int64_t>(label));

                std::vector<uint8_t> image(3072); // 32x32x3 = 3072
                file.read(reinterpret_cast<char *>(image.data()), image.size());

                // Reshape the image to 3x32x32 and convert to a Torch tensor
                auto tensor_image = torch::from_blob(image.data(), {3, 32, 32}, torch::kByte).clone(); // Clone to ensure memory management
                tensor_image = tensor_image.permute({0, 2, 1}); // Permute to get the correct order (C, H, W)

                data.push_back(tensor_image); // Store the tensor in the data vector
            }

            file.close();
        }
    }
};

struct Net : torch::nn::Module {
    torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr , layer4 = nullptr , layer5 = nullptr ;
    torch::nn::Sequential fc = nullptr , fc1 = nullptr , fc2 = nullptr;

    Net(int num_classes) {
        //TODO layer1
        layer1 = torch::nn::Sequential();
        //             nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        torch::nn::Conv2d conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 96, 11).stride(4).padding(0));
        layer1->push_back(conv1);
        //             nn.BatchNorm2d(96),
        torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(96);
        layer1->push_back(batch1);
        //             nn.ReLU(),
        torch::nn::ReLU relu1 = torch::nn::ReLU();
        layer1->push_back(relu1);
        //             nn.MaxPool2d(kernel_size = 3, stride = 2))
        torch::nn::MaxPool2d pool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        layer1->push_back(pool1);
        register_module("layer1", layer1);

        //TODO layer2
        layer2 = torch::nn::Sequential();
        //             nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        torch::nn::Conv2d conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 256, 5).stride(1).padding(2));
        layer2->push_back(conv2);
        //             nn.BatchNorm2d(256),
        torch::nn::BatchNorm2d batch2 = torch::nn::BatchNorm2d(256);
        layer2->push_back(batch2);
        //             nn.ReLU(),
        torch::nn::ReLU relu2 = torch::nn::ReLU();
        layer2->push_back(relu2);
        //             nn.MaxPool2d(kernel_size = 3, stride = 2))
        torch::nn::MaxPool2d pool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        layer2->push_back(pool2);
        register_module("layer2", layer2);

        //TODO layer3
        layer3 = torch::nn::Sequential();
        //             nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 384, 3).stride(1).padding(1));
        layer3->push_back(conv3);
        //             nn.BatchNorm2d(384),
        torch::nn::BatchNorm2d batch3 = torch::nn::BatchNorm2d(384);
        layer3->push_back(batch3);
        //             nn.ReLU())
        torch::nn::ReLU relu3 = torch::nn::ReLU();
        layer3->push_back(relu3);
        register_module("layer3", layer3);

        //TODO layer4
        layer4 = torch::nn::Sequential();
        //             nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 384, 3).stride(1).padding(1));
        layer4->push_back(conv4);
        //             nn.BatchNorm2d(384),
        torch::nn::BatchNorm2d batch4 = torch::nn::BatchNorm2d(384);
        layer4->push_back(batch4);
        //             nn.ReLU())
        torch::nn::ReLU relu4 = torch::nn::ReLU();
        layer4->push_back(relu4);
        register_module("layer4", layer4);

        //TODO layer5
        layer5 = torch::nn::Sequential();
        //             nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1));
        layer5->push_back(conv5);
        //             nn.BatchNorm2d(256),
        torch::nn::BatchNorm2d batch5 = torch::nn::BatchNorm2d(256);
        layer5->push_back(batch5);
        //             nn.ReLU(),
        torch::nn::ReLU relu5 = torch::nn::ReLU();
        layer5->push_back(relu5);
        //             nn.MaxPool2d(kernel_size = 3, stride = 2))
        torch::nn::MaxPool2d pool5 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        layer5->push_back(pool5);
        register_module("layer5", layer5);

        //TODO fc
        fc = torch::nn::Sequential();
        //             nn.Dropout(0.5),
        torch::nn::Dropout drop10 = torch::nn::Dropout(0.5);
        fc->push_back(drop10);
        //             nn.Linear(9216, 4096),
        torch::nn::Linear linear10 = torch::nn::Linear(9216, 4096);
        fc->push_back(linear10);
        //             nn.ReLU())
        torch::nn::ReLU relu10 = torch::nn::ReLU();
        fc->push_back(relu10);
        register_module("fc", fc);


        //TODO fc1
        fc1 = torch::nn::Sequential();
        //             nn.Dropout(0.5),
        torch::nn::Dropout drop11 = torch::nn::Dropout(0.5);
        fc1->push_back(drop11);
        //             nn.Linear(4096, 4096),
        torch::nn::Linear linear11 = torch::nn::Linear(4096, 4096);
        fc1->push_back(linear11);
        //             nn.ReLU())
        torch::nn::ReLU relu11 = torch::nn::ReLU();
        fc1->push_back(relu11);
        register_module("fc1", fc1);

        //TODO fc2
        fc2 = torch::nn::Sequential();
        //             nn.Linear(4096, num_classes))
        torch::nn::Linear linear12 = torch::nn::Linear(4096, num_classes);
        fc2->push_back(linear12);
        register_module("fc2", fc2);

        //TODO DONE

    }

    torch::Tensor forward(torch::Tensor x) {
//        cout << "size 01: " << x.sizes()<< endl;
        x = layer1->forward(x);
//        cout << "size 02: " << x.sizes()<< endl;
        x = layer2->forward(x);
//        cout << "size 03: " << x.sizes()<< endl;
        x = layer3->forward(x);
//        cout << "size 04: " << x.sizes()<< endl;
        x = layer4->forward(x);
//        cout << "size 05: " << x.sizes()<< endl;
        x = layer5->forward(x);
//        cout << "size 06: " << x.sizes()<< endl;
        x = x.view({x.size(0),-1});
//        cout << "size 07: " << x.sizes()<< endl;
        x = fc->forward(x);
//        cout << "size 08: " << x.sizes()<< endl;
        x = fc1->forward(x);
//        cout << "size 09: " << x.sizes()<< endl;
        x = fc2->forward(x);
//        cout << "size 10: " << x.sizes()<< endl;
        return  x;
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
//    cout << "inside 01: " <<  tensor.sizes() << endl;
//    auto  t1 = tensor.unsqueeze(0);
//    cout << "inside 02: " <<  t1.sizes() << endl;
//    auto  t2 = t1.squeeze(0);
//    cout << "inside 02: " <<  t2.sizes() << endl;
    return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
    ).squeeze(0);
}

//torch::Tensor resize_tensor(const torch::Tensor& tensor, const std::vector<int64_t>& size) {
//    // Check if the tensor is in the format [C, H, W]
//    cout << "inside: " <<  tensor.sizes() << endl;
//    TORCH_CHECK(tensor.size(0) == 3, "Input tensor must have 3 channels for RGB.");
//
//    auto out =  torch::nn::functional::interpolate(
//            tensor.unsqueeze(0), // Add a batch dimension
//            torch::nn::functional::InterpolateFuncOptions()
//                    .size(size)
//                    .mode(torch::kBilinear)
//                    .align_corners(false)
//    ).squeeze(0); // Remove the batch dimension
//
//    cout << "inside: " <<  out.sizes() << endl;
//    return  out;
//
//}

int main() {
    std::string dataset_path = "/home/kami/datasets/cifar-10-batches-bin/"; // Update with your path
    std::cout.precision(10);
    torch::Device device(torch::kCPU);

    auto resize_transform = torch::data::transforms::Lambda<torch::data::Example<>>(
            [](torch::data::Example<> example) {
                example.data = resize_tensor(example.data, {227, 227});
                return example;
            }
    );

    CIFAR10 cifar10(dataset_path);

    // Create a data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(cifar10.map(resize_transform).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(
                    torch::data::transforms::Stack<>())),
            /*batch_size=*/64);


    Net model(10);
    model.to(device);
    model.train();

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion;



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

//            cout << output.sizes() << " " << targets.sizes() << endl;
//            cout << targets << endl;
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

